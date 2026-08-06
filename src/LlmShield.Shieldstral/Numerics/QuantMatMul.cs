// Weight-stationary GEMM against GGUF-quantized weights.
using System.Buffers;
using System.Numerics.Tensors;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Quantization;

namespace LlmShield.Shieldstral.Numerics;

/// <summary>How <see cref="QuantMatMul"/> should evaluate a product.</summary>
public enum MatMulStrategy
{
    /// <summary>Integer where the weight type has a kernel for it, float otherwise.</summary>
    Auto,
    /// <summary>Always decode weights to float32 and use an FMA dot.</summary>
    Float,
    /// <summary>Always use the integer path, throwing if the weight type has no kernel.</summary>
    Integer,
}

/// <summary>
/// <c>y = x · Wᵀ</c> where W stays in its GGUF quantization and x is float32.
///
/// The loop order is the whole design. Weights dominate both the memory traffic
/// (a 3B checkpoint is gigabytes; the activations are megabytes) and the decode
/// cost, so each weight row is touched exactly once per call and decoded into a
/// scratch buffer that stays in L1 while every token in the current tile dots
/// against it. Tiling the tokens keeps that activation slab inside L2, which is
/// what stops a 150-token prefill from re-streaming the activations once per
/// output row.
///
/// The row loop is the only place this library spends more than one core, so the
/// caller's <see cref="ParallelOptions"/> is what bounds the whole runtime's CPU
/// use — see <see cref="ForwardAsync(WeightMatrix, ReadOnlyMemory{float}, int, Memory{float}, ParallelOptions)"/>.
/// </summary>
public static class QuantMatMul
{
    /// <summary>
    /// Which arithmetic to use. Process-wide, and intended for benchmarking and
    /// tests — production leaves it at <see cref="MatMulStrategy.Auto"/>.
    /// </summary>
    public static MatMulStrategy Strategy { get; set; } = MatMulStrategy.Auto;

    /// <summary>Bytes of activations to keep resident per tile — sized for a typical L3.</summary>
    private const int TokenTileBytes = 4 * 1024 * 1024;

    /// <summary>Rows handed to one worker at a time; large enough to amortise the scratch rent.</summary>
    private const int RowChunk = 64;

    /// <summary>
    /// Computes <paramref name="destination"/>[t, r] = dot(x[t, :], W[r, :]).
    /// <paramref name="x"/> is <paramref name="tokens"/> × <c>W.Cols</c> row-major;
    /// <paramref name="destination"/> is <paramref name="tokens"/> × <c>W.Rows</c>.
    /// <para>
    /// <paramref name="options"/> bounds the fan-out over rows and carries the
    /// cancellation token; its <see cref="ParallelOptions.MaxDegreeOfParallelism"/>
    /// is the knob a host uses to leave cores for everything else it is running.
    /// </para>
    /// </summary>
    public static async ValueTask ForwardAsync(
        WeightMatrix          w,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
    {
        ArgumentNullException.ThrowIfNull(options);

        int cols = w.Cols, rows = w.Rows;

        if (x.Length < (long)tokens * cols)
            throw new ArgumentException($"Expected {(long)tokens * cols} activations, got {x.Length}.", nameof(x));
        if (destination.Length < (long)tokens * rows)
            throw new ArgumentException($"Expected room for {(long)tokens * rows} outputs, got {destination.Length}.", nameof(destination));

        bool useInteger = Strategy switch
        {
            MatMulStrategy.Float => false,
            MatMulStrategy.Integer => true,
            _ => IntegerDot.Supports(w.Type) && cols % IntegerDot.BlockSize == 0,
        };

        if (useInteger)
        {
            await ForwardIntegerAsync(w, x, tokens, destination, options).ConfigureAwait(false);
            return;
        }

        int tile = Math.Clamp(TokenTileBytes / (cols * sizeof(float)), 1, tokens);
        int chunks = (rows + RowChunk - 1) / RowChunk;

        for (int t0 = 0; t0 < tokens; t0 += tile)
        {
            int tokenStart = t0;
            int tileLen = Math.Min(tile, tokens - t0);

            // One token is a pure memory walk over the weights; splitting it
            // across cores still helps, so parallelise on rows either way.
            await Parallel.ForAsync(0, chunks, options, (chunk, _) =>
            {
                int r0 = chunk * RowChunk;
                int r1 = Math.Min(r0 + RowChunk, rows);
                RunRowsFloat(w, x.Span, destination.Span, tokenStart, tileLen, r0, r1, cols, rows);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Single-token convenience overload — the decode path, where the whole
    /// forward pass is one row of activations.
    /// </summary>
    public static ValueTask ForwardAsync(
        WeightMatrix w, ReadOnlyMemory<float> x, Memory<float> destination, ParallelOptions options)
        => ForwardAsync(w, x, tokens: 1, destination, options);

    private static unsafe void RunRowsFloat(
        WeightMatrix w, ReadOnlySpan<float> x, Span<float> y,
        int tokenStart, int tokenCount, int rowStart, int rowEnd, int cols, int rows)
    {
        bool needsDecode = w.Type != GgmlType.F32;
        float[]? rented = needsDecode ? ArrayPool<float>.Shared.Rent(cols) : null;
        try
        {
            fixed (float* scratch = rented)
            {
                for (int r = rowStart; r < rowEnd; r++)
                {
                    float* weightRow;
                    if (needsDecode)
                    {
                        Dequantizer.Dequantize(w.Type, w.Row(r), scratch, cols);
                        weightRow = scratch;
                    }
                    else
                    {
                        weightRow = (float*)w.Row(r);
                    }

                    var wSpan = new ReadOnlySpan<float>(weightRow, cols);
                    for (int t = 0; t < tokenCount; t++)
                    {
                        var xSpan = x.Slice((tokenStart + t) * cols, cols);
                        y[(tokenStart + t) * rows + r] = TensorPrimitives.Dot(xSpan, wSpan);
                    }
                }
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Integer path: quantize the activations to Q8_0 once, then dot in 8-bit.
    ///
    /// The activations are quantized for the whole call rather than per tile —
    /// they are two orders of magnitude smaller than the weights, so there is
    /// nothing to gain from tiling them here, and one pass keeps the code honest
    /// about where the cost actually is.
    /// </summary>
    private static async ValueTask ForwardIntegerAsync(
        WeightMatrix          w,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
    {
        int cols = w.Cols, rows = w.Rows;
        int quantizedRowBytes = cols / IntegerDot.BlockSize * IntegerDot.ActivationBlockBytes;

        int blocks = cols / IntegerDot.BlockSize;
        bool hasOffset = IntegerDot.HasOffset(w.Type);

        byte[] quantized = ArrayPool<byte>.Shared.Rent(quantizedRowBytes * tokens);
        float[] activationSums = ArrayPool<float>.Shared.Rent(blocks * tokens);
        try
        {
            QuantizeActivations(x.Span, quantized, activationSums, tokens, cols, blocks, quantizedRowBytes, hasOffset);

            int chunks = (rows + RowChunk - 1) / RowChunk;

            await Parallel.ForAsync(0, chunks, options, (chunk, _) =>
            {
                int r0 = chunk * RowChunk;
                int r1 = Math.Min(r0 + RowChunk, rows);
                RunRowsInteger(w, quantized, activationSums, destination.Span,
                               tokens, r0, r1, cols, rows, blocks, quantizedRowBytes, hasOffset);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(quantized);
            ArrayPool<float>.Shared.Return(activationSums);
        }
    }

    private static void QuantizeActivations(
        ReadOnlySpan<float> x, byte[] quantized, float[] activationSums,
        int tokens, int cols, int blocks, int quantizedRowBytes, bool hasOffset)
    {
        for (int t = 0; t < tokens; t++)
        {
            IntegerDot.QuantizeActivations(
                x.Slice(t * cols, cols),
                quantized.AsSpan(t * quantizedRowBytes, quantizedRowBytes),
                hasOffset ? activationSums.AsSpan(t * blocks, blocks) : default);
        }
    }

    private static unsafe void RunRowsInteger(
        WeightMatrix w, byte[] quantized, float[] activationSums, Span<float> y,
        int tokens, int rowStart, int rowEnd, int cols, int rows,
        int blocks, int quantizedRowBytes, bool hasOffset)
    {
        // Unpack once per row, dot once per (row, token) — the same
        // amortisation the float path gets from decoding a row once.
        sbyte[] values = ArrayPool<sbyte>.Shared.Rent(cols);
        float[] scales = ArrayPool<float>.Shared.Rent(blocks);
        float[] offsets = ArrayPool<float>.Shared.Rent(blocks);
        try
        {
            fixed (byte* activations = quantized)
            fixed (float* sums = activationSums)
            fixed (sbyte* v = values)
            fixed (float* s = scales)
            fixed (float* o = offsets)
            {
                float* offsetPtr = hasOffset ? o : null;
                bool packedQ8 = w.Type == GgmlType.Q8_0;

                for (int r = rowStart; r < rowEnd; r++)
                {
                    byte* row = w.Row(r);
                    if (packedQ8)
                    {
                        for (int t = 0; t < tokens; t++)
                            y[t * rows + r] = IntegerDot.DotPackedQ8_0(
                                row, activations + (long)t * quantizedRowBytes, blocks);
                        continue;
                    }

                    IntegerDot.UnpackRow(w.Type, row, v, s, offsetPtr, blocks);
                    for (int t = 0; t < tokens; t++)
                    {
                        float value = IntegerDot.DotUnpacked(
                            v, s, activations + (long)t * quantizedRowBytes, blocks);
                        if (offsetPtr is not null)
                            value += IntegerDot.OffsetContribution(
                                offsetPtr, sums + (long)t * blocks, blocks);
                        y[t * rows + r] = value;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(values);
            ArrayPool<float>.Shared.Return(scales);
            ArrayPool<float>.Shared.Return(offsets);
        }
    }

    /// <summary>
    /// Gathers embedding rows for <paramref name="tokenIds"/> out of a quantized
    /// embedding table. The table is the single largest tensor in the checkpoint
    /// (131072 × 3072), so only the rows actually referenced get decoded.
    /// </summary>
    public static void GatherRows(in WeightMatrix table, ReadOnlySpan<int> tokenIds, Span<float> destination)
    {
        int cols = table.Cols;
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if ((uint)id >= (uint)table.Rows)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token id {id} is outside the {table.Rows}-entry embedding table.");
            table.DequantizeRow(id, destination.Slice(i * cols, cols));
        }
    }
}
