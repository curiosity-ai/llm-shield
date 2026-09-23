// Weight-stationary GEMM against GGUF-quantized weights.
using System.Buffers;
using System.Numerics;
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
/// cost, so each weight row is unpacked exactly once per call, into a scratch
/// buffer that stays in cache while every token sweeps past it. What happens
/// then is a register-blocked GEMM rather than a dot per output: on the integer
/// path <see cref="IntegerGemm"/> takes eight rows against four tokens per
/// kernel call, on the float path two rows against four. Either way one load
/// feeds several multiply-adds, which is what a dot product per (row, token)
/// cannot do — it streams both operands for every output and stalls on load
/// bandwidth long before it runs out of arithmetic.
///
/// Every output is computed by the same sequence of operations whichever tile
/// or tail kernel it lands in, so a token's result does not depend on the batch
/// around it and <see cref="ForwardRowsAsync"/> can compute a subset of rows
/// bit-identically. The prefix cache and the verdict-only LM head rely on both.
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

    /// <summary>
    /// Bytes of float activations to keep resident per tile in the float path —
    /// sized to stay inside a core's L2 alongside the decoded weight rows, since
    /// every row pair streams the whole tile past its accumulators.
    /// </summary>
    private const int TokenTileBytes = 1024 * 1024;

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
    public static ValueTask ForwardAsync(
        WeightMatrix          w,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
        => ForwardCoreAsync(w, rowIds: default, mapped: false, x, tokens, destination, options);

    /// <summary>
    /// Single-token convenience overload — the decode path, where the whole
    /// forward pass is one row of activations.
    /// </summary>
    public static ValueTask ForwardAsync(
        WeightMatrix w, ReadOnlyMemory<float> x, Memory<float> destination, ParallelOptions options)
        => ForwardAsync(w, x, tokens: 1, destination, options);

    /// <summary>
    /// Like <see cref="ForwardAsync(WeightMatrix, ReadOnlyMemory{float}, int, Memory{float}, ParallelOptions)"/>
    /// but only for the weight rows in <paramref name="rowIds"/>: <paramref name="destination"/>
    /// is <paramref name="tokens"/> × <c>rowIds.Length</c>, column <c>i</c> holding row <c>rowIds[i]</c>.
    /// <para>
    /// Each output is computed by exactly the arithmetic the full product would use
    /// for it, so the values are bit-identical to the matching columns of a full
    /// call. That is what lets a caller that only reads a handful of logits skip
    /// the other 131 thousand without changing the answer.
    /// </para>
    /// </summary>
    public static ValueTask ForwardRowsAsync(
        WeightMatrix          w,
        ReadOnlyMemory<int>   rowIds,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
    {
        foreach (int id in rowIds.Span)
            if ((uint)id >= (uint)w.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowIds), $"Row {id} is outside the matrix's {w.Rows} rows.");
        return ForwardCoreAsync(w, rowIds, mapped: true, x, tokens, destination, options);
    }

    private static async ValueTask ForwardCoreAsync(
        WeightMatrix          w,
        ReadOnlyMemory<int>   rowIds,
        bool                  mapped,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
    {
        ArgumentNullException.ThrowIfNull(options);

        int cols = w.Cols, rows = mapped ? rowIds.Length : w.Rows;

        if (x.Length < (long)tokens * cols)
            throw new ArgumentException($"Expected {(long)tokens * cols} activations, got {x.Length}.", nameof(x));
        if (destination.Length < (long)tokens * rows)
            throw new ArgumentException($"Expected room for {(long)tokens * rows} outputs, got {destination.Length}.", nameof(destination));
        if (rows == 0 || tokens == 0) return;

        bool useInteger = Strategy switch
        {
            MatMulStrategy.Float => false,
            MatMulStrategy.Integer => true,
            _ => IntegerDot.Supports(w.Type) && cols % IntegerDot.BlockSize == 0,
        };

        if (useInteger)
        {
            if (IntegerGemm.Supports(w.Type) && cols % IntegerGemm.BlockSize == 0)
                await ForwardGemmAsync(w, rowIds, mapped, x, tokens, destination, options).ConfigureAwait(false);
            else
                await ForwardIntegerAsync(w, rowIds, mapped, x, tokens, destination, options).ConfigureAwait(false);
            return;
        }

        int tile = Math.Clamp(TokenTileBytes / (cols * sizeof(float)), FloatTokens, Math.Max(tokens, FloatTokens));
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
                RunRowsFloat(w, rowIds.Span, mapped, x.Span, destination.Span, tokenStart, tileLen, r0, r1, cols, rows);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
    }

    private static int SourceRow(ReadOnlySpan<int> rowIds, bool mapped, int r) => mapped ? rowIds[r] : r;

    // ------------------------------------------------------------ float path

    /// <summary>Weight rows decoded and dotted together in the float path.</summary>
    private const int FloatRows = 2;
    /// <summary>Tokens dotted against each decoded row pair at once.</summary>
    private const int FloatTokens = 4;

    /// <summary>
    /// Decodes two weight rows at a time and dots them against four tokens at a
    /// time: eight accumulators fed by six loads, where a plain dot is one
    /// accumulator fed by two. The single-row and single-token tails reuse
    /// <see cref="Dot"/>, whose per-output arithmetic is identical, so an output
    /// does not depend on which tile it fell into.
    /// </summary>
    private static unsafe void RunRowsFloat(
        WeightMatrix w, ReadOnlySpan<int> rowIds, bool mapped, ReadOnlySpan<float> x, Span<float> y,
        int tokenStart, int tokenCount, int rowStart, int rowEnd, int cols, int rows)
    {
        bool needsDecode = w.Type != GgmlType.F32;
        float[]? rented = needsDecode ? ArrayPool<float>.Shared.Rent(FloatRows * cols) : null;
        try
        {
            fixed (float* scratch = rented)
            fixed (float* xBase = x)
            fixed (float* yBase = y)
            {
                int tokenEnd = tokenStart + tokenCount;
                for (int r = rowStart; r < rowEnd; r += FloatRows)
                {
                    bool pair = r + 1 < rowEnd;
                    float* w0 = WeightRow(w, SourceRow(rowIds, mapped, r), scratch, cols, needsDecode);
                    float* w1 = pair ? WeightRow(w, SourceRow(rowIds, mapped, r + 1), scratch + cols, cols, needsDecode) : null;

                    int t = tokenStart;
                    if (pair)
                    {
                        for (; t + FloatTokens <= tokenEnd; t += FloatTokens)
                            Dot2x4(w0, w1, xBase + (long)t * cols, cols, yBase + (long)t * rows + r, rows);
                    }
                    for (; t < tokenEnd; t++)
                    {
                        float* xt = xBase + (long)t * cols;
                        yBase[(long)t * rows + r] = Dot(w0, xt, cols);
                        if (pair) yBase[(long)t * rows + r + 1] = Dot(w1, xt, cols);
                    }
                }
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static unsafe float* WeightRow(WeightMatrix w, int row, float* scratch, int cols, bool needsDecode)
    {
        if (!needsDecode) return (float*)w.Row(row);
        Dequantizer.Dequantize(w.Type, w.Row(row), scratch, cols);
        return scratch;
    }

    /// <summary>
    /// Σ w[i]·x[i] with one vector accumulator and a scalar tail — the reference
    /// order every float-path output is computed in.
    /// </summary>
    private static unsafe float Dot(float* w, float* x, int n)
    {
        int width = Vector<float>.Count, i = 0;
        Vector<float> acc = Vector<float>.Zero;
        for (; i + width <= n; i += width)
            acc = Vector.FusedMultiplyAdd(Vector.Load(x + i), Vector.Load(w + i), acc);
        float sum = Vector.Sum(acc);
        for (; i < n; i++) sum += x[i] * w[i];
        return sum;
    }

    /// <summary>
    /// Two rows against four consecutive tokens; <c>y[t·yStride + {0,1}]</c>. Each of
    /// the eight outputs follows exactly <see cref="Dot"/>'s sequence of operations.
    /// </summary>
    private static unsafe void Dot2x4(float* w0, float* w1, float* x, int n, float* y, int yStride)
    {
        int width = Vector<float>.Count, i = 0;
        float* x0 = x, x1 = x + n, x2 = x1 + n, x3 = x2 + n;
        Vector<float> a00 = Vector<float>.Zero, a01 = a00, a10 = a00, a11 = a00,
                      a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        for (; i + width <= n; i += width)
        {
            Vector<float> v0 = Vector.Load(w0 + i), v1 = Vector.Load(w1 + i);
            Vector<float> t = Vector.Load(x0 + i);
            a00 = Vector.FusedMultiplyAdd(t, v0, a00); a01 = Vector.FusedMultiplyAdd(t, v1, a01);
            t = Vector.Load(x1 + i);
            a10 = Vector.FusedMultiplyAdd(t, v0, a10); a11 = Vector.FusedMultiplyAdd(t, v1, a11);
            t = Vector.Load(x2 + i);
            a20 = Vector.FusedMultiplyAdd(t, v0, a20); a21 = Vector.FusedMultiplyAdd(t, v1, a21);
            t = Vector.Load(x3 + i);
            a30 = Vector.FusedMultiplyAdd(t, v0, a30); a31 = Vector.FusedMultiplyAdd(t, v1, a31);
        }

        float s00 = Vector.Sum(a00), s01 = Vector.Sum(a01), s10 = Vector.Sum(a10), s11 = Vector.Sum(a11),
              s20 = Vector.Sum(a20), s21 = Vector.Sum(a21), s30 = Vector.Sum(a30), s31 = Vector.Sum(a31);
        for (; i < n; i++)
        {
            float v0 = w0[i], v1 = w1[i];
            s00 += x0[i] * v0; s01 += x0[i] * v1;
            s10 += x1[i] * v0; s11 += x1[i] * v1;
            s20 += x2[i] * v0; s21 += x2[i] * v1;
            s30 += x3[i] * v0; s31 += x3[i] * v1;
        }

        y[0] = s00; y[1] = s01;
        y += yStride; y[0] = s10; y[1] = s11;
        y += yStride; y[0] = s20; y[1] = s21;
        y += yStride; y[0] = s30; y[1] = s31;
    }

    // ---------------------------------------------------------- integer GEMM

    /// <summary>
    /// The integer path where <see cref="IntegerGemm"/> has a kernel: quantize the
    /// activations once, then per group of eight rows unpack into the interleaved
    /// layout and sweep every token past it four at a time.
    /// </summary>
    private static async ValueTask ForwardGemmAsync(
        WeightMatrix          w,
        ReadOnlyMemory<int>   rowIds,
        bool                  mapped,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
    {
        int cols = w.Cols, rows = mapped ? rowIds.Length : w.Rows;
        int blocks = cols / IntegerGemm.BlockSize;

        sbyte[] values = ArrayPool<sbyte>.Shared.Rent(tokens * cols);
        float[] scales = ArrayPool<float>.Shared.Rent(tokens * blocks);
        int[] sums = ArrayPool<int>.Shared.Rent(tokens * blocks);
        try
        {
            QuantizeForGemm(x.Span, values, scales, sums, tokens, cols, blocks);

            const int groupsPerChunk = RowChunk / IntegerGemm.GroupRows;
            int groups = (rows + IntegerGemm.GroupRows - 1) / IntegerGemm.GroupRows;
            int chunks = (groups + groupsPerChunk - 1) / groupsPerChunk;

            await Parallel.ForAsync(0, chunks, options, (chunk, _) =>
            {
                int g0 = chunk * groupsPerChunk;
                int g1 = Math.Min(g0 + groupsPerChunk, groups);
                RunGroups(w, rowIds.Span, mapped, values, scales, sums, destination.Span, tokens, g0, g1, rows, blocks);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(values);
            ArrayPool<float>.Shared.Return(scales);
            ArrayPool<int>.Shared.Return(sums);
        }
    }

    private static unsafe void QuantizeForGemm(
        ReadOnlySpan<float> x, sbyte[] values, float[] scales, int[] sums, int tokens, int cols, int blocks)
    {
        fixed (float* xp = x)
        fixed (sbyte* v = values)
        fixed (float* s = scales)
        fixed (int* q = sums)
        {
            for (int t = 0; t < tokens; t++)
                IntegerGemm.QuantizeActivations(xp + (long)t * cols, cols,
                    v + (long)t * cols, s + (long)t * blocks, q + (long)t * blocks);
        }
    }

    private static unsafe void RunGroups(
        WeightMatrix w, ReadOnlySpan<int> rowIds, bool mapped,
        sbyte[] values, float[] activationScales, int[] activationSums, Span<float> y,
        int tokens, int groupStart, int groupEnd, int rows, int blocks)
    {
        const int group = IntegerGemm.GroupRows, tileTokens = IntegerGemm.TokenTile;
        int cols = w.Cols;
        bool hasOffset = IntegerGemm.HasOffset(w.Type);
        int zeroPoint = IntegerGemm.ZeroPoint(w.Type);

        byte[] packedBuffer = ArrayPool<byte>.Shared.Rent(blocks * IntegerGemm.PackedBlockBytes);
        float[] scaleBuffer = ArrayPool<float>.Shared.Rent(blocks * group);
        float[]? offsetBuffer = hasOffset ? ArrayPool<float>.Shared.Rent(blocks * group) : null;
        try
        {
            fixed (byte* packed = packedBuffer)
            fixed (float* scales = scaleBuffer)
            fixed (float* offsetStorage = offsetBuffer)
            fixed (sbyte* a = values)
            fixed (float* ad = activationScales)
            fixed (int* asum = activationSums)
            fixed (float* yBase = y)
            {
                float* offsets = hasOffset ? offsetStorage : null;
                byte** rowPointers = stackalloc byte*[group];
                // A group that runs off the end of the matrix still writes eight
                // lanes; they land here and only the real ones are copied out.
                float* spill = stackalloc float[tileTokens * group];

                for (int g = groupStart; g < groupEnd; g++)
                {
                    int r0 = g * group;
                    int count = Math.Min(group, rows - r0);
                    for (int i = 0; i < count; i++)
                        rowPointers[i] = w.Row(SourceRow(rowIds, mapped, r0 + i));

                    IntegerGemm.PackGroup(w.Type, rowPointers, count, blocks, packed, scales, offsets);

                    int t = 0;
                    for (; t + tileTokens <= tokens; t += tileTokens)
                    {
                        bool full = count == group;
                        IntegerGemm.Multiply4(packed, scales, offsets, blocks, zeroPoint,
                            a + (long)t * cols, ad + (long)t * blocks, asum + (long)t * blocks, cols,
                            full ? yBase + (long)t * rows + r0 : spill, full ? rows : group);
                        if (!full)
                            for (int j = 0; j < tileTokens; j++)
                                new ReadOnlySpan<float>(spill + j * group, count)
                                    .CopyTo(new Span<float>(yBase + (long)(t + j) * rows + r0, count));
                    }
                    for (; t < tokens; t++)
                    {
                        IntegerGemm.Multiply1(packed, scales, offsets, blocks, zeroPoint,
                            a + (long)t * cols, ad + (long)t * blocks, asum + (long)t * blocks, spill);
                        new ReadOnlySpan<float>(spill, count).CopyTo(new Span<float>(yBase + (long)t * rows + r0, count));
                    }
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(packedBuffer);
            ArrayPool<float>.Shared.Return(scaleBuffer);
            if (offsetBuffer is not null) ArrayPool<float>.Shared.Return(offsetBuffer);
        }
    }

    // -------------------------------------------------------- integer (dots)

    /// <summary>
    /// Integer path for hardware or types <see cref="IntegerGemm"/> does not cover:
    /// quantize the activations to Q8_0 once, then dot in 8-bit, one output at a time.
    ///
    /// The activations are quantized for the whole call rather than per tile —
    /// they are two orders of magnitude smaller than the weights, so there is
    /// nothing to gain from tiling them here, and one pass keeps the code honest
    /// about where the cost actually is.
    /// </summary>
    private static async ValueTask ForwardIntegerAsync(
        WeightMatrix          w,
        ReadOnlyMemory<int>   rowIds,
        bool                  mapped,
        ReadOnlyMemory<float> x,
        int                   tokens,
        Memory<float>         destination,
        ParallelOptions       options)
    {
        int cols = w.Cols, rows = mapped ? rowIds.Length : w.Rows;
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
                RunRowsInteger(w, rowIds.Span, mapped, quantized, activationSums, destination.Span,
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
        WeightMatrix w, ReadOnlySpan<int> rowIds, bool mapped,
        byte[] quantized, float[] activationSums, Span<float> y,
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
                    byte* row = w.Row(SourceRow(rowIds, mapped, r));
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
