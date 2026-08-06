// Weight-stationary GEMM against GGUF-quantized weights.
using System.Buffers;
using System.Numerics.Tensors;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Quantization;

namespace LlmShield.Shieldstral.Numerics;

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
/// </summary>
public static unsafe class QuantMatMul
{
    /// <summary>
    /// Bytes of activations to keep resident per tile. Every extra tile means
    /// another full decode pass over the weights, so this is sized generously (a
    /// typical L3 is 8-32 MiB) — at 4 MiB a 340-token prompt at hidden size 3072
    /// still fits in one tile, and no realistic Shieldstral prompt pays twice.
    /// </summary>
    private const int TokenTileBytes = 4 * 1024 * 1024;

    /// <summary>Rows handed to one worker at a time; large enough to amortise the scratch rent.</summary>
    private const int RowChunk = 64;

    /// <summary>
    /// Computes <paramref name="destination"/>[t, r] = dot(x[t, :], W[r, :]).
    /// <paramref name="x"/> is <paramref name="tokens"/> × <c>W.Cols</c> row-major;
    /// <paramref name="destination"/> is <paramref name="tokens"/> × <c>W.Rows</c>.
    /// </summary>
    public static void Forward(in WeightMatrix w, ReadOnlySpan<float> x, int tokens, Span<float> destination)
    {
        int cols = w.Cols, rows = w.Rows;
        if (x.Length < (long)tokens * cols)
            throw new ArgumentException($"Expected {(long)tokens * cols} activations, got {x.Length}.", nameof(x));
        if (destination.Length < (long)tokens * rows)
            throw new ArgumentException($"Expected room for {(long)tokens * rows} outputs, got {destination.Length}.", nameof(destination));

        int tile = Math.Clamp(TokenTileBytes / (cols * sizeof(float)), 1, tokens);

        fixed (float* xp = x)
        fixed (float* yp = destination)
        {
            float* xBase = xp, yBase = yp;
            WeightMatrix weight = w;

            for (int t0 = 0; t0 < tokens; t0 += tile)
            {
                int tileLen = Math.Min(tile, tokens - t0);
                int chunks = (rows + RowChunk - 1) / RowChunk;

                // One token is a pure memory walk over the weights; splitting it
                // across cores still helps, so parallelise on rows either way.
                Parallel.For(0, chunks, chunk =>
                {
                    int r0 = chunk * RowChunk;
                    int r1 = Math.Min(r0 + RowChunk, rows);
                    RunRows(weight, xBase, yBase, t0, tileLen, r0, r1, cols, rows);
                });
            }
        }
    }

    private static void RunRows(
        WeightMatrix w, float* x, float* y,
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
                        var xSpan = new ReadOnlySpan<float>(x + (long)(tokenStart + t) * cols, cols);
                        y[(long)(tokenStart + t) * rows + r] = TensorPrimitives.Dot(xSpan, wSpan);
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
    /// Single-token convenience overload — the decode path, where the whole
    /// forward pass is one row of activations.
    /// </summary>
    public static void Forward(in WeightMatrix w, ReadOnlySpan<float> x, Span<float> destination)
        => Forward(w, x, tokens: 1, destination);

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
