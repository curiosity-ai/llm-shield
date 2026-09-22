using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Numerics;
using LlmShield.Shieldstral.Quantization;
using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The matmul is register-blocked — eight rows by four tokens in the integer
/// GEMM, two by four in the float path — with narrower kernels for whatever is
/// left over. Two things above it depend on every output being computed by the
/// same arithmetic whichever of those kernels it landed in:
/// <list type="bullet">
/// <item>the prefix cache, whose cached and uncached runs put the same token at
/// different positions in the batch and must still agree bit for bit;</item>
/// <item>the verdict's LM head, which asks for two rows out of 131072 and must
/// get exactly what the full product would have given for them.</item>
/// </list>
/// These tests pin both, with shapes chosen to leave ragged edges everywhere:
/// 100 rows is not a multiple of 8 or of the 64-row chunk, and 13 tokens is three
/// full tiles and a tail.
/// </summary>
[Collection(MatMulStrategyCollection.Name)]
public class MatMulTilingTests : IDisposable
{
    private const int Rows = 100, Cols = 256;

    public void Dispose() => QuantMatMul.Strategy = MatMulStrategy.Auto;

    public static TheoryData<GgmlType, MatMulStrategy> Paths() => new()
    {
        { GgmlType.Q4_0, MatMulStrategy.Integer },
        { GgmlType.Q4_1, MatMulStrategy.Integer },
        { GgmlType.Q5_0, MatMulStrategy.Integer },
        { GgmlType.Q5_1, MatMulStrategy.Integer },
        { GgmlType.Q8_0, MatMulStrategy.Integer },
        { GgmlType.Q4_0, MatMulStrategy.Float },
        { GgmlType.Q4_K, MatMulStrategy.Float },
        { GgmlType.F32, MatMulStrategy.Float },
    };

    [Theory]
    [MemberData(nameof(Paths))]
    public async Task ATokensResultDoesNotDependOnTheBatchItIsIn(GgmlType type, MatMulStrategy strategy)
    {
        const int tokens = 13;
        byte[] weights = Weights(type, Rows, Cols, seed: 3);
        float[] x = Activations(tokens * Cols, seed: 4);
        QuantMatMul.Strategy = strategy;

        var together = new float[tokens * Rows];
        await PinnedMatMul.ForwardAsync(type, weights, Rows, Cols, x, tokens, together);

        // One token at a time, and split 5 + 8 so every token changes tile position.
        var alone = new float[Rows];
        for (int t = 0; t < tokens; t++)
        {
            await PinnedMatMul.ForwardAsync(type, weights, Rows, Cols, x.AsSpan(t * Cols, Cols).ToArray(), 1, alone);
            Assert.Equal(together.AsSpan(t * Rows, Rows).ToArray(), alone);
        }

        var tail = new float[8 * Rows];
        await PinnedMatMul.ForwardAsync(type, weights, Rows, Cols, x[(5 * Cols)..], 8, tail);
        Assert.Equal(together[(5 * Rows)..], tail);
    }

    [Theory]
    [MemberData(nameof(Paths))]
    public async Task ARowSubsetIsBitIdenticalToTheFullProduct(GgmlType type, MatMulStrategy strategy)
    {
        const int tokens = 5;
        // Out of order, repeated, and straddling group and chunk boundaries.
        int[] ids = [99, 0, 57, 57, 8, 7, 64, 63, 98];
        byte[] weights = Weights(type, Rows, Cols, seed: 5);
        float[] x = Activations(tokens * Cols, seed: 6);
        QuantMatMul.Strategy = strategy;

        var full = new float[tokens * Rows];
        await PinnedMatMul.ForwardAsync(type, weights, Rows, Cols, x, tokens, full);

        var subset = new float[tokens * ids.Length];
        await PinnedMatMul.ForwardRowsAsync(type, weights, Rows, Cols, ids, x, tokens, subset);

        for (int t = 0; t < tokens; t++)
            for (int i = 0; i < ids.Length; i++)
                Assert.Equal(full[t * Rows + ids[i]], subset[t * ids.Length + i]);
    }

    [Fact]
    public async Task ARowSubsetRejectsIdsOutsideTheMatrix()
    {
        byte[] weights = Weights(GgmlType.Q4_0, Rows, Cols, seed: 1);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => PinnedMatMul.ForwardRowsAsync(
            GgmlType.Q4_0, weights, Rows, Cols, [0, Rows], Activations(Cols, 2), 1, new float[2]));
    }

    /// <summary>
    /// Packing converts every block scale with a vectorised bit-trick rather than
    /// <see cref="Half"/>'s own conversion, so it has to agree with it everywhere —
    /// subnormals, infinities and NaN payloads included, not just the scales a
    /// sane checkpoint contains.
    /// </summary>
    [Fact]
    public void TheVectorisedHalfConversionIsExactForEveryBitPattern()
    {
        if (!IntegerGemm.IsHardwareAccelerated) return;

        for (int start = 0; start < 65536; start += 8)
        {
            var bits = System.Runtime.Intrinsics.Vector256.Create(
                start, start + 1, start + 2, start + 3, start + 4, start + 5, start + 6, start + 7);
            var converted = IntegerGemm.HalfToSingle(bits);
            for (int i = 0; i < 8; i++)
            {
                float expected = (float)BitConverter.UInt16BitsToHalf((ushort)(start + i));
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(converted[i]));
            }
        }
    }

    // ---------------------------------------------------------------- inputs

    private static float[] Activations(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    /// <summary>Random payloads with every f16 scale (and offset) forced to a sane value.</summary>
    private static byte[] Weights(GgmlType type, int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        if (type == GgmlType.F32)
        {
            var floats = new float[rows * cols];
            for (int i = 0; i < floats.Length; i++) floats[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            var raw = new byte[floats.Length * sizeof(float)];
            Buffer.BlockCopy(floats, 0, raw, 0, raw.Length);
            return raw;
        }

        var bytes = new byte[rows * GgmlTypeInfo.RowBytes(type, cols)];
        rng.NextBytes(bytes);
        int typeSize = GgmlTypeInfo.TypeSize(type);
        bool hasSecondHalf = type is GgmlType.Q4_1 or GgmlType.Q5_1 or GgmlType.Q4_K;
        for (int offset = 0; offset + typeSize <= bytes.Length; offset += typeSize)
        {
            BitConverter.GetBytes((Half)(0.005f + 0.02f * (float)rng.NextDouble())).CopyTo(bytes, offset);
            if (hasSecondHalf) BitConverter.GetBytes((Half)(-0.03f * (float)rng.NextDouble())).CopyTo(bytes, offset + 2);
        }
        return bytes;
    }
}
