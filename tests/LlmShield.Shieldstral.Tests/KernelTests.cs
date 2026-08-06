using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Numerics;
using LlmShield.Shieldstral.Quantization;
using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The vectorised primitives against straightforward scalar definitions. These
/// kernels are where a width-dependent bug hides: a loop that handles
/// Vector&lt;float&gt;.Count elements correctly and drops a ragged tail produces
/// activations that are almost right, which is the worst kind of wrong.
/// </summary>
[Collection(MatMulStrategyCollection.Name)]
public class KernelTests
{
    private static float[] Random(int n, int seed = 7)
    {
        var rng = new Random(seed);
        var values = new float[n];
        for (int i = 0; i < n; i++) values[i] = (float)(rng.NextDouble() * 4 - 2);
        return values;
    }

    // Deliberately includes lengths that are not multiples of any vector width.
    public static TheoryData<int> Lengths() => [1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 127, 3072, 9216];

    [Theory]
    [MemberData(nameof(Lengths))]
    public void RmsNormMatchesTheDefinition(int n)
    {
        float[] x = Random(n), w = Random(n, 11);
        const float eps = 1e-5f;

        double mean = 0;
        for (int i = 0; i < n; i++) mean += (double)x[i] * x[i];
        mean /= n;
        float scale = 1f / MathF.Sqrt((float)mean + eps);

        var expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = x[i] * scale * w[i];

        var actual = new float[n];
        Kernels.RmsNorm(x, w, eps, actual);
        Numeric.Close(expected, actual, 1e-5, $"rmsnorm({n})");
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SoftmaxSumsToOneAndIsShiftInvariant(int n)
    {
        float[] x = Random(n);
        var a = (float[])x.Clone();
        Kernels.Softmax(a);

        Numeric.Close(1.0, a.Sum(), 1e-5, "softmax total");
        Assert.All(a, v => Assert.InRange(v, 0f, 1f));

        // Adding a constant to every logit must not change the distribution.
        var shifted = x.Select(v => v + 13.5f).ToArray();
        Kernels.Softmax(shifted);
        Numeric.Close(a, shifted, 1e-5, "softmax shift invariance");
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SwiGluMatchesTheDefinition(int n)
    {
        float[] gate = Random(n), up = Random(n, 23);
        var expected = new float[n];
        for (int i = 0; i < n; i++)
            expected[i] = gate[i] / (1f + MathF.Exp(-gate[i])) * up[i];

        var destination = new float[n];
        Kernels.SwiGlu(gate, (float[])up.Clone(), destination);
        Numeric.Close(expected, destination, 1e-5, $"swiglu({n})");
    }

    [Fact]
    public void SwiGluToleratesDestinationAliasingGate()
    {
        // The FFN calls it this way to avoid a third 9216-wide buffer per token.
        float[] gate = Random(129), up = Random(129, 5);
        var expected = new float[129];
        Kernels.SwiGlu(gate, (float[])up.Clone(), expected);

        var inPlace = (float[])gate.Clone();
        Kernels.SwiGlu(inPlace, (float[])up.Clone(), inPlace);
        Numeric.Close(expected, inPlace, 1e-6, "swiglu aliased");
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void AddScaledMatchesTheDefinition(int n)
    {
        float[] dst = Random(n), src = Random(n, 3);
        const float scale = 0.375f;
        var expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = dst[i] + src[i] * scale;

        var actual = (float[])dst.Clone();
        Kernels.AddScaled(actual, src, scale);
        Numeric.Close(expected, actual, 1e-6, $"addScaled({n})");
    }

    /// <summary>
    /// Attention scores reach the 90s in this model's deepest layers, where
    /// <c>exp</c> overflows float32. A softmax that does not subtract the maximum
    /// first returns NaN there — and only there, so it survives short prompts and
    /// shallow layers and then poisons the residual stream on a real one.
    /// </summary>
    [Theory]
    [InlineData(50f)]
    [InlineData(95f)]
    [InlineData(400f)]
    public void SoftmaxSurvivesLargeLogits(float magnitude)
    {
        var x = new float[64];
        for (int i = 0; i < x.Length; i++) x[i] = magnitude * (i / (float)(x.Length - 1));

        var actual = (float[])x.Clone();
        Kernels.Softmax(actual);

        Assert.All(actual, v => Assert.True(float.IsFinite(v), $"softmax produced {v}"));
        Numeric.Close(1.0, actual.Sum(), 1e-5, "softmax total");

        // The largest logit must still dominate, and the shape must be the same as
        // for the numerically-safe reference.
        Assert.Equal(x.Length - 1, Array.IndexOf(actual, actual.Max()));

        var expected = new double[x.Length];
        double sum = 0;
        for (int i = 0; i < x.Length; i++) { expected[i] = Math.Exp(x[i] - magnitude); sum += expected[i]; }
        for (int i = 0; i < x.Length; i++) Numeric.Close(expected[i] / sum, actual[i], 1e-5, $"p[{i}]");
    }

    [Fact]
    public void SoftmaxOfAllMaskedPositionsStaysFinite()
    {
        // A fully masked row cannot happen under a causal mask, but a NaN here
        // would propagate silently through every later layer, so guard it.
        var x = new float[8];
        Array.Fill(x, float.NegativeInfinity);
        Kernels.Softmax(x);
        Assert.All(x, v => Assert.False(float.IsNaN(v), "softmax produced NaN"));
    }

    /// <summary>
    /// The GEMM against a plain triple loop, including a token count that spans
    /// more than one internal tile.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(400)]
    public unsafe void QuantMatMulMatchesNaiveProduct(int tokens)
    {
        const int rows = 37, cols = 64;
        float[] weights = Random(rows * cols, 31);
        float[] x = Random(tokens * cols, 41);

        fixed (float* w = weights)
        {
            var matrix = new WeightMatrix(GgmlType.F32, (byte*)w, rows, cols);
            var actual = new float[tokens * rows];
            QuantMatMul.Forward(matrix, x, tokens, actual);

            for (int t = 0; t < tokens; t++)
                for (int r = 0; r < rows; r++)
                {
                    double expected = 0;
                    for (int k = 0; k < cols; k++) expected += (double)x[t * cols + k] * weights[r * cols + k];
                    Numeric.Close(expected, actual[t * rows + r], 1e-5, $"C[{t},{r}]");
                }
        }
    }

    /// <summary>
    /// A quantized weight must give the same answer as its dequantized twin —
    /// that is what lets the matmul decode rows lazily instead of materialising
    /// the whole matrix.
    /// </summary>
    /// <remarks>
    /// Pinned to the float strategy: this is about lazy versus eager decoding, and
    /// the integer path deliberately introduces activation quantization error.
    /// <c>IntegerDotTests</c> covers that comparison separately.
    /// </remarks>
    [Fact]
    public unsafe void QuantMatMulAgreesWithDequantizedWeights()
    {
        QuantMatMul.Strategy = MatMulStrategy.Float;
        try
        {
            AssertQuantizedMatchesDequantized();
        }
        finally
        {
            QuantMatMul.Strategy = MatMulStrategy.Auto;
        }
    }

    private static unsafe void AssertQuantizedMatchesDequantized()
    {
        const int rows = 8, cols = 256;
        var rng = new Random(99);
        var blocks = new byte[rows * GgmlTypeInfo.RowBytes(GgmlType.Q8_0, cols)];
        rng.NextBytes(blocks);
        // Keep the per-block scales in a sane range so the comparison is meaningful.
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < cols / 32; b++)
            {
                int offset = (int)(r * GgmlTypeInfo.RowBytes(GgmlType.Q8_0, cols)) + b * 34;
                BitConverter.GetBytes((Half)0.01f).CopyTo(blocks, offset);
            }

        float[] x = Random(cols, 12);
        var dequantized = new float[rows * cols];
        fixed (byte* p = blocks)
        {
            var quantized = new WeightMatrix(GgmlType.Q8_0, p, rows, cols);
            for (int r = 0; r < rows; r++) quantized.DequantizeRow(r, dequantized.AsSpan(r * cols, cols));

            var fromQuantized = new float[rows];
            QuantMatMul.Forward(quantized, x, 1, fromQuantized);

            fixed (float* d = dequantized)
            {
                var plain = new WeightMatrix(GgmlType.F32, (byte*)d, rows, cols);
                var fromFloat = new float[rows];
                QuantMatMul.Forward(plain, x, 1, fromFloat);
                Numeric.Close(fromFloat, fromQuantized, 1e-6, "quantized vs dequantized");
            }
        }
    }
}
