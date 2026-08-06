using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Numerics;
using LlmShield.Shieldstral.Quantization;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The integer matmul path against the float one it is meant to replace.
///
/// It is not expected to be bit-identical: quantizing the activations to Q8_0
/// loses about 0.4% per element. What matters is that the *dot* stays accurate —
/// a 3072-length sum averages that error down by roughly √n — and that the
/// unpacking of each weight family is right, which a systematic error would show
/// as a bias rather than noise.
/// </summary>
[Collection(MatMulStrategyCollection.Name)]
public class IntegerDotTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    public IntegerDotTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => QuantMatMul.Strategy = MatMulStrategy.Auto;

    public static TheoryData<GgmlType> IntegerTypes() =>
        [GgmlType.Q8_0, GgmlType.Q4_0, GgmlType.Q4_1, GgmlType.Q5_0, GgmlType.Q5_1];

    /// <summary>
    /// Builds weight bytes of the given type by quantizing real values, so the
    /// scales and packed fields are internally consistent rather than random.
    /// </summary>
    private static unsafe byte[] QuantizeRows(GgmlType type, int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        int rowBytes = checked((int)GgmlTypeInfo.RowBytes(type, cols));
        var bytes = new byte[(long)rows * rowBytes];

        // Round-trip through the dequantizer's inverse is not available for every
        // type, so synthesise the block fields directly: a fixed sensible scale,
        // random payload nibbles. That exercises the full payload range while
        // keeping the reconstructed magnitudes realistic.
        rng.NextBytes(bytes);
        int typeSize = GgmlTypeInfo.TypeSize(type);
        bool hasMin = type is GgmlType.Q4_1 or GgmlType.Q5_1;
        for (long offset = 0; offset + typeSize <= bytes.Length; offset += typeSize)
        {
            BitConverter.GetBytes((Half)(0.005f + 0.02f * (float)rng.NextDouble())).CopyTo(bytes, offset);
            if (hasMin) BitConverter.GetBytes((Half)(-0.03f * (float)rng.NextDouble())).CopyTo(bytes, offset + 2);
        }
        return bytes;
    }

    [Theory]
    [MemberData(nameof(IntegerTypes))]
    public async Task IntegerPathAgreesWithFloatPath(GgmlType type)
    {
        const int rows = 96, cols = 3072, tokens = 3;
        byte[] weights = QuantizeRows(type, rows, cols, seed: 21);

        var x = new float[tokens * cols];
        var rng = new Random(5);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        var viaFloat = new float[tokens * rows];
        var viaInteger = new float[tokens * rows];

        QuantMatMul.Strategy = MatMulStrategy.Float;
        await PinnedMatMul.ForwardAsync(type, weights, rows, cols, x, tokens, viaFloat);

        QuantMatMul.Strategy = MatMulStrategy.Integer;
        await PinnedMatMul.ForwardAsync(type, weights, rows, cols, x, tokens, viaInteger);

        double norm = 0, difference = 0, bias = 0;
        for (int i = 0; i < viaFloat.Length; i++)
        {
            norm += (double)viaFloat[i] * viaFloat[i];
            double d = viaInteger[i] - (double)viaFloat[i];
            difference += d * d;
            bias += d;
        }
        double relative = Math.Sqrt(difference / norm);
        double meanBias = bias / viaFloat.Length / Math.Sqrt(norm / viaFloat.Length);

        _output.WriteLine($"{type}: relative L2 {relative:E3}, normalised bias {meanBias:E3}");

        // Activation quantization is ~0.4% per element; over a 3072-term dot the
        // errors are independent enough to average down to well under 1%.
        Assert.True(relative < 0.01,
            $"{type}: integer path differs from float by {relative:E3} in relative L2");

        // A wrong unpacking (an inverted nibble order, a missing offset) shows up as
        // a systematic shift rather than noise, so bound the bias much more tightly.
        Assert.True(Math.Abs(meanBias) < 2e-3,
            $"{type}: integer path is systematically offset by {meanBias:E3} — this looks like " +
            "an unpacking error rather than quantization noise");
    }

    [Fact]
    public void AutoStrategyUsesIntegerOnlyWhereThereIsAKernel()
    {
        foreach (GgmlType type in Enum.GetValues<GgmlType>().Where(Dequantizer.Supports))
        {
            bool supported = IntegerDot.Supports(type);
            // Every supported type must have a single per-block scale; the k-quants
            // and i-quants carry per-sub-block scales and must not be claimed.
            if (supported)
                Assert.Equal(32, GgmlTypeInfo.BlockSize(type));
        }
        Assert.False(IntegerDot.Supports(GgmlType.Q4_K));
        Assert.False(IntegerDot.Supports(GgmlType.IQ4_XS));
        Assert.False(IntegerDot.Supports(GgmlType.F32));
    }

    [Fact]
    public void ActivationQuantizationRoundTripsWithinOnePercent()
    {
        const int n = 1024;
        var values = new float[n];
        var rng = new Random(11);
        for (int i = 0; i < n; i++) values[i] = (float)(rng.NextDouble() * 8 - 4);

        var quantized = new byte[n / IntegerDot.BlockSize * IntegerDot.ActivationBlockBytes];
        IntegerDot.QuantizeActivations(values, quantized);

        var restored = new float[n];
        Dequantizer.Dequantize(GgmlType.Q8_0, quantized, restored);

        // Q8_0's error is absolute, not relative: it is at most half a step of a
        // 254-level scale spanning the block's own dynamic range. Measuring against
        // the value itself would report unbounded error for anything near zero, so
        // normalise by the block's magnitude instead.
        double worst = 0;
        for (int b = 0; b < n / IntegerDot.BlockSize; b++)
        {
            float blockMax = 0;
            for (int i = b * IntegerDot.BlockSize; i < (b + 1) * IntegerDot.BlockSize; i++)
                blockMax = Math.Max(blockMax, Math.Abs(values[i]));
            for (int i = b * IntegerDot.BlockSize; i < (b + 1) * IntegerDot.BlockSize; i++)
                worst = Math.Max(worst, Math.Abs(restored[i] - values[i]) / blockMax);
        }
        _output.WriteLine($"worst round-trip error, relative to block scale: {worst:E3}");

        Assert.True(worst < 1.0 / 254 * 1.05,
            $"activation round-trip lost {worst:E3} of the block scale, more than half a quantizer step");
    }

    [Fact]
    public void ActivationQuantizationHandlesAnAllZeroBlock()
    {
        // An all-zero block gives a zero scale; the reciprocal must not become
        // infinity and poison the whole row.
        var values = new float[32];
        var quantized = new byte[IntegerDot.ActivationBlockBytes];
        IntegerDot.QuantizeActivations(values, quantized);

        var restored = new float[32];
        Dequantizer.Dequantize(GgmlType.Q8_0, quantized, restored);
        Assert.All(restored, v => Assert.Equal(0f, v));
    }
}
