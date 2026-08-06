using System.Text.Json;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Quantization;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Checks every GGML type the runtime claims to read against the reference `gguf`
/// Python package, on both realistic quantizer output and random bytes.
///
/// The random-bytes case is the one that finds real bugs: a k-quant's packed
/// 6-bit scales or an i-quant's sign mask are almost always benign on realistic
/// weights and catastrophic on the corners, so exercising the full encoding space
/// is what makes "we support all quantization types" a claim rather than a hope.
/// </summary>
public class DequantizerParityTests
{
    private readonly ITestOutputHelper _output;
    public DequantizerParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Tolerance is on the reconstructed value. Both sides do the same arithmetic
    /// but in a different order and, in Python, at float64 in places — so exact
    /// equality would be testing numpy's evaluation order, not the unpacking.
    /// </summary>
    private const float Tolerance = 1e-4f;

    public static TheoryData<string> AllTypes()
    {
        var data = new TheoryData<string>();
        using JsonDocument doc = Fixtures.Load("quantization.json");
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
            data.Add(c.GetProperty("type").GetString()!);
        return data;
    }

    [Theory]
    [MemberData(nameof(AllTypes))]
    public void Dequantize_MatchesReference(string typeName)
    {
        using JsonDocument doc = Fixtures.Load("quantization.json");
        JsonElement entry = doc.RootElement.GetProperty("cases").EnumerateArray()
            .First(c => c.GetProperty("type").GetString() == typeName);

        Assert.True(GgmlTypeInfo.TryParse(typeName, out GgmlType type), $"unknown GGML type '{typeName}'");
        Assert.True(Dequantizer.Supports(type), $"{type} is in the fixture but Supports() says no");

        // Block geometry has to agree before the contents can mean anything.
        Assert.Equal(entry.GetProperty("block_size").GetInt32(), GgmlTypeInfo.BlockSize(type));
        Assert.Equal(entry.GetProperty("type_size").GetInt32(), GgmlTypeInfo.TypeSize(type));

        int compared = 0;
        foreach (string kind in (string[])["quantized", "synthetic"])
        {
            if (!entry.TryGetProperty(kind, out JsonElement blob)) continue;

            byte[] bytes = Convert.FromBase64String(blob.GetProperty("bytes").GetString()!);
            float[] expected = [.. blob.GetProperty("expected").EnumerateArray().Select(v => v.GetSingle())];

            var actual = new float[expected.Length];
            Dequantizer.Dequantize(type, bytes, actual);

            int worstIndex = -1;
            float worst = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                float delta = MathF.Abs(actual[i] - expected[i]);
                float scale = MathF.Max(1f, MathF.Abs(expected[i]));
                if (delta / scale > worst) { worst = delta / scale; worstIndex = i; }
            }

            Assert.True(worst <= Tolerance,
                $"{type} ({kind}): element {worstIndex} is {actual[Math.Max(worstIndex, 0)]} " +
                $"but the reference says {expected[Math.Max(worstIndex, 0)]} " +
                $"(relative error {worst:E3}, tolerance {Tolerance:E0})");

            _output.WriteLine($"{type,-9} {kind,-9} {expected.Length,5} values, worst relative error {worst:E2}");
            compared++;
        }

        Assert.True(compared > 0, $"{type} had no fixture data to compare against");
    }

    [Fact]
    public void EveryTypeTheReaderAcceptsHasAFixture()
    {
        using JsonDocument doc = Fixtures.Load("quantization.json");
        var covered = doc.RootElement.GetProperty("cases").EnumerateArray()
            .Select(c => c.GetProperty("type").GetString()!)
            .ToHashSet(StringComparer.Ordinal);

        // Integer element types exist in the format but never appear in a weight
        // tensor, so gguf has no dequantizer to compare against; they are covered
        // by DequantizeIntegerTypes below instead.
        string[] integerOnly = ["I8", "I16", "I32", "I64", "F64", "Q8_1", "Q8_K"];

        var missing = Enum.GetValues<GgmlType>()
            .Where(Dequantizer.Supports)
            .Select(t => t.ToString())
            .Where(name => !covered.Contains(name) && !integerOnly.Contains(name))
            .ToArray();

        Assert.True(missing.Length == 0,
            $"these supported types have no parity fixture: {string.Join(", ", missing)}");
    }

    /// <summary>
    /// The types gguf has no dequantizer for. Their decoding is trivial enough to
    /// state directly, which is the point — they still must not silently return
    /// zeros or throw.
    /// </summary>
    [Fact]
    public void DequantizeIntegerTypes()
    {
        AssertDecodes(GgmlType.I8, [0x01, 0xFF, 0x7F, 0x80], [1f, -1f, 127f, -128f]);
        AssertDecodes(GgmlType.I16, [0x01, 0x00, 0xFF, 0xFF], [1f, -1f]);
        AssertDecodes(GgmlType.I32, [0x02, 0x00, 0x00, 0x00], [2f]);
        AssertDecodes(GgmlType.F64, BitConverter.GetBytes(-3.5), [-3.5f]);

        static void AssertDecodes(GgmlType type, byte[] bytes, float[] expected)
        {
            var actual = new float[expected.Length];
            Dequantizer.Dequantize(type, bytes, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Q8_1_DecodesValuesIgnoringTheBlockSum()
    {
        // block_q8_1 is {half d, half s, int8 qs[32]}: s is a precomputed sum the
        // dot kernels use and must not leak into the reconstructed values.
        var block = new byte[GgmlTypeInfo.TypeSize(GgmlType.Q8_1)];
        BitConverter.GetBytes((Half)0.5f).CopyTo(block, 0);
        BitConverter.GetBytes((Half)999f).CopyTo(block, 2);   // the sum, deliberately absurd
        for (int i = 0; i < 32; i++) block[4 + i] = unchecked((byte)(sbyte)(i - 16));

        var values = new float[32];
        Dequantizer.Dequantize(GgmlType.Q8_1, block, values);
        for (int i = 0; i < 32; i++)
            Assert.Equal(0.5f * (i - 16), values[i], 5);
    }

    [Fact]
    public void RejectsPartialBlocks()
    {
        var destination = new float[17];
        var bytes = new byte[1024];
        Assert.Throws<ArgumentException>(() => Dequantizer.Dequantize(GgmlType.Q4_0, bytes, destination));
    }

    [Fact]
    public void RowBytesRejectsUnalignedRows()
    {
        Assert.Equal(18, GgmlTypeInfo.RowBytes(GgmlType.Q4_0, 32));
        Assert.Equal(144, GgmlTypeInfo.RowBytes(GgmlType.Q4_K, 256));
        Assert.Throws<ArgumentException>(() => GgmlTypeInfo.RowBytes(GgmlType.Q4_K, 200));
    }
}
