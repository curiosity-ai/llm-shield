using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Model;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The GGUF reader against both a hand-built file (so the header parsing is
/// exercised without a 3.5 GB download) and the real checkpoint.
/// </summary>
public class GgufReaderTests
{
    private readonly ITestOutputHelper _output;
    public GgufReaderTests(ITestOutputHelper output) => _output = output;

    /// <summary>Block geometry, cross-checked against the constants in ggml.h.</summary>
    [Theory]
    [InlineData(GgmlType.F32, 1, 4)]
    [InlineData(GgmlType.F16, 1, 2)]
    [InlineData(GgmlType.BF16, 1, 2)]
    [InlineData(GgmlType.Q4_0, 32, 18)]
    [InlineData(GgmlType.Q4_1, 32, 20)]
    [InlineData(GgmlType.Q5_0, 32, 22)]
    [InlineData(GgmlType.Q5_1, 32, 24)]
    [InlineData(GgmlType.Q8_0, 32, 34)]
    [InlineData(GgmlType.Q8_1, 32, 36)]
    [InlineData(GgmlType.Q2_K, 256, 84)]
    [InlineData(GgmlType.Q3_K, 256, 110)]
    [InlineData(GgmlType.Q4_K, 256, 144)]
    [InlineData(GgmlType.Q5_K, 256, 176)]
    [InlineData(GgmlType.Q6_K, 256, 210)]
    [InlineData(GgmlType.Q8_K, 256, 292)]
    [InlineData(GgmlType.IQ2_XXS, 256, 66)]
    [InlineData(GgmlType.IQ2_XS, 256, 74)]
    [InlineData(GgmlType.IQ2_S, 256, 82)]
    [InlineData(GgmlType.IQ3_XXS, 256, 98)]
    [InlineData(GgmlType.IQ3_S, 256, 110)]
    [InlineData(GgmlType.IQ1_S, 256, 50)]
    [InlineData(GgmlType.IQ1_M, 256, 56)]
    [InlineData(GgmlType.IQ4_NL, 32, 18)]
    [InlineData(GgmlType.IQ4_XS, 256, 136)]
    [InlineData(GgmlType.TQ1_0, 256, 54)]
    [InlineData(GgmlType.TQ2_0, 256, 66)]
    [InlineData(GgmlType.MXFP4, 32, 17)]
    public void BlockGeometryMatchesGgml(GgmlType type, int blockSize, int typeSize)
    {
        Assert.Equal(blockSize, GgmlTypeInfo.BlockSize(type));
        Assert.Equal(typeSize, GgmlTypeInfo.TypeSize(type));
        Assert.Equal(blockSize > 1, GgmlTypeInfo.IsQuantized(type));
    }

    [Fact]
    public void ReadsAHandBuiltFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"tiny-{Guid.NewGuid():N}.gguf");
        var weights = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        try
        {
            WriteMinimalGguf(path, weights);
            using var gguf = new GgufFile(path);

            Assert.Equal(3u, gguf.Version);
            Assert.Equal("test", gguf.GetString("general.architecture"));
            Assert.Equal(7u, gguf.GetUInt32("test.block_count"));
            Assert.Equal(2.5f, gguf.GetFloat32("test.scale"));
            Assert.True(gguf.GetBool("test.flag"));
            Assert.Equal(["a", "b", "c"], gguf.GetStringArray("test.list")!);
            Assert.Equal([10, 20], gguf.GetInt32Array("test.ints")!);

            GgufTensorInfo info = gguf.GetTensor("w");
            Assert.Equal([3, 2], info.Shape);        // GGUF order: ne0 first
            Assert.Equal(GgmlType.F32, info.Type);
            Assert.Equal(6, info.ElementCount);
            Assert.Equal(24, info.ByteCount);

            var read = new float[6];
            Quantization.Dequantizer.Dequantize(GgmlType.F32, gguf.GetTensorBytes(info), read);
            Assert.Equal(weights, read);

            Assert.Throws<KeyNotFoundException>(() => gguf.GetTensor("missing"));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void RejectsATruncatedFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"short-{Guid.NewGuid():N}.gguf");
        try
        {
            WriteMinimalGguf(path, [1f, 2f, 3f, 4f, 5f, 6f]);
            // Lose the last tensor bytes, exactly as an interrupted download would.
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
                fs.SetLength(fs.Length - 8);

            IOException error = Assert.Throws<IOException>(() => new GgufFile(path));
            Assert.Contains("incomplete", error.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void RejectsANonGgufFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"junk-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, [0, 1, 2, 3, 4, 5, 6, 7]);
            Assert.Throws<InvalidDataException>(() => new GgufFile(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ReadsTheRealCheckpoint()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using var gguf = new GgufFile(modelPath);
        ModelConfig config = ModelConfig.FromGguf(gguf);
        _output.WriteLine(config.ToString());

        // One tensor per layer of each kind, plus embeddings and the output norm.
        Assert.Equal(config.LayerCount * 9 + 2, gguf.Tensors.Count);

        // Every weight the runtime will touch must be a type it can decode.
        foreach (GgufTensorInfo info in gguf.Tensors.Values)
            Assert.True(Quantization.Dequantizer.Supports(info.Type),
                $"{info.Name} is {info.Type}, which the dequantizer does not handle");

        // Shieldstral ties its LM head to the embedding table.
        Assert.False(gguf.TryGetTensor("output.weight", out _));
        Assert.True(gguf.TryGetTensor("token_embd.weight", out GgufTensorInfo? embd));
        Assert.Equal([config.HiddenSize, config.VocabSize], embd!.Shape);
    }

    // ------------------------------------------------------------------ helper

    /// <summary>Writes a minimal but spec-correct GGUF v3 with one F32 tensor.</summary>
    private static void WriteMinimalGguf(string path, float[] weights)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(stream);

        w.Write(0x46554747u);            // "GGUF"
        w.Write(3u);                     // version
        w.Write(1ul);                    // tensor count
        w.Write(6ul);                    // kv count

        WriteString(w, "general.architecture"); w.Write((uint)GgufValueType.String); WriteString(w, "test");
        WriteString(w, "test.block_count"); w.Write((uint)GgufValueType.UInt32); w.Write(7u);
        WriteString(w, "test.scale"); w.Write((uint)GgufValueType.Float32); w.Write(2.5f);
        WriteString(w, "test.flag"); w.Write((uint)GgufValueType.Bool); w.Write((byte)1);
        WriteString(w, "test.list"); w.Write((uint)GgufValueType.Array);
        w.Write((uint)GgufValueType.String); w.Write(3ul);
        WriteString(w, "a"); WriteString(w, "b"); WriteString(w, "c");
        WriteString(w, "test.ints"); w.Write((uint)GgufValueType.Array);
        w.Write((uint)GgufValueType.Int32); w.Write(2ul); w.Write(10); w.Write(20);

        WriteString(w, "w");
        w.Write(2u);                     // dimensions
        w.Write(3ul); w.Write(2ul);      // ne0, ne1
        w.Write((uint)GgmlType.F32);
        w.Write(0ul);                    // offset within the data section

        // Pad to the default 32-byte alignment, then the tensor data.
        long pos = stream.Position;
        long padding = (32 - pos % 32) % 32;
        w.Write(new byte[padding]);
        foreach (float value in weights) w.Write(value);

        static void WriteString(BinaryWriter w, string s)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(s);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
        }
    }
}
