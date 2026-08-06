using System.Reflection;
using System.Text.Json;
using LlmShield.Shieldstral.Gguf;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Loads the JSON oracles produced by <c>tools/*.py</c> and locates the model for
/// the tests that need real weights.
/// </summary>
internal static class Fixtures
{
    /// <summary>Directory (or GGUF path) holding a converted Shieldstral checkpoint.</summary>
    public const string ModelEnvironmentVariable = "SHIELDSTRAL_MODEL";

    private static readonly string Root = Path.Combine(
        Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, "fixtures");

    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
    };

    public static bool Exists(string name) => File.Exists(Path.Combine(Root, name));

    public static JsonDocument Load(string name)
    {
        string path = Path.Combine(Root, name);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Fixture '{name}' is missing. Regenerate it — see CLAUDE.md, 'Regenerating fixtures'.", path);
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    public static T Load<T>(string name)
    {
        using JsonDocument doc = Load(name);
        return doc.Deserialize<T>(Json)!;
    }

    /// <summary>
    /// Path to the Shieldstral GGUF, or null when <c>SHIELDSTRAL_MODEL</c> is unset
    /// or points nowhere. Model-backed tests skip rather than fail in that case, so
    /// a checkout without a 3.5 GB checkpoint still runs the rest of the suite.
    /// </summary>
    public static string? ModelPath
    {
        get
        {
            string? configured = Environment.GetEnvironmentVariable(ModelEnvironmentVariable);
            if (string.IsNullOrWhiteSpace(configured)) return null;

            if (File.Exists(configured)) return configured;
            if (!Directory.Exists(configured)) return null;

            return Directory.EnumerateFiles(configured, "*.gguf", SearchOption.AllDirectories)
                .Where(p => !Path.GetFileName(p).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                .OrderBy(p => p, StringComparer.Ordinal)
                .FirstOrDefault();
        }
    }

    /// <summary>Path to the Pixtral mmproj sitting next to the model, if there is one.</summary>
    public static string? VisionModelPath
    {
        get
        {
            string? model = ModelPath;
            if (model is null) return null;
            return Directory.EnumerateFiles(Path.GetDirectoryName(model)!, "*.gguf")
                .FirstOrDefault(p => Path.GetFileName(p).Contains("mmproj", StringComparison.OrdinalIgnoreCase));
        }
    }

    /// <summary>
    /// The coarsest quantization any layer weight in the checkpoint is stored in, read from the
    /// GGUF rather than guessed from the file name — a converted or renamed model would
    /// otherwise be measured against the wrong bound, which is the failure this exists to stop.
    /// <para>
    /// The coarsest rather than a representative tensor, because llama.cpp's <c>_S</c>/<c>_M</c>/
    /// <c>_L</c> recipes are per-tensor *mixtures*: a "Q5_K_M" file is mostly Q5_K but keeps
    /// <c>attn_v</c> and the output head at Q6_K. Whatever the file is called, the accuracy the
    /// activations actually see is set by the worst tensor in the mix.
    /// </para>
    /// </summary>
    public static GgmlType WeightType(string modelPath)
    {
        using var gguf = new GgufFile(modelPath);

        GgmlType coarsest = GgmlType.F32;
        double worstBits = double.MaxValue;

        foreach ((string name, GgufTensorInfo tensor) in gguf.Tensors)
        {
            if (!name.StartsWith("blk.", StringComparison.Ordinal)) continue;
            if (!GgmlTypeInfo.IsQuantized(tensor.Type)) continue;

            double bits = GgmlTypeInfo.TypeSize(tensor.Type) * 8.0 / GgmlTypeInfo.BlockSize(tensor.Type);

            if (bits < worstBits)
            {
                worstBits = bits;
                coarsest = tensor.Type;
            }
        }

        return coarsest;
    }

    /// <summary>
    /// How far a tensor may drift from the bf16 NumPy reference before it counts as a wiring
    /// error rather than quantization noise.
    /// <para>
    /// One bound per type, because the spread between them is an order of magnitude and a single
    /// number is either useless on Q8_0 or red on everything else. These are the measured worst
    /// tensor plus headroom, not guesses: Q8_0 3.4e-2, Q5_1 2.0e-1, Q4_0 3.25e-1 on the
    /// published checkpoints. The error does not grow with depth in any of them — the last layer
    /// is consistently the tightest — so a bound that a *later* layer breaches is a real defect
    /// however loose it looks here.
    /// </para>
    /// </summary>
    public static double ParityTolerance(string modelPath) => WeightType(modelPath) switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.BF16   => 0.02,
        GgmlType.Q8_0                                   => 0.05,
        GgmlType.Q6_K or GgmlType.Q5_K                  => 0.25,
        GgmlType.Q5_1 or GgmlType.Q5_0                  => 0.25,
        GgmlType.Q4_1 or GgmlType.Q4_0 or GgmlType.Q4_K => 0.40,
        //Anything coarser than 4 bits is not something the reference was ever compared against.
        _                                               => 0.60,
    };
}
