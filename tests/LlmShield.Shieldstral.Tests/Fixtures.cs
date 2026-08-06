using System.Reflection;
using System.Text.Json;

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

    /// <summary>True when the model in use is the Q8_0 build the numeric references were captured on.</summary>
    public static bool IsQ8Reference(string modelPath)
        => Path.GetFileName(modelPath).Contains("Q8_0", StringComparison.OrdinalIgnoreCase);
}
