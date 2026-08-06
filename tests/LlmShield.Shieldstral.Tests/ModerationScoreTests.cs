using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// End-to-end safety scores against two independent references.
///
/// <c>tests/fixtures/scores.json</c> comes from the NumPy implementation running
/// the original bfloat16 weights. The <see cref="LlamaCppReference"/> numbers
/// below were captured from llama.cpp b8.18.1 on a Q8_0 GGUF of the same
/// checkpoint. Three implementations — one managed, one NumPy, one native C++ —
/// agreeing on a borderline score is far better evidence than any one of them
/// agreeing with itself.
/// </summary>
public class ModerationScoreTests
{
    private readonly ITestOutputHelper _output;
    public ModerationScoreTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// llama.cpp b8.18.1, /completion with n_probs=20 at temperature 0, on a Q8_0
    /// GGUF converted from mistralai/Shieldstral-1.0-3B.
    /// </summary>
    public static readonly Dictionary<string, double> LlamaCppReference = new()
    {
        ["violent-request"] = 0.997343,
        ["benign-cooking"] = 0.000000,
        ["borderline-sarcasm"] = 0.439276,
        ["borderline-fiction"] = 0.049938,
        ["borderline-lenient"] = 0.000885,
        ["borderline-selfharm"] = 0.025094,
    };

    /// <summary>
    /// Deliberately much looser than the measured agreement (worst case ~1e-2), so
    /// the test tracks "still matches the reference implementations" rather than
    /// "bit-identical to one build of one of them".
    /// </summary>
    private const double Tolerance = 0.05;

    private sealed record Case(
        string Name, string Instruct, string Query, string Document,
        double Score, double YesLogit, double NoLogit, int Tokens, string TopPiece);

    private static Case[] LoadCases()
    {
        using JsonDocument doc = Fixtures.Load("scores.json");
        return [.. doc.RootElement.EnumerateArray().Select(e => new Case(
            e.GetProperty("name").GetString()!,
            e.GetProperty("instruct").GetString()!,
            e.GetProperty("query").GetString()!,
            e.GetProperty("document").GetString()!,
            e.GetProperty("score").GetDouble(),
            e.GetProperty("yes_logit").GetDouble(),
            e.GetProperty("no_logit").GetDouble(),
            e.GetProperty("tokens").GetInt32(),
            e.GetProperty("top_piece").GetString()!))];
    }

    [Fact]
    public async Task ScoresMatchBothReferenceImplementations()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath);
        double worstNumpy = 0, worstLlamaCpp = 0;
        var failures = new List<string>();

        foreach (Case c in LoadCases())
        {
            ModerationResult result = await moderator.ModerateAsync(c.Instruct, c.Query, c.Document);

            Assert.Equal(c.Tokens, result.PromptTokens);
            // A non-finite logit means something overflowed upstream; the score would
            // then be NaN, which compares false against everything and could otherwise
            // slip past a tolerance check.
            Assert.True(float.IsFinite(result.YesLogit) && float.IsFinite(result.NoLogit),
                $"{c.Name}: verdict logits are yes={result.YesLogit} no={result.NoLogit}");
            Assert.True(float.IsFinite(result.Score) && result.Score is >= 0f and <= 1f,
                $"{c.Name}: score {result.Score} is not a probability");

            double deltaNumpy = Math.Abs(result.Score - c.Score);
            worstNumpy = Math.Max(worstNumpy, deltaNumpy);

            string line = $"{c.Name,-22} score={result.Score:F6} numpy={c.Score:F6} Δ={deltaNumpy:E2}";
            if (LlamaCppReference.TryGetValue(c.Name, out double llamaCpp))
            {
                double deltaLlamaCpp = Math.Abs(result.Score - llamaCpp);
                worstLlamaCpp = Math.Max(worstLlamaCpp, deltaLlamaCpp);
                line += $"  llama.cpp={llamaCpp:F6} Δ={deltaLlamaCpp:E2}";
                if (deltaLlamaCpp > Tolerance)
                    failures.Add($"{c.Name}: {result.Score:F6} vs llama.cpp {llamaCpp:F6}");
                // The safe/unsafe call must agree regardless of quantization.
                Assert.Equal(llamaCpp > 0.5, result.IsUnsafe);
            }
            _output.WriteLine(line);

            if (deltaNumpy > Tolerance)
                failures.Add($"{c.Name}: {result.Score:F6} vs NumPy {c.Score:F6}");
        }

        _output.WriteLine($"worst |Δ|: NumPy {worstNumpy:E2}, llama.cpp {worstLlamaCpp:E2}");
        Assert.True(failures.Count == 0,
            $"scores drifted beyond {Tolerance}:\n  " + string.Join("\n  ", failures));
    }

    /// <summary>
    /// Shieldstral is a single-token classifier: if the most likely next token is
    /// not "yes" or "no", the prompt framing or the tokenizer mapping has drifted,
    /// whatever the score says.
    /// </summary>
    [Fact]
    public async Task TheVerdictTokenIsAlwaysYesOrNo()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath);
        foreach (Case c in LoadCases())
        {
            string piece = (await moderator.TopVerdictTokenAsync(new ModerationRequest(c.Instruct, c.Query, c.Document)))
                .Trim().ToLowerInvariant();
            _output.WriteLine($"{c.Name,-22} '{piece}' (reference '{c.TopPiece}')");
            Assert.True(piece is "yes" or "no", $"{c.Name}: expected a yes/no verdict, got '{piece}'");
            Assert.Equal(c.TopPiece.Trim().ToLowerInvariant(), piece);
        }
    }

    [Fact]
    public async Task ConfigMatchesThePublishedHyperparameters()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath, cacheSystemPrompt: false);
        var config = moderator.Config;

        // mistralai/Shieldstral-1.0-3B/params.json
        Assert.Equal("mistral3", config.Architecture);
        Assert.Equal(26, config.LayerCount);
        Assert.Equal(3072, config.HiddenSize);
        Assert.Equal(32, config.HeadCount);
        Assert.Equal(8, config.KvHeadCount);
        Assert.Equal(128, config.HeadDim);
        Assert.Equal(9216, config.FeedForwardSize);
        Assert.Equal(131072, config.VocabSize);
        Assert.Equal(1_000_000f, config.RopeFreqBase);
        Assert.Equal(1e-5f, config.RmsNormEps, 8);

        // YaRN: factor 16 over a 16384-token training window, magnitude scaling off.
        Assert.True(config.YarnActive);
        Assert.Equal(16f, config.RopeScaleFactor, 3);
        Assert.Equal(16384, config.RopeOriginalContextLength);
        Assert.Equal(32f, config.YarnBetaFast, 3);
        Assert.Equal(1f, config.YarnBetaSlow, 3);

        // llama-4 attention temperature.
        Assert.Equal(0.1f, config.AttentionTemperatureScale, 5);
    }

    /// <summary>
    /// The scoring rule renormalises "yes" against "no" only, so the score must be
    /// invariant to a constant shift of every logit — which is what makes it stable
    /// across quantizations that move the whole distribution slightly.
    /// </summary>
    [Fact]
    public async Task ScoreIsMonotonicInTheLogitGap()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath);
        var scored = new List<ModerationResult>();
        foreach (Case c in LoadCases())
            scored.Add(await moderator.ModerateAsync(c.Instruct, c.Query, c.Document));

        var results = scored.OrderBy(r => r.YesLogit - r.NoLogit).ToArray();

        for (int i = 1; i < results.Length; i++)
            Assert.True(results[i].Score >= results[i - 1].Score,
                $"score {results[i].Score:F6} at gap {results[i].YesLogit - results[i].NoLogit:F4} " +
                $"is below {results[i - 1].Score:F6} at gap " +
                $"{results[i - 1].YesLogit - results[i - 1].NoLogit:F4}");
    }
}
