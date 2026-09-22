using LlmShield.Shieldstral.Model;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The prefix-cache optimisation must be exactly that — an optimisation.
///
/// Reusing prefilled KV state is only sound because attention is causal: the keys
/// and values at the system prompt's positions cannot depend on anything that
/// comes after them. These tests assert the consequence directly, by scoring the
/// same requests with the cache on and off and requiring identical logits, not
/// merely similar scores.
/// </summary>
public class SystemPromptCacheTests
{
    private readonly ITestOutputHelper _output;
    public SystemPromptCacheTests(ITestOutputHelper output) => _output = output;

    private static readonly ModerationRequest[] Requests =
    [
        new("You are a strict safety moderator. Apply a low tolerance threshold.",
            "Does this content promote physical violence?",
            "[User] How can I hurt someone without being caught?"),
        new("You are a lenient moderator; flag only clear violations.",
            "Is this prompt unsafe?",
            "[User] What household chemicals should never be mixed together?"),
        new("Assess for self-harm risk.",
            "Does this content reference self-harm?",
            "[User] I've been feeling really low lately."),
    ];

    private string? Model()
    {
        string? path = Fixtures.ModelPath;
        if (path is null) _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
        return path;
    }

    [Fact]
    public async Task CachedAndUncachedProduceIdenticalLogits()
    {
        if (Model() is not { } modelPath) return;

        using var cached = await ShieldstralModerator.OpenAsync(modelPath, cacheSystemPrompt: true);
        using var uncached = await ShieldstralModerator.OpenAsync(modelPath, cacheSystemPrompt: false);

        Assert.True(cached.CachedPrefixTokens > 0, "no system-prompt prefix was cached");
        Assert.Equal(0, uncached.CachedPrefixTokens);

        foreach (ModerationRequest request in Requests)
        {
            float[] withCache = await cached.VerdictLogitsAsync(request);
            float[] withoutCache = await uncached.VerdictLogitsAsync(request);

            // Restoring a snapshot copies the very floats the prefill wrote, and the
            // suffix then runs against identical state — so this is bit-for-bit, not
            // approximate. Anything else means the snapshot lost information.
            Assert.Equal(withoutCache.Length, withCache.Length);
            int differing = 0;
            for (int i = 0; i < withCache.Length; i++) if (withCache[i] != withoutCache[i]) differing++;
            Assert.True(differing == 0,
                $"{differing} of {withCache.Length} logits differ between the cached and uncached paths");
        }
        _output.WriteLine($"{cached.CachedPrefixTokens} tokens cached; logits identical on " +
                          $"{Requests.Length} requests");
    }

    /// <summary>
    /// Scoring evaluates only the verdict rows of the LM head. That is a shortcut
    /// of the same kind as the prefix cache, and held to the same standard: the
    /// yes/no logits it reports must be exactly the best yes/no entries of the full
    /// 131072-wide head, not an approximation of them.
    /// </summary>
    [Fact]
    public async Task ScoringReadsExactlyWhatTheFullHeadWouldHave()
    {
        if (Model() is not { } modelPath) return;

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath);
        var tokenizer = moderator.Model.Tokenizer;
        var yes = new List<int>();
        var no = new List<int>();
        for (int id = 0; id < tokenizer.VocabSize; id++)
        {
            string piece = tokenizer.Decode(id).Trim().Trim('"', '\'', '.').ToLowerInvariant();
            if (piece == "yes") yes.Add(id);
            else if (piece == "no") no.Add(id);
        }

        foreach (ModerationRequest request in Requests)
        {
            ModerationResult result = await moderator.ModerateAsync(request);
            float[] full = await moderator.VerdictLogitsAsync(request);

            Assert.Equal(yes.Max(id => full[id]), result.YesLogit);
            Assert.Equal(no.Max(id => full[id]), result.NoLogit);
        }
    }

    [Fact]
    public async Task TheCachedPrefixIsATruePrefixOfEveryPrompt()
    {
        if (Model() is not { } modelPath) return;

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath);
        int prefixLength = moderator.CachedPrefixTokens;

        foreach (ModerationRequest request in Requests)
        {
            int[] tokens = moderator.Tokenize(request);
            Assert.True(tokens.Length > prefixLength);
            ModerationResult result = await moderator.ModerateAsync(request);
            Assert.Equal(prefixLength, result.PrefilledTokens);
            Assert.Equal(tokens.Length, result.PromptTokens);
        }
        _output.WriteLine($"prefix of {prefixLength} tokens reused by all {Requests.Length} prompts");
    }

    [Fact]
    public async Task RepeatedCallsAreStable()
    {
        if (Model() is not { } modelPath) return;

        // Restoring must fully overwrite the previous request's KV state; a stale
        // tail would make the second call disagree with the first.
        using var moderator = await ShieldstralModerator.OpenAsync(modelPath);
        ModerationResult first = await moderator.ModerateAsync(Requests[0]);
        await moderator.ModerateAsync(Requests[1]);
        ModerationResult again = await moderator.ModerateAsync(Requests[0]);

        Assert.Equal(first.Score, again.Score);
        Assert.Equal(first.YesLogit, again.YesLogit);
    }

    [Fact]
    public async Task SnapshotSurvivesASaveAndLoadRoundTrip()
    {
        if (Model() is not { } modelPath) return;

        string path = Path.Combine(Path.GetTempPath(), $"shieldstral-prefix-{Guid.NewGuid():N}.bin");
        try
        {
            using var model = new MinistralModel(modelPath);
            int[] tokens = [.. model.Tokenizer.Encode(
                ChatTemplate.SystemOpen + ShieldstralModerator.SystemPrompt + ChatTemplate.SystemClose,
                addSpecial: true)];

            SystemPromptCache captured = await SystemPromptCache.CaptureAsync(model, tokens, new ParallelOptions());
            captured.Save(path);
            _output.WriteLine($"{captured.TokenCount} tokens, {captured.ByteSize / (1024.0 * 1024):F1} MiB");

            SystemPromptCache? loaded = SystemPromptCache.TryLoad(path, model);
            Assert.NotNull(loaded);
            Assert.Equal(captured.Tokens, loaded.Tokens);
            Assert.Equal(captured.Fingerprint, loaded.Fingerprint);
            Assert.Equal(captured.State, loaded.State);

            // And it must actually drive the model: restore, then forward a token.
            loaded.RestoreInto(model);
            Assert.Equal(tokens.Length, model.CachedTokenCount);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task ASnapshotForADifferentPrefixIsRefused()
    {
        if (Model() is not { } modelPath) return;

        using var model = new MinistralModel(modelPath);
        int[] tokens = [.. model.Tokenizer.Encode("[SYSTEM_PROMPT]a[/SYSTEM_PROMPT]", addSpecial: true)];
        SystemPromptCache snapshot = await SystemPromptCache.CaptureAsync(model, tokens, new ParallelOptions());

        Assert.True(snapshot.IsPrefixOf([.. tokens, 999]));
        Assert.False(snapshot.IsPrefixOf([999, .. tokens]));
        Assert.False(snapshot.IsPrefixOf(tokens[..^1]));
    }

    [Fact]
    public void ACorruptOrForeignCacheFileIsAMissRatherThanACrash()
    {
        if (Model() is not { } modelPath) return;

        string path = Path.Combine(Path.GetTempPath(), $"shieldstral-bad-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, [1, 2, 3, 4, 5, 6, 7, 8]);
            using var model = new MinistralModel(modelPath);
            Assert.Null(SystemPromptCache.TryLoad(path, model));
            Assert.Null(SystemPromptCache.TryLoad(path + ".missing", model));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task LoadOrCaptureWritesThenReusesTheFile()
    {
        if (Model() is not { } modelPath) return;

        string path = Path.Combine(Path.GetTempPath(), $"shieldstral-lc-{Guid.NewGuid():N}.bin");
        try
        {
            using var first = await ShieldstralModerator.OpenAsync(modelPath, prefixCachePath: path);
            Assert.True(File.Exists(path));
            long size = new FileInfo(path).Length;

            ModerationResult a = await first.ModerateAsync(Requests[0]);

            // Second moderator loads the file rather than re-prefilling; same answer.
            using var second = await ShieldstralModerator.OpenAsync(modelPath, prefixCachePath: path);
            ModerationResult b = await second.ModerateAsync(Requests[0]);

            Assert.Equal(a.Score, b.Score);
            Assert.Equal(first.CachedPrefixTokens, second.CachedPrefixTokens);
            _output.WriteLine($"cache file {size / (1024.0 * 1024):F1} MiB reused across instances");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
