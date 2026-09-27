using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// A host puts a deadline on every verdict, so a cancelled request is an ordinary
/// event rather than a failure path. What matters is what it leaves behind: the
/// next request on the same instance must score exactly as if the cancelled one
/// had never started, and a token handed to the open must not outlive the open.
/// </summary>
public class ModerationCancellationTests
{
    private readonly ITestOutputHelper _output;
    public ModerationCancellationTests(ITestOutputHelper output) => _output = output;

    private static readonly ModerationRequest Short = new(
        "You are a strict safety moderator. Apply a low tolerance threshold.",
        "Does this content promote physical violence?",
        "[User] How can I hurt someone without being caught?");

    /// <summary>Long enough that a forward pass is still running when the deadline fires.</summary>
    private static readonly ModerationRequest Long = Short with
    {
        Document = string.Concat(Enumerable.Repeat("[User] Tell me about the history of the printing press. ", 60)),
    };

    private static readonly ParallelOptions OneWorker = new() { MaxDegreeOfParallelism = 1 };

    private string? Model()
    {
        string? path = Fixtures.ModelPath;
        if (path is null) _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
        return path;
    }

    [Fact]
    public async Task AnAlreadyCancelledTokenThrowsBeforeAnyWork()
    {
        if (Model() is not { } modelPath) return;

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath, options: OneWorker);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await moderator.ModerateAsync(Short, cancellationToken: cts.Token));
    }

    [Fact]
    public async Task ACancelledPassLeavesTheInstanceScoringExactlyAsBefore()
    {
        if (Model() is not { } modelPath) return;

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath, options: OneWorker);
        ModerationResult before = await moderator.ModerateAsync(Short);

        using (var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200)))
        {
            var started = System.Diagnostics.Stopwatch.StartNew();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                async () => await moderator.ModerateAsync(Long, cancellationToken: cts.Token));
            _output.WriteLine($"cancelled after {started.ElapsedMilliseconds} ms");

            // Observed between work chunks, not at the end of the pass.
            Assert.True(started.ElapsedMilliseconds < 5_000,
                $"cancellation took {started.ElapsedMilliseconds} ms to be observed");
        }

        ModerationResult after = await moderator.ModerateAsync(Short);
        Assert.Equal(before.YesLogit, after.YesLogit);
        Assert.Equal(before.NoLogit, after.NoLogit);
        Assert.Equal(before.PrefilledTokens, after.PrefilledTokens);
    }

    [Fact]
    public async Task TheOpenTokenIsNotRetainedByTheInstance()
    {
        if (Model() is not { } modelPath) return;

        using var cts = new CancellationTokenSource();
        using var moderator = await ShieldstralModerator.OpenAsync(modelPath, options: OneWorker, cancellationToken: cts.Token);
        cts.Cancel();

        ModerationResult result = await moderator.ModerateAsync(Short);
        Assert.InRange(result.Score, 0f, 1f);
        Assert.Same(OneWorker, moderator.ParallelOptions);
        Assert.False(OneWorker.CancellationToken.CanBeCanceled);
    }

    [Fact]
    public async Task EitherTheOptionsTokenOrThePerCallTokenCancels()
    {
        if (Model() is not { } modelPath) return;

        using var moderator = await ShieldstralModerator.OpenAsync(modelPath, options: OneWorker);
        using var fromOptions = new CancellationTokenSource();
        using var perCall = new CancellationTokenSource();
        var options = new ParallelOptions { MaxDegreeOfParallelism = 1, CancellationToken = fromOptions.Token };

        fromOptions.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await moderator.ModerateAsync(Short, options, perCall.Token));
    }
}
