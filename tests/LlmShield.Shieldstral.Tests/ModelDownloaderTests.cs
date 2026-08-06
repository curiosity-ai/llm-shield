using Xunit;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Everything here runs against a loopback socket, never models.curiosity.ai — the point is the
/// downloader's behaviour when a transfer goes wrong, and a 2.4 GiB happy path proves none of it.
/// </summary>
public sealed class ModelDownloaderTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), "llm-shield-download-tests", Guid.NewGuid().ToString("N"));

    private string Destination => Path.Combine(_directory, "Shieldstral-1.0-3B-Q5_1.gguf");
    private string Partial => Destination + ".download";
    private string Metadata => Partial + ".meta";

    public ModelDownloaderTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch (IOException) { }
    }

    /// <summary>Deterministic bytes, so a splice or an off-by-one shows up as a content mismatch.</summary>
    private static byte[] Payload(int length)
    {
        var bytes = new byte[length];
        for (int i = 0; i < length; i++) bytes[i] = (byte)(i * 31 + (i >> 8));
        return bytes;
    }

    [Fact]
    public async Task DownloadsAFileAndLeavesNoTemporaries()
    {
        byte[] payload = Payload(64 * 1024);
        using var server = new TestFileServer(payload);

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.False(File.Exists(Partial));
        Assert.False(File.Exists(Metadata));
    }

    [Fact]
    public async Task ReportsProgressEndingAtTheFullSize()
    {
        byte[] payload = Payload(64 * 1024);
        using var server = new TestFileServer(payload);
        var reports = new List<DownloadProgress>();

        await ModelDownloader.DownloadFileAsync(server.Url, Destination, reports.Add);

        Assert.NotEmpty(reports);
        Assert.All(reports, r => Assert.Equal("Shieldstral-1.0-3B-Q5_1.gguf", r.FileName));
        Assert.All(reports, r => Assert.Equal(payload.Length, r.TotalBytes));
        DownloadProgress last = reports[^1];
        Assert.Equal(payload.Length, last.DownloadedBytes);
        Assert.Equal(1f, last.Fraction);
    }

    [Fact]
    public async Task AnExistingFileIsNotFetchedAgain()
    {
        using var server = new TestFileServer(Payload(1024));
        await File.WriteAllBytesAsync(Destination, [1, 2, 3]);

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(new byte[] { 1, 2, 3 }, await File.ReadAllBytesAsync(Destination));
        Assert.Empty(server.Requests);
    }

    /// <summary>
    /// A body that stops early but *cleanly* — no error, just a short read — is the failure that
    /// silently produces a truncated model. It has to be caught by size, not by an exception.
    /// </summary>
    [Fact]
    public async Task AShortBodyIsResumedRatherThanAccepted()
    {
        byte[] payload = Payload(64 * 1024);
        using var server = new TestFileServer(payload) { OmitContentLength = true };
        server.Truncations.Enqueue(20_000);

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.Contains(server.Requests, r => r.Contains("bytes=20000-", StringComparison.Ordinal));
    }

    /// <summary>A connection dropped mid-body resumes from where it stopped, not from zero.</summary>
    [Fact]
    public async Task ADroppedConnectionResumesFromTheBytesOnDisk()
    {
        byte[] payload = Payload(256 * 1024);
        using var server = new TestFileServer(payload);
        server.Truncations.Enqueue(100_000);

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.Contains(server.Requests, r => r.Contains("bytes=100000-", StringComparison.Ordinal));
    }

    /// <summary>
    /// A partial file left by an earlier process is picked up where it stopped — the reason the
    /// bytes go to a sidecar instead of being thrown away when the process exits.
    /// </summary>
    [Fact]
    public async Task ResumesAPartialFileFromAnEarlierRun()
    {
        byte[] payload = Payload(64 * 1024);
        using var server = new TestFileServer(payload);
        await File.WriteAllBytesAsync(Partial, payload[..30_000]);
        await File.WriteAllTextAsync(Metadata, $"{payload.Length}\t{server.ETag}");

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.Contains(server.Requests, r => r.Contains("bytes=30000-", StringComparison.Ordinal));
        Assert.DoesNotContain(server.Requests, r => r.StartsWith("GET", StringComparison.Ordinal) && !r.Contains("bytes=", StringComparison.Ordinal));
    }

    /// <summary>
    /// The dangerous resume: the file was republished, so the leftover bytes belong to a different
    /// model. Resuming would produce something the right length and wrong throughout, which is why
    /// the recorded entity tag has to be checked before any of it is reused.
    /// </summary>
    [Fact]
    public async Task DiscardsAPartialFileWhenTheResourceChanged()
    {
        byte[] payload = Payload(64 * 1024);
        using var server = new TestFileServer(payload) { ETag = "\"v2\"" };
        await File.WriteAllBytesAsync(Partial, new byte[30_000]);
        await File.WriteAllTextAsync(Metadata, $"{payload.Length}\t\"v1\"");

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.DoesNotContain(server.Requests, r => r.Contains("bytes=", StringComparison.Ordinal));
    }

    /// <summary>A partial file with no record of what it came from is not trusted either.</summary>
    [Fact]
    public async Task DiscardsAPartialFileWithNoMetadata()
    {
        byte[] payload = Payload(64 * 1024);
        using var server = new TestFileServer(payload);
        await File.WriteAllBytesAsync(Partial, new byte[30_000]);

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.DoesNotContain(server.Requests, r => r.Contains("bytes=", StringComparison.Ordinal));
    }

    /// <summary>
    /// A server that answers a ranged request with the whole file again must not have that body
    /// appended to what is already on disk.
    /// </summary>
    [Fact]
    public async Task StartsOverWhenTheServerIgnoresRange()
    {
        byte[] payload = Payload(128 * 1024);
        using var server = new TestFileServer(payload) { SupportsRange = false };
        server.Truncations.Enqueue(50_000);

        await ModelDownloader.DownloadFileAsync(server.Url, Destination);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
    }

    /// <summary>
    /// The object store the models are published on answers HEAD with 405, so the size, the entity
    /// tag and range support all have to come out of a one-byte ranged GET instead. Losing that
    /// fallback would not fail anything visibly — it would just quietly stop resuming.
    /// </summary>
    [Fact]
    public async Task LearnsTheSizeFromARangedGetWhenHeadIsRefused()
    {
        byte[] payload = Payload(256 * 1024);
        using var server = new TestFileServer(payload) { AnswerHead = false };
        server.Truncations.Enqueue(100_000);
        var reports = new List<DownloadProgress>();

        await ModelDownloader.DownloadFileAsync(server.Url, Destination, reports.Add);

        Assert.Equal(payload, await File.ReadAllBytesAsync(Destination));
        Assert.All(reports, r => Assert.Equal(payload.Length, r.TotalBytes));
        Assert.Contains(server.Requests, r => r.Contains("bytes=0-0", StringComparison.Ordinal));
        Assert.Contains(server.Requests, r => r.Contains("bytes=100000-", StringComparison.Ordinal));
    }

    [Fact]
    public async Task RejectsANonHttpUrl()
    {
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => ModelDownloader.DownloadFileAsync("file:///etc/passwd", Destination));
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => ModelDownloader.DownloadFileAsync("models.curiosity.ai/x.gguf", Destination));
    }

    /// <summary>
    /// Interrupting a transfer must never leave anything at the destination path: everything else
    /// in the library treats a file that is there as a file worth memory-mapping.
    /// </summary>
    [Fact]
    public async Task CancellationLeavesNothingAtTheDestination()
    {
        byte[] payload = Payload(4 * 1024 * 1024);
        using var server = new TestFileServer(payload) { ChunkDelay = TimeSpan.FromMilliseconds(20) };
        using var cancellation = new CancellationTokenSource();

        Task download = ModelDownloader.DownloadFileAsync(
            server.Url, Destination,
            progress => { if (progress.DownloadedBytes > 0) cancellation.Cancel(); },
            cancellation.Token);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => download);
        Assert.False(File.Exists(Destination));
        Assert.True(File.Exists(Partial), "the bytes already fetched should be kept for the next attempt");
    }

    [Theory]
    [InlineData(ShieldstralQuantization.Q5_1, "Shieldstral-1.0-3B-Q5_1.gguf")]
    [InlineData(ShieldstralQuantization.Q5_0, "Shieldstral-1.0-3B-Q5_0.gguf")]
    [InlineData(ShieldstralQuantization.Q4_0, "Shieldstral-1.0-3B-Q4_0.gguf")]
    public void ResolvesThePublishedUrls(ShieldstralQuantization quantization, string expected)
    {
        Assert.Equal(expected, ModelDownloader.FileNameFor(quantization));
        Assert.Equal("https://models.curiosity.ai/shieldstral/" + expected, ModelDownloader.UrlFor(quantization));
        Assert.Equal(expected, Path.GetFileName(ModelDownloader.DefaultPathFor(quantization)));
    }

    [Fact]
    public void RejectsAQuantizationThatIsNotPublished()
        => Assert.Throws<ArgumentOutOfRangeException>(() => ModelDownloader.UrlFor((ShieldstralQuantization)99));
}
