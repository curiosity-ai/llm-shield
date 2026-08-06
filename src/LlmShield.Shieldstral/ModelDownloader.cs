using System.Globalization;
using System.Net;
using System.Net.Http.Headers;

namespace LlmShield.Shieldstral;

/// <summary>Progress snapshot for an in-progress model download, reported a couple of times per second.</summary>
/// <param name="DownloadedBytes">Bytes on disk so far. A resumed download counts the bytes it started with.</param>
/// <param name="TotalBytes">Full size of the file, or <c>null</c> when the server did not say.</param>
/// <param name="Fraction">Progress in <c>[0, 1]</c> when <paramref name="TotalBytes"/> is known; <c>0</c> otherwise.</param>
/// <param name="FileName">The file being written — the last segment of the local path.</param>
public sealed record DownloadProgress(long DownloadedBytes, long? TotalBytes, float Fraction, string FileName);

/// <summary>
/// The pre-converted Shieldstral weights published at models.curiosity.ai.
///
/// All three are the same model; they differ only in how the weights are packed.
/// Every one of them runs on the integer matmul path (<see cref="Numerics.IntegerDot"/>),
/// which is why these three and not, say, a k-quant.
/// </summary>
public enum ShieldstralQuantization
{
    /// <summary>2.4 GiB. The closest of the three to the original weights.</summary>
    Q5_1 = 0,

    /// <summary>2.2 GiB. Q5_1 without the per-block offset.</summary>
    Q5_0 = 1,

    /// <summary>1.8 GiB. The smallest and, measured, the fastest to decode.</summary>
    Q4_0 = 2,
}

/// <summary>
/// Fetches a Shieldstral GGUF from models.curiosity.ai and caches it on disk.
///
/// A GGUF is the whole model — weights, hyperparameters and vocabulary in one file —
/// so this is all the deployment there is; see <c>ShieldstralModerator.CreateAsync</c>
/// for the one-call version.
///
/// The download is resumable, including across process restarts: bytes go to a
/// <c>.download</c> sidecar and are only renamed into place once the file is complete,
/// so a path that exists is always a file worth memory-mapping. Restarting a 2.4 GiB
/// transfer from zero because a laptop lid closed is the failure this is built to avoid.
/// </summary>
public static class ModelDownloader
{
    /// <summary>Where the converted models are published.</summary>
    public const string BaseUrl = "https://models.curiosity.ai/shieldstral/";

    /// <summary>512 KB — big enough that the write syscall is not the bottleneck on a fast link.</summary>
    private const int BufferBytes = 512 * 1024;

    private const int ProgressIntervalMs = 500;

    /// <summary>Consecutive failures *without progress* tolerated before giving up. A retry that moves
    /// bytes resets the count, so a long download over a flaky link can survive any number of drops.</summary>
    private const int MaxStalledRetries = 8;

    /// <summary>One download at a time per process: two concurrent 2.4 GiB transfers finish no sooner
    /// than one after the other, and they compete for the same disk.</summary>
    private static readonly SemaphoreSlim OneDownloadAtATime = new(1, 1);

    /// <summary>No timeout worth the name applies to a multi-gigabyte body, so there effectively is none.
    /// Stalls are caught by the retry loop instead.</summary>
    private static readonly HttpClient Client = new() { Timeout = TimeSpan.FromDays(1) };

    /// <summary>True while a download started by this process is running.</summary>
    public static bool IsDownloading() => OneDownloadAtATime.CurrentCount < 1;

    /// <summary>The published file name for a quantization, e.g. <c>Shieldstral-1.0-3B-Q5_1.gguf</c>.</summary>
    public static string FileNameFor(ShieldstralQuantization quantization) => quantization switch
    {
        ShieldstralQuantization.Q5_1 => "Shieldstral-1.0-3B-Q5_1.gguf",
        ShieldstralQuantization.Q5_0 => "Shieldstral-1.0-3B-Q5_0.gguf",
        ShieldstralQuantization.Q4_0 => "Shieldstral-1.0-3B-Q4_0.gguf",
        _ => throw new ArgumentOutOfRangeException(
            nameof(quantization), quantization,
            "No model is published for this quantization. Convert one with tools/convert_shieldstral_to_gguf.py."),
    };

    /// <summary>The download URL for a quantization.</summary>
    public static string UrlFor(ShieldstralQuantization quantization) => BaseUrl + FileNameFor(quantization);

    /// <summary>
    /// Where a model is cached when the caller does not choose: a per-user folder under the
    /// system temp directory. Pass an explicit path to keep it somewhere that survives a
    /// temp sweep — on most machines these files are large enough to be worth keeping.
    /// </summary>
    public static string DefaultPathFor(ShieldstralQuantization quantization)
        => Path.Combine(Path.GetTempPath(), "LlmShield.Shieldstral", FileNameFor(quantization));

    /// <summary>
    /// Makes sure a Shieldstral GGUF is on disk and returns its path. A complete file at the
    /// destination is used as-is, so this is cheap to call on every start-up.
    /// </summary>
    /// <param name="quantization">Which published model to fetch. Defaults to <see cref="ShieldstralQuantization.Q5_1"/>.</param>
    /// <param name="downloadToPath">Where to cache it. Defaults to <see cref="DefaultPathFor"/>.</param>
    /// <param name="reportProgress">Optional progress callback (~2 Hz).</param>
    /// <param name="cancellationToken">Cancels the transfer; the partial file is kept and resumes next time.</param>
    public static async Task<string> EnsureModelAsync(
        ShieldstralQuantization quantization = ShieldstralQuantization.Q5_1,
        string? downloadToPath = null,
        Action<DownloadProgress>? reportProgress = null,
        CancellationToken cancellationToken = default)
    {
        string path = downloadToPath ?? DefaultPathFor(quantization);
        await DownloadFileAsync(UrlFor(quantization), path, reportProgress, cancellationToken).ConfigureAwait(false);
        return path;
    }

    /// <summary>
    /// Downloads <paramref name="url"/> to <paramref name="localPath"/>, resuming a partial
    /// transfer left by an earlier call or an earlier process. Returns immediately when the
    /// destination already exists.
    /// </summary>
    public static async Task DownloadFileAsync(
        string url,
        string localPath,
        Action<DownloadProgress>? reportProgress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);
        ArgumentException.ThrowIfNullOrWhiteSpace(localPath);
        if (!Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) || uri.Scheme is not ("https" or "http"))
            throw new InvalidOperationException($"Invalid model URL: '{url}'. Use a valid http(s) URL.");

        if (File.Exists(localPath)) return;

        localPath = Path.GetFullPath(localPath);
        string fileName = Path.GetFileName(localPath);
        string directory = Path.GetDirectoryName(localPath)
            ?? throw new ArgumentException($"'{localPath}' has no directory to download into.", nameof(localPath));
        string tempPath = localPath + ".download";
        string metaPath = tempPath + ".meta";

        // Declared out here so the progress callback at the bottom can see the size as it is
        // refined: the HEAD probe usually knows it, and the response headers always do.
        long? totalBytes = null;

        await OneDownloadAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Another caller may have finished this exact file while we waited on the gate.
            if (File.Exists(localPath)) return;
            Directory.CreateDirectory(directory);

            (long? probedLength, bool supportsRange, string? validator) = await ProbeAsync(url, cancellationToken).ConfigureAwait(false);
            totalBytes = probedLength;

            long resumeFrom = ResumePoint(tempPath, metaPath, totalBytes, validator, supportsRange);
            if (totalBytes is long complete && resumeFrom == complete)
            {
                // The last run wrote every byte and was interrupted before the rename.
                Finish(tempPath, localPath, metaPath);
                Report(complete);
                return;
            }

            EnsureRoomOnDisk(directory, fileName, tempPath, totalBytes, resumeFrom);
            WriteResumeMetadata(metaPath, totalBytes, validator);

            long received = resumeFrom;
            long lastReportTicks = 0;
            int stalledRetries = 0;
            byte[] buffer = new byte[BufferBytes];

            FileStream fileStream = new(
                tempPath, resumeFrom > 0 ? FileMode.Open : FileMode.Create,
                FileAccess.Write, FileShare.None, BufferBytes, useAsync: true);
            await using (fileStream.ConfigureAwait(false))
            {
                fileStream.SetLength(resumeFrom);
                fileStream.Position = resumeFrom;
                Report(received);

                while (true)
                {
                    long before = received;
                    HttpResponseMessage? response = null;
                    try
                    {
                        if (received > 0 && !supportsRange) Rewind();

                        using (var request = new HttpRequestMessage(HttpMethod.Get, url))
                        {
                            if (received > 0) request.Headers.Range = new RangeHeaderValue(received, null);
                            response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                        }
                        response.EnsureSuccessStatusCode();

                        // A server that ignores Range answers 200 with the whole body. Taking that
                        // as a continuation would splice the head of the file onto its own middle.
                        if (received > 0 && response.StatusCode != HttpStatusCode.PartialContent) Rewind();

                        if (response.Content.Headers.ContentLength is long length) totalBytes = received + length;

                        Stream content = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                        await using (content.ConfigureAwait(false))
                        {
                            int read;
                            while ((read = await content.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                            {
                                await fileStream.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                                received += read;

                                long now = Environment.TickCount64;
                                if (now - lastReportTicks >= ProgressIntervalMs)
                                {
                                    lastReportTicks = now;
                                    Report(received);
                                }
                            }
                        }

                        if (totalBytes is long expected && received < expected)
                        {
                            // The body ended early without an error. Throwing sends us round the
                            // loop to resume rather than accepting a truncated file as done.
                            throw new IOException(
                                $"Download of '{fileName}' ended prematurely: received {received} of {expected} bytes.");
                        }

                        await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
                        Report(received);
                        break;
                    }
                    catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                    {
                        throw;
                    }
                    catch (Exception ex) when (ex is HttpRequestException or IOException or TaskCanceledException)
                    {
                        ThrowIfDiskIsFull(ex, directory, fileName, tempPath, totalBytes, received);

                        // Only a retry that moved no bytes counts against the budget; otherwise a
                        // download that drops every few hundred megabytes could never finish.
                        stalledRetries = received > before ? 0 : stalledRetries + 1;
                        if (stalledRetries > MaxStalledRetries)
                        {
                            throw new IOException(
                                $"Downloading '{fileName}' failed {stalledRetries} times without receiving anything. " +
                                $"The partial file is kept at '{tempPath}' — try again later to resume it.", ex);
                        }

                        await Task.Delay(RetryDelay(stalledRetries), cancellationToken).ConfigureAwait(false);
                    }
                    finally
                    {
                        response?.Dispose();
                    }

                    void Rewind()
                    {
                        received = 0;
                        fileStream.SetLength(0);
                        fileStream.Position = 0;
                    }
                }
            }

            Finish(tempPath, localPath, metaPath);
        }
        finally
        {
            OneDownloadAtATime.Release();
        }

        void Report(long downloaded)
        {
            if (reportProgress is null) return;
            float fraction = totalBytes is long total && total > 0
                ? Math.Clamp(downloaded / (float)total, 0f, 1f)
                : 0f;
            reportProgress(new DownloadProgress(downloaded, totalBytes, fraction, fileName));
        }
    }

    // ------------------------------------------------------------------ internals

    /// <summary>
    /// Asks how big the file is, whether ranged requests work, and what identifies this version
    /// of it — the three things resuming needs to be safe.
    ///
    /// HEAD is the polite way to ask and the answer is free, but plenty of object stores answer it
    /// with 405 (models.curiosity.ai's does, allowing only GET/PUT/DELETE). Asking for the first
    /// byte of the file gets the same three answers out of a <c>Content-Range</c> header for the
    /// price of one byte. Only if both fail does the download go ahead blind, which costs the
    /// ability to resume but still works.
    /// </summary>
    private static async Task<(long? Length, bool SupportsRange, string? Validator)> ProbeAsync(
        string url, CancellationToken cancellationToken)
    {
        (long? Length, bool SupportsRange, string? Validator) head = await AskAsync(HttpMethod.Head, null).ConfigureAwait(false);
        return head.Length is not null ? head : await AskAsync(HttpMethod.Get, new RangeHeaderValue(0, 0)).ConfigureAwait(false);

        async Task<(long? Length, bool SupportsRange, string? Validator)> AskAsync(HttpMethod method, RangeHeaderValue? range)
        {
            try
            {
                using var request = new HttpRequestMessage(method, url);
                request.Headers.Range = range;
                using HttpResponseMessage response = await Client
                    .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                if (!response.IsSuccessStatusCode) return (null, false, null);

                bool partial = response.StatusCode == HttpStatusCode.PartialContent;
                long? length = partial
                    ? response.Content.Headers.ContentRange?.Length
                    : response.Content.Headers.ContentLength;

                string? validator = response.Headers.ETag?.Tag
                    ?? response.Content.Headers.LastModified?.UtcDateTime.ToString("O", CultureInfo.InvariantCulture);

                return (length, partial || response.Headers.AcceptRanges.Contains("bytes"), validator);
            }
            catch (HttpRequestException)
            {
                return (null, false, null);
            }
        }
    }

    /// <summary>
    /// How many bytes of a previous attempt can be trusted.
    ///
    /// Resuming into a file that has been *republished* since would produce a GGUF that is the
    /// right length and quietly wrong, so a partial file is only reused when the size and the
    /// server's validator still match what was recorded when it was written.
    /// </summary>
    private static long ResumePoint(
        string tempPath, string metaPath, long? totalBytes, string? validator, bool supportsRange)
    {
        if (!File.Exists(tempPath)) return 0;

        long have = new FileInfo(tempPath).Length;
        bool usable =
            have > 0 &&
            supportsRange &&
            (totalBytes is not long total || have <= total) &&
            MetadataMatches(metaPath, totalBytes, validator);

        if (usable) return have;

        Delete(tempPath);
        Delete(metaPath);
        return 0;
    }

    private static bool MetadataMatches(string metaPath, long? totalBytes, string? validator)
    {
        // Nothing to compare against: a partial file from before this sidecar existed, or a
        // server that offers neither an ETag nor a Last-Modified. Start over rather than guess.
        if (validator is null || totalBytes is null) return false;
        if (!File.Exists(metaPath)) return false;

        try
        {
            string[] parts = File.ReadAllText(metaPath).Split('\t');
            return parts.Length == 2
                && long.TryParse(parts[0], CultureInfo.InvariantCulture, out long length)
                && length == totalBytes
                && parts[1] == validator;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static void WriteResumeMetadata(string metaPath, long? totalBytes, string? validator)
    {
        if (totalBytes is not long total || validator is null) return;
        try
        {
            File.WriteAllText(metaPath, $"{total.ToString(CultureInfo.InvariantCulture)}\t{validator}");
        }
        catch (IOException)
        {
            // Losing the sidecar costs a restart from zero next time, not correctness.
        }
    }

    /// <summary>Renames the finished download into place and clears the resume bookkeeping.</summary>
    private static void Finish(string tempPath, string localPath, string metaPath)
    {
        File.Move(tempPath, localPath, overwrite: true);
        Delete(metaPath);
    }

    /// <summary>
    /// Refuses to start a transfer that cannot fit. Discovering this three quarters of the way
    /// through a 2.4 GiB download wastes both the bandwidth and the user's afternoon.
    /// </summary>
    private static void EnsureRoomOnDisk(
        string directory, string fileName, string tempPath, long? totalBytes, long resumeFrom)
    {
        if (totalBytes is not long total) return;
        long needed = total - resumeFrom;
        long? free = FreeSpace(directory);
        if (free is not long available || available >= needed) return;

        throw new IOException(
            $"Not enough disk space for '{fileName}': {Human(needed)} still to download but only " +
            $"{Human(available)} free on '{directory}'." +
            (resumeFrom > 0 ? $" The {Human(resumeFrom)} already fetched is kept at '{tempPath}'." : string.Empty));
    }

    /// <summary>
    /// Turns the write failure a full disk produces into a message that says so.
    ///
    /// The errno does not survive into a portable exception, so this asks the file system
    /// instead — which is the number the user needs to see either way.
    /// </summary>
    private static void ThrowIfDiskIsFull(
        Exception ex, string directory, string fileName, string tempPath, long? totalBytes, long received)
    {
        if (ex is not IOException) return;
        if (FreeSpace(directory) is not long free || free > BufferBytes * 4L) return;

        string remaining = totalBytes is long total ? $" about {Human(total - received)} more is needed;" : string.Empty;
        throw new IOException(
            $"Ran out of disk space writing '{fileName}' after {Human(received)}:{remaining} " +
            $"only {Human(free)} is free on '{directory}'. The partial download is kept at " +
            $"'{tempPath}' — free some space and try again to resume it.", ex);
    }

    private static long? FreeSpace(string directory)
    {
        try
        {
            var drive = new DriveInfo(Path.GetPathRoot(directory) ?? directory);
            return drive.IsReady ? drive.AvailableFreeSpace : null;
        }
        catch (Exception e) when (e is ArgumentException or DriveNotFoundException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Backs off 2, 4, 8 … seconds, capped at half a minute.</summary>
    private static TimeSpan RetryDelay(int stalledRetries)
        => TimeSpan.FromSeconds(Math.Min(30, 1 << Math.Min(stalledRetries, 5)));

    private static void Delete(string path)
    {
        try { File.Delete(path); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }

    private static string Human(long bytes)
    {
        string[] units = ["B", "KiB", "MiB", "GiB"];
        double value = bytes;
        int unit = 0;
        while (value >= 1024 && unit < units.Length - 1) { value /= 1024; unit++; }
        return unit == 0
            ? $"{bytes} B"
            : string.Create(CultureInfo.InvariantCulture, $"{value:F1} {units[unit]}");
    }
}
