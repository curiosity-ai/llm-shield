using System.Net;
using System.Net.Sockets;
using System.Text;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// A one-file HTTP server, written on a raw socket rather than <c>HttpListener</c> because the
/// interesting cases are all about framing: a body shorter than the announced size, a server that
/// ignores <c>Range</c>, a resource that changed between two attempts. <c>HttpListener</c> insists
/// on producing well-formed responses, which is precisely what the downloader has to survive not
/// getting.
/// </summary>
internal sealed class TestFileServer : IDisposable
{
    private readonly TcpListener _listener;
    private readonly CancellationTokenSource _stopping = new();
    private readonly Task _loop;
    private readonly Lock _sync = new();
    private readonly List<string> _requests = [];

    public TestFileServer(byte[] content)
    {
        Content = content;
        _listener = new TcpListener(IPAddress.Loopback, 0);
        _listener.Start();
        _loop = Task.Run(AcceptLoopAsync);
    }

    /// <summary>The bytes served. Replace it to simulate the file being republished.</summary>
    public byte[] Content { get; set; }

    /// <summary>The entity tag reported by HEAD, and the one the downloader records for resuming.</summary>
    public string ETag { get; set; } = "\"v1\"";

    /// <summary>When false, HEAD omits <c>Accept-Ranges</c> and GET ignores <c>Range</c>.</summary>
    public bool SupportsRange { get; set; } = true;

    /// <summary>When false, HEAD is answered with 405 — as models.curiosity.ai's object store does.</summary>
    public bool AnswerHead { get; set; } = true;

    /// <summary>When true, GET responses carry no <c>Content-Length</c> and are framed by closing the
    /// connection — the case where a short body looks like a clean end of stream.</summary>
    public bool OmitContentLength { get; set; }

    /// <summary>Body bytes to send before hanging up, consumed in order by the GETs they actually
    /// shorten — a probe asking for one byte does not use one up. An empty queue serves everything
    /// in full.</summary>
    public Queue<int> Truncations { get; } = new();

    /// <summary>Pause between 64 KB body chunks, to make a transfer last long enough to interrupt.</summary>
    public TimeSpan ChunkDelay { get; set; }

    public int Port => ((IPEndPoint)_listener.LocalEndpoint).Port;

    public string Url => $"http://127.0.0.1:{Port}/Shieldstral-1.0-3B-Q5_1.gguf";

    /// <summary>Every request as <c>"METHOD path"</c>, plus the <c>Range</c> value when one was sent.</summary>
    public IReadOnlyList<string> Requests
    {
        get { lock (_sync) return [.. _requests]; }
    }

    private async Task AcceptLoopAsync()
    {
        while (!_stopping.IsCancellationRequested)
        {
            TcpClient client;
            try { client = await _listener.AcceptTcpClientAsync(_stopping.Token); }
            catch (OperationCanceledException) { return; }
            catch (SocketException) { return; }

            _ = Task.Run(async () =>
            {
                using (client)
                {
                    try { await ServeAsync(client); }
                    catch (IOException) { /* the client hung up; that is one of the cases under test */ }
                    catch (SocketException) { }
                }
            });
        }
    }

    private async Task ServeAsync(TcpClient client)
    {
        await using NetworkStream stream = client.GetStream();

        string head = await ReadHeadersAsync(stream);
        if (head.Length == 0) return;

        string[] lines = head.Split("\r\n");
        string[] requestLine = lines[0].Split(' ');
        string method = requestLine[0];
        string path = requestLine.Length > 1 ? requestLine[1] : "/";
        string? range = lines.Skip(1)
            .FirstOrDefault(l => l.StartsWith("Range:", StringComparison.OrdinalIgnoreCase))
            ?.Split(':', 2)[1].Trim();

        lock (_sync) _requests.Add(range is null ? $"{method} {path}" : $"{method} {path} {range}");

        bool headOnly = method == "HEAD";
        if (headOnly && !AnswerHead)
        {
            await stream.WriteAsync(Encoding.ASCII.GetBytes(
                "HTTP/1.1 405 Method Not Allowed\r\nAllow: GET\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
            await stream.FlushAsync();
            return;
        }

        byte[] content = Content;
        long start = 0, last = content.Length - 1;
        bool partial = SupportsRange && range is not null && range.StartsWith("bytes=", StringComparison.Ordinal);
        if (partial)
        {
            string[] spec = range![6..].Split('-');
            long.TryParse(spec[0], out start);
            if (spec.Length > 1 && spec[1].Length > 0 && long.TryParse(spec[1], out long requestedEnd))
                last = Math.Min(requestedEnd, last);
            start = Math.Min(start, content.Length);
        }

        var header = new StringBuilder();
        header.Append(partial ? "HTTP/1.1 206 Partial Content\r\n" : "HTTP/1.1 200 OK\r\n");
        header.Append($"ETag: {ETag}\r\n");
        if (SupportsRange) header.Append("Accept-Ranges: bytes\r\n");
        if (partial) header.Append($"Content-Range: bytes {start}-{last}/{content.Length}\r\n");

        long bodyLength = last - start + 1;
        if (headOnly || !OmitContentLength) header.Append($"Content-Length: {bodyLength}\r\n");
        header.Append("Connection: close\r\n\r\n");

        await stream.WriteAsync(Encoding.ASCII.GetBytes(header.ToString()));
        if (headOnly)
        {
            await stream.FlushAsync();
            return;
        }

        int send = (int)bodyLength;
        lock (_sync)
        {
            if (Truncations.Count > 0 && Truncations.Peek() < bodyLength) send = Truncations.Dequeue();
        }

        const int chunk = 64 * 1024;
        for (int sent = 0; sent < send; sent += chunk)
        {
            if (sent > 0 && ChunkDelay > TimeSpan.Zero) await Task.Delay(ChunkDelay);
            await stream.WriteAsync(content.AsMemory((int)start + sent, Math.Min(chunk, send - sent)));
            await stream.FlushAsync();
        }
    }

    private static async Task<string> ReadHeadersAsync(NetworkStream stream)
    {
        var buffer = new byte[8192];
        int filled = 0;
        while (filled < buffer.Length)
        {
            int read = await stream.ReadAsync(buffer.AsMemory(filled));
            if (read == 0) break;
            filled += read;
            string sofar = Encoding.ASCII.GetString(buffer, 0, filled);
            int end = sofar.IndexOf("\r\n\r\n", StringComparison.Ordinal);
            if (end >= 0) return sofar[..end];
        }
        return Encoding.ASCII.GetString(buffer, 0, filled);
    }

    public void Dispose()
    {
        _stopping.Cancel();
        _listener.Stop();
        try { _loop.Wait(TimeSpan.FromSeconds(5)); } catch (AggregateException) { }
        _stopping.Dispose();
    }
}
