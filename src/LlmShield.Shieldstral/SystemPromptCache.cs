using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using LlmShield.Shieldstral.Model;

namespace LlmShield.Shieldstral;

/// <summary>
/// Snapshot of the KV state produced by prefilling Shieldstral's fixed system
/// prompt.
///
/// Shieldstral is always driven with the same system message, and attention is
/// causal — so the keys and values for those leading positions depend on nothing
/// downstream. Prefilling them is pure repeated work: identical inputs, identical
/// weights, identical outputs, every single request. Computing that state once and
/// restoring it turns the per-request prefill into just the variable
/// instruct/query/document suffix.
///
/// The snapshot is only valid for the exact (weights, token prefix, cache
/// geometry) it was captured under, so <see cref="Fingerprint"/> ties it to all
/// three and a mismatch is refused rather than silently producing wrong logits.
/// </summary>
public sealed class SystemPromptCache
{
    private const uint MagicNumber = 0x53484B56;  // "SHKV"
    private const int FormatVersion = 1;

    /// <summary>Token ids the snapshot covers, in order.</summary>
    public int[] Tokens { get; }

    /// <summary>Flat KV state: keys for every layer and head, then values.</summary>
    public float[] State { get; }

    /// <summary>Identity of the model and prefix this snapshot belongs to.</summary>
    public string Fingerprint { get; }

    public int TokenCount => Tokens.Length;
    public long ByteSize => (long)State.Length * sizeof(float);

    private SystemPromptCache(int[] tokens, float[] state, string fingerprint)
    {
        Tokens = tokens;
        State = state;
        Fingerprint = fingerprint;
    }

    /// <summary>
    /// Prefills <paramref name="tokens"/> into a freshly reset cache and captures
    /// the result. The model's KV cache is left holding exactly that prefix.
    /// </summary>
    public static async ValueTask<SystemPromptCache> CaptureAsync(
        MinistralModel model, ReadOnlyMemory<int> tokens, ParallelOptions options)
    {
        model.ResetKvCache();
        await model.PrefillAsync(tokens, options).ConfigureAwait(false);
        return new SystemPromptCache(
            tokens.ToArray(),
            model.KvCache.Snapshot(tokens.Length),
            ComputeFingerprint(model, tokens.Span));
    }

    /// <summary>
    /// Restores the snapshot into <paramref name="model"/>, leaving its KV cache
    /// holding the prefix and nothing else. Cheap: a memcpy of a few megabytes
    /// against a prefill that is otherwise the whole model.
    /// </summary>
    public void RestoreInto(MinistralModel model)
    {
        string expected = ComputeFingerprint(model, Tokens);
        if (expected != Fingerprint)
            throw new InvalidOperationException(
                $"This snapshot was captured for a different model or prefix " +
                $"(fingerprint {Fingerprint}, model reports {expected}).");
        model.KvCache.Restore(State, Tokens.Length);
    }

    /// <summary>
    /// Whether <paramref name="tokens"/> begins with this snapshot's prefix, i.e.
    /// whether restoring it is valid for that prompt.
    /// </summary>
    public bool IsPrefixOf(ReadOnlySpan<int> tokens)
        => tokens.Length >= Tokens.Length && tokens[..Tokens.Length].SequenceEqual(Tokens);

    /// <summary>
    /// Ties a snapshot to the weights that produced it and the cache geometry it
    /// was laid out for. The model file's identity comes from its path, length and
    /// last-write time — hashing gigabytes of weights on every load would cost more
    /// than the prefill this cache exists to avoid.
    /// </summary>
    private static string ComputeFingerprint(MinistralModel model, ReadOnlySpan<int> tokens)
    {
        var info = new FileInfo(model.ModelPath);
        ModelConfig c = model.Config;
        var sb = new StringBuilder()
            .Append(info.Name).Append('|')
            .Append(info.Exists ? info.Length : -1).Append('|')
            .Append(info.Exists ? info.LastWriteTimeUtc.Ticks : -1).Append('|')
            .Append(c.Architecture).Append('|')
            .Append(c.LayerCount).Append('x').Append(c.KvHeadCount).Append('x').Append(c.HeadDim).Append('|')
            .Append(tokens.Length).Append(':');
        foreach (int t in tokens) sb.Append(t).Append(',');

        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()));
        return Convert.ToHexStringLower(digest);
    }

    // ------------------------------------------------------------ persistence

    /// <summary>Writes the snapshot so a later process can skip the prefill entirely.</summary>
    public void Save(string path)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(MagicNumber);
        writer.Write(FormatVersion);
        writer.Write(Fingerprint);
        writer.Write(Tokens.Length);
        foreach (int token in Tokens) writer.Write(token);
        writer.Write((long)State.Length);
        writer.Flush();

        // One bulk write of the float payload; a BinaryWriter loop over a few
        // million floats is measurably slower than the prefill it replaces.
        byte[] bytes = new byte[State.Length * sizeof(float)];
        Buffer.BlockCopy(State, 0, bytes, 0, bytes.Length);
        stream.Write(bytes);
    }

    /// <summary>
    /// Loads a snapshot previously written by <see cref="Save"/>, or returns null
    /// when the file is absent, unreadable, or belongs to a different model or
    /// prefix. A stale cache is a miss, not an error.
    /// </summary>
    public static SystemPromptCache? TryLoad(string path, MinistralModel model)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            if (reader.ReadUInt32() != MagicNumber) return null;
            if (reader.ReadInt32() != FormatVersion) return null;

            string fingerprint = reader.ReadString();
            int tokenCount = reader.ReadInt32();
            if (tokenCount is < 0 or > 1_000_000) return null;
            var tokens = new int[tokenCount];
            for (int i = 0; i < tokenCount; i++) tokens[i] = reader.ReadInt32();

            long stateLength = reader.ReadInt64();
            if (stateLength is < 0 or > int.MaxValue) return null;
            var state = new float[stateLength];
            byte[] bytes = new byte[stateLength * sizeof(float)];
            stream.ReadExactly(bytes);
            Buffer.BlockCopy(bytes, 0, state, 0, bytes.Length);

            var cache = new SystemPromptCache(tokens, state, fingerprint);
            if (ComputeFingerprint(model, tokens) != fingerprint) return null;
            return cache;
        }
        catch (Exception e) when (e is IOException or EndOfStreamException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Returns the snapshot for <paramref name="tokens"/>, loading it from
    /// <paramref name="path"/> when it is still valid and otherwise recomputing
    /// and writing it back.
    /// </summary>
    public static async ValueTask<SystemPromptCache> LoadOrCaptureAsync(
        MinistralModel model, ReadOnlyMemory<int> tokens, string path, ParallelOptions options)
    {
        SystemPromptCache? cached = TryLoad(path, model);
        if (cached is not null && cached.Tokens.AsSpan().SequenceEqual(tokens.Span))
            return cached;

        SystemPromptCache fresh = await CaptureAsync(model, tokens, options).ConfigureAwait(false);
        try { fresh.Save(path); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // A read-only or full cache directory costs a prefill, not a failure.
        }
        return fresh;
    }
}
