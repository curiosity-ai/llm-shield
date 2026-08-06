// Derived from TensorSharp's BpeTokenizer (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause. See third-party/TensorSharp-LICENSE.
using System.Text;
using System.Text.RegularExpressions;
using LlmShield.Shieldstral.Gguf;

namespace LlmShield.Shieldstral.Tokenization;

/// <summary>
/// Mistral's "tekken" byte-level BPE, driven entirely by the vocabulary and
/// merge table carried in the GGUF.
///
/// Two details make this tokenizer specific rather than generic GPT-2 BPE:
/// the pre-tokenizer regex (tekken splits digits one at a time and lets a
/// punctuation run absorb trailing <c>/</c>), and the fact that control tokens
/// such as <c>[SYSTEM_PROMPT]</c> are matched as literal strings before any
/// merging happens, so a prompt template's markers always land on their own ids.
/// </summary>
public sealed partial class TekkenTokenizer
{
    // llama.cpp's LLAMA_VOCAB_PRE_TYPE_TEKKEN pattern. Note `\p{N}` alone: tekken
    // emits one token per digit, unlike the `\p{N}{1,3}` of the GPT-4o family.
    [GeneratedRegex(
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|" +
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|" +
        @"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")]
    private static partial Regex TekkenPreTokenizer();

    private const int TokenTypeControl = 3;
    private const int TokenTypeUserDefined = 4;

    private readonly string[] _vocab;
    private readonly Dictionary<string, int> _vocabLookup;
    private readonly Dictionary<string, int> _mergeRanks;
    private readonly string[] _specialTokens;
    private readonly HashSet<int> _eosIds;
    private readonly Regex _preTokenizer;

    public IReadOnlyList<string> Vocab => _vocab;
    public int VocabSize => _vocab.Length;
    public int BosTokenId { get; }
    public IReadOnlyList<int> EosTokenIds { get; }
    public bool AddBosByDefault { get; }
    public bool AddEosByDefault { get; }

    public TekkenTokenizer(
        string[] vocab,
        int[]? tokenTypes,
        string[] merges,
        int bosTokenId,
        int[] eosTokenIds,
        bool addBos,
        bool addEos)
    {
        _vocab = vocab;
        BosTokenId = bosTokenId;
        EosTokenIds = eosTokenIds;
        _eosIds = [.. eosTokenIds];
        AddBosByDefault = addBos;
        AddEosByDefault = addEos;
        _preTokenizer = TekkenPreTokenizer();

        _vocabLookup = new Dictionary<string, int>(vocab.Length, StringComparer.Ordinal);
        for (int i = 0; i < vocab.Length; i++)
            _vocabLookup.TryAdd(vocab[i], i);

        _mergeRanks = new Dictionary<string, int>(merges.Length, StringComparer.Ordinal);
        for (int i = 0; i < merges.Length; i++)
            _mergeRanks.TryAdd(merges[i], i);

        var special = new List<string>();
        for (int i = 0; i < vocab.Length; i++)
        {
            bool isControl = tokenTypes is not null && i < tokenTypes.Length
                && tokenTypes[i] is TokenTypeControl or TokenTypeUserDefined;
            if (isControl || _eosIds.Contains(i))
                special.Add(vocab[i]);
        }
        // Longest first, so "[/SYSTEM_PROMPT]" is never shadowed by a shorter marker
        // that happens to be a prefix of it.
        special.Sort(static (a, b) => b.Length.CompareTo(a.Length));
        _specialTokens = [.. special];
    }

    /// <summary>Builds the tokenizer described by a GGUF's <c>tokenizer.ggml.*</c> metadata.</summary>
    public static TekkenTokenizer FromGguf(GgufFile gguf)
    {
        string[] vocab = gguf.GetStringArray("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException($"{gguf.Path} has no tokenizer.ggml.tokens.");
        string[] merges = gguf.GetStringArray("tokenizer.ggml.merges") ?? [];
        int[]? types = gguf.GetInt32Array("tokenizer.ggml.token_type");

        int bos = gguf.GetInt32("tokenizer.ggml.bos_token_id", 1);
        int eos = gguf.GetInt32("tokenizer.ggml.eos_token_id", 2);
        var eosIds = new List<int> { eos };
        if (gguf.GetInt32Array("tokenizer.ggml.eos_token_ids") is { } extra)
            foreach (int id in extra) if (!eosIds.Contains(id)) eosIds.Add(id);

        // Mistral prompts always open with exactly one BOS; the template itself
        // emits no `<s>`, so the tokenizer has to supply it.
        bool addBos = gguf.GetBool("tokenizer.ggml.add_bos_token", true);
        bool addEos = gguf.GetBool("tokenizer.ggml.add_eos_token", false);

        string model = gguf.GetString("tokenizer.ggml.model", "gpt2")!;
        if (model is not ("gpt2" or "tekken"))
            throw new NotSupportedException(
                $"{gguf.Path} declares tokenizer model '{model}'; this runtime implements tekken/gpt2 byte-level BPE only.");

        return new TekkenTokenizer(vocab, types, merges, bos, [.. eosIds], addBos, addEos);
    }

    /// <summary>
    /// Encodes text to token ids. <paramref name="addSpecial"/> controls whether the
    /// checkpoint's BOS/EOS are added; control markers embedded in the text are
    /// always recognised regardless.
    /// </summary>
    public List<int> Encode(string text, bool addSpecial = true)
    {
        var ids = new List<int>(text.Length / 3 + 8);
        if (addSpecial && AddBosByDefault) ids.Add(BosTokenId);

        foreach ((string chunk, int specialId) in SplitOnSpecialTokens(text))
        {
            if (specialId >= 0) { ids.Add(specialId); continue; }
            foreach (Match match in _preTokenizer.Matches(chunk))
                EncodePiece(match.ValueSpan, ids);
        }

        if (addSpecial && AddEosByDefault && EosTokenIds.Count > 0) ids.Add(EosTokenIds[0]);
        return ids;
    }

    private IEnumerable<(string Text, int SpecialId)> SplitOnSpecialTokens(string text)
    {
        var pending = new List<(string Text, int SpecialId)> { (text, -1) };
        foreach (string marker in _specialTokens)
        {
            if (!_vocabLookup.TryGetValue(marker, out int id)) continue;

            var next = new List<(string, int)>(pending.Count);
            foreach ((string body, int already) in pending)
            {
                if (already >= 0) { next.Add((body, already)); continue; }

                int cursor = 0;
                while (true)
                {
                    int hit = body.IndexOf(marker, cursor, StringComparison.Ordinal);
                    if (hit < 0)
                    {
                        if (cursor < body.Length) next.Add((body[cursor..], -1));
                        break;
                    }
                    if (hit > cursor) next.Add((body[cursor..hit], -1));
                    next.Add((marker, id));
                    cursor = hit + marker.Length;
                }
            }
            pending = next;
        }
        return pending;
    }

    private void EncodePiece(ReadOnlySpan<char> piece, List<int> ids)
    {
        string normalized = ByteEncode(piece);
        if (_vocabLookup.TryGetValue(normalized, out int direct)) { ids.Add(direct); return; }
        MergeGreedy(normalized, ids);
    }

    /// <summary>
    /// GPT-2's byte→printable-codepoint alphabet: control bytes and space move up
    /// by 0x100, the 0x7F-0xA0 band by 0xA2, and the soft hyphen to U+0143. The
    /// vocabulary is stored in this alphabet, so text has to enter it before any
    /// lookup.
    /// </summary>
    private static string ByteEncode(ReadOnlySpan<char> piece)
    {
        int maxBytes = Encoding.UTF8.GetMaxByteCount(piece.Length);
        byte[]? rented = maxBytes > 512 ? System.Buffers.ArrayPool<byte>.Shared.Rent(maxBytes) : null;
        Span<byte> bytes = rented ?? stackalloc byte[512];
        try
        {
            int written = Encoding.UTF8.GetBytes(piece, bytes);
            var sb = new StringBuilder(written);
            for (int i = 0; i < written; i++)
            {
                char c = (char)bytes[i];
                if (c == (char)0x00AD) c = (char)0x0143;
                else if (c <= (char)0x0020) c = (char)(c + 0x0100);
                else if (c is >= (char)0x007F and <= (char)0x00A0) c = (char)(c + 0x00A2);
                sb.Append(c);
            }
            return sb.ToString();
        }
        finally
        {
            if (rented is not null) System.Buffers.ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Standard BPE: repeatedly apply the lowest-ranked adjacent merge. The
    /// candidate heap can hold entries whose neighbours have since merged, so
    /// each pop re-checks adjacency and re-derives the rank before applying —
    /// without that, a stale candidate can jump ahead of a newly created,
    /// higher-priority merge and produce a different (wrong) split.
    /// </summary>
    private void MergeGreedy(string normalized, List<int> ids)
    {
        int n = normalized.Length;
        if (n == 0) return;
        if (n == 1)
        {
            if (_vocabLookup.TryGetValue(normalized, out int single)) ids.Add(single);
            return;
        }

        var text = new string[n];
        var prev = new int[n];
        var next = new int[n];
        var alive = new bool[n];
        for (int i = 0; i < n; i++)
        {
            text[i] = normalized[i].ToString();
            prev[i] = i - 1;
            next[i] = i + 1;
            alive[i] = true;
        }

        var queue = new PriorityQueue<(int A, int B, int Rank), long>();
        void Offer(int a, int b)
        {
            if (a < 0 || b >= n) return;
            int rank = MergeRank(text[a], text[b]);
            if (rank >= 0) queue.Enqueue((a, b, rank), ((long)rank << 20) | (uint)a);
        }
        for (int i = 0; i + 1 < n; i++) Offer(i, i + 1);

        while (queue.TryDequeue(out (int A, int B, int Rank) cand, out _))
        {
            int a = cand.A, b = cand.B;
            if (!alive[a] || !alive[b]) continue;
            if (next[a] != b || prev[b] != a) continue;
            if (MergeRank(text[a], text[b]) != cand.Rank) continue;

            string merged = text[a] + text[b];
            if (!_vocabLookup.ContainsKey(merged)) continue;

            text[a] = merged;
            next[a] = next[b];
            alive[b] = false;
            if (next[a] < n) prev[next[a]] = a;

            Offer(prev[a], a);
            Offer(a, next[a]);
        }

        for (int i = 0; i < n; i++)
        {
            if (!alive[i]) continue;
            if (_vocabLookup.TryGetValue(text[i], out int id)) ids.Add(id);
        }
    }

    private int MergeRank(string left, string right)
        => _mergeRanks.TryGetValue(left + " " + right, out int rank) ? rank : -1;

    /// <summary>Appends a token's raw bytes, undoing the GPT-2 byte alphabet.</summary>
    public void AppendTokenBytes(int tokenId, List<byte> buffer)
    {
        if ((uint)tokenId >= (uint)_vocab.Length)
            throw new ArgumentOutOfRangeException(nameof(tokenId));

        foreach (char c in _vocab[tokenId])
        {
            if (c is >= (char)0x0100 and <= (char)0x0120) { buffer.Add((byte)(c - 0x0100)); continue; }
            if (c is >= (char)0x0121 and <= (char)0x0142) { buffer.Add((byte)(c - 0x00A2)); continue; }
            if (c == 0x0143) { buffer.Add(0xAD); continue; }
            if (c == 0x2581) { buffer.Add(0x20); continue; }
            if (c < (char)0x100) { buffer.Add((byte)c); continue; }
            buffer.AddRange(Encoding.UTF8.GetBytes([c]));
        }
    }

    public string Decode(IEnumerable<int> ids)
    {
        var bytes = new List<byte>(64);
        foreach (int id in ids) AppendTokenBytes(id, bytes);
        return Encoding.UTF8.GetString(CollectionsMarshalHelper(bytes));
    }

    public string Decode(int id) => Decode(new[] { id });

    private static byte[] CollectionsMarshalHelper(List<byte> list) => [.. list];

    public bool IsEos(int tokenId) => _eosIds.Contains(tokenId);

    /// <summary>Id of a literal token string, or -1 when the vocabulary has no such entry.</summary>
    public int Lookup(string token) => _vocabLookup.TryGetValue(token, out int id) ? id : -1;
}
