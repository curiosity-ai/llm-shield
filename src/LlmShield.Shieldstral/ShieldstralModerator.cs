using LlmShield.Shieldstral.Model;
using LlmShield.Shieldstral.Numerics;

namespace LlmShield.Shieldstral;

/// <summary>What Shieldstral was asked to judge.</summary>
/// <param name="Instruct">Evaluation context and how strict to be.</param>
/// <param name="Query">A single yes/no question about the content.</param>
/// <param name="Document">The content under review.</param>
public readonly record struct ModerationRequest(string Instruct, string Query, string Document);

/// <summary>Shieldstral's answer to one request.</summary>
/// <param name="Score">
/// Probability the content violates the policy, in [0, 1]. It is the softmax of
/// the "yes" and "no" logits against each other only — the rest of the vocabulary
/// is renormalised away, exactly as the model card's reference scorer does.
/// </param>
/// <param name="YesLogit">Best logit among the "yes" surface forms.</param>
/// <param name="NoLogit">Best logit among the "no" surface forms.</param>
/// <param name="PromptTokens">Tokens in the rendered prompt.</param>
/// <param name="PrefilledTokens">How many of those came from the cached system prompt.</param>
public readonly record struct ModerationResult(
    float Score, float YesLogit, float NoLogit, int PromptTokens, int PrefilledTokens)
{
    /// <summary>Verdict at the model card's 0.5 threshold.</summary>
    public bool IsUnsafe => Score > 0.5f;

    /// <summary>Verdict at a caller-chosen threshold.</summary>
    public bool IsUnsafeAt(float threshold) => Score > threshold;

    public override string ToString()
        => $"{(IsUnsafe ? "UNSAFE" : "safe")} score={Score:F6} (yes={YesLogit:F4} no={NoLogit:F4})";
}

/// <summary>
/// Shieldstral 1.0 3B as a safety classifier.
///
/// The model answers one yes/no question per forward pass: the entire verdict
/// lives in the distribution over the first generated token, so there is no
/// sampling loop and no generated text — just a prefill and a look at two logits.
///
/// Because the system prompt is fixed, its KV state is prefilled once and reused
/// for every request (see <see cref="SystemPromptCache"/>), leaving only the
/// caller's instruct/query/document to run through the model.
///
/// Not thread-safe — one instance owns one KV cache. Create one per worker.
/// </summary>
public sealed class ShieldstralModerator : IDisposable
{
    /// <summary>The system message Shieldstral was trained with. Changing it changes the model's behaviour.</summary>
    public const string SystemPrompt =
        "Judge whether the Document meets the requirements based on the Query " +
        "and the Instruction provided. Note that the answer can only be \"yes\" or \"no\".";

    /// <summary>Surface forms counted as an affirmative verdict, after trimming and lowercasing.</summary>
    private static readonly string[] YesForms = ["yes"];
    private static readonly string[] NoForms = ["no"];

    /// <summary>How deep into the verdict distribution to look for the yes/no tokens.</summary>
    private const int VerdictSearchDepth = 20;

    private readonly MinistralModel _model;
    private readonly bool _ownsModel;
    private readonly SystemPromptCache? _prefix;
    private readonly int[] _yesTokens;
    private readonly int[] _noTokens;

    public MinistralModel Model => _model;
    public ModelConfig Config => _model.Config;

    /// <summary>Tokens of the cached system-prompt prefix, or 0 when caching is off.</summary>
    public int CachedPrefixTokens => _prefix?.TokenCount ?? 0;

    /// <summary>
    /// Opens a Shieldstral GGUF.
    /// </summary>
    /// <param name="ggufPath">Path to the converted language model.</param>
    /// <param name="cacheSystemPrompt">
    /// Prefill and reuse the fixed system prompt's KV state. On by default; the
    /// results are identical either way, which <c>SystemPromptCacheTests</c> asserts.
    /// </param>
    /// <param name="prefixCachePath">
    /// Optional file to persist the prefix state in, so even the first request of a
    /// fresh process skips the system-prompt prefill.
    /// </param>
    public ShieldstralModerator(string ggufPath, bool cacheSystemPrompt = true, string? prefixCachePath = null)
        : this(new MinistralModel(ggufPath), ownsModel: true, cacheSystemPrompt, prefixCachePath) { }

    public ShieldstralModerator(
        MinistralModel model, bool ownsModel = false,
        bool cacheSystemPrompt = true, string? prefixCachePath = null)
    {
        _model = model;
        _ownsModel = ownsModel;

        // Rejecting a checkpoint here must also let go of it: the file stays memory-mapped
        // otherwise, and Windows will not delete a mapped file — which is what CreateAsync
        // does before fetching a corrupt model again.
        try
        {
            _yesTokens = ResolveVerdictTokens(YesForms);
            _noTokens = ResolveVerdictTokens(NoForms);
            if (_yesTokens.Length == 0 || _noTokens.Length == 0)
                throw new InvalidDataException(
                    "The model's vocabulary has no 'yes'/'no' tokens; this does not look like a Shieldstral checkpoint.");

            if (cacheSystemPrompt)
            {
                int[] prefix = BuildSystemPrefixTokens();
                _prefix = prefixCachePath is null
                    ? SystemPromptCache.Capture(_model, prefix)
                    : SystemPromptCache.LoadOrCapture(_model, prefix, prefixCachePath);
            }
        }
        catch
        {
            if (ownsModel) model.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Downloads a published Shieldstral GGUF if it is not already cached, then opens it.
    ///
    /// This is the whole deployment story: a GGUF carries the weights, the hyperparameters
    /// and the vocabulary, so there is no second file to find and nothing to configure.
    /// A model already on disk is opened without touching the network, which makes this
    /// safe to call on every start-up.
    /// </summary>
    /// <param name="quantization">Which published model to use. Defaults to <see cref="ShieldstralQuantization.Q5_1"/>.</param>
    /// <param name="modelUrl">Overrides the download URL — for a mirror, or a model you converted yourself.</param>
    /// <param name="downloadToPath">Where to cache the file. Defaults to <see cref="ModelDownloader.DefaultPathFor"/>.</param>
    /// <param name="cacheSystemPrompt">See the constructor.</param>
    /// <param name="prefixCachePath">See the constructor.</param>
    /// <param name="reportProgress">Optional download progress callback (~2 Hz).</param>
    /// <param name="cancellationToken">Cancels the download; a partial file is kept and resumes next time.</param>
    public static async Task<ShieldstralModerator> CreateAsync(
        ShieldstralQuantization quantization = ShieldstralQuantization.Q5_1,
        string? modelUrl = null,
        string? downloadToPath = null,
        bool cacheSystemPrompt = true,
        string? prefixCachePath = null,
        Action<DownloadProgress>? reportProgress = null,
        CancellationToken cancellationToken = default)
    {
        string url = modelUrl ?? ModelDownloader.UrlFor(quantization);
        string path = downloadToPath ?? ModelDownloader.DefaultPathFor(quantization);

        await ModelDownloader.DownloadFileAsync(url, path, reportProgress, cancellationToken).ConfigureAwait(false);
        try
        {
            return await OpenAsync().ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or NotSupportedException or KeyNotFoundException)
        {
            // The cached file is not a Shieldstral GGUF this build can read. Much the likeliest
            // cause is a file truncated by an older version, or by something outside this process
            // writing to the cache directory — so delete it, fetch it once more, and only then
            // conclude that the model itself is the problem.
            try { File.Delete(path); } catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
            await ModelDownloader.DownloadFileAsync(url, path, reportProgress, cancellationToken).ConfigureAwait(false);
            return await OpenAsync().ConfigureAwait(false);
        }

        // Opening is not I/O-bound — the weights are memory-mapped in milliseconds, but capturing
        // the system-prompt prefix runs a real prefill. Off the caller's thread it goes.
        Task<ShieldstralModerator> OpenAsync() => Task.Run(
            () => new ShieldstralModerator(path, cacheSystemPrompt, prefixCachePath), cancellationToken);
    }

    /// <summary>Scores one request.</summary>
    public ModerationResult Moderate(string instruct, string query, string document)
        => Moderate(new ModerationRequest(instruct, query, document));

    public ModerationResult Moderate(ModerationRequest request)
    {
        int[] tokens = Tokenize(request);
        int prefilled = PrepareCache(tokens);
        ReadOnlySpan<float> logits = _model.Forward(tokens.AsSpan(prefilled));
        return Score(logits, tokens.Length, prefilled);
    }

    /// <summary>
    /// Full logits for the verdict position. Exposed for callers that want the
    /// whole distribution (calibration work, or a different scoring rule).
    /// </summary>
    public float[] VerdictLogits(ModerationRequest request)
    {
        int[] tokens = Tokenize(request);
        int prefilled = PrepareCache(tokens);
        return _model.Forward(tokens.AsSpan(prefilled)).ToArray();
    }

    /// <summary>The exact prompt string sent to the tokenizer, for auditing.</summary>
    public string RenderPrompt(ModerationRequest request) => ChatTemplate.Render(
    [
        ChatMessage.System(SystemPrompt),
        ChatMessage.User(FormatUserMessage(request)),
    ]);

    /// <summary>
    /// The instruct/query/document framing from the model card. The exact
    /// separators are part of the trained format, not a display choice.
    /// </summary>
    public static string FormatUserMessage(ModerationRequest request)
        => $"<Instruct>: {request.Instruct}\n\n<Query>: {request.Query}\n\n<Document>: {request.Document}";

    public int[] Tokenize(ModerationRequest request)
        => [.. _model.Tokenizer.Encode(RenderPrompt(request), addSpecial: true)];

    // -------------------------------------------------------------- internals

    /// <summary>
    /// The invariant leading tokens: BOS plus the wrapped system message. Rendering
    /// a request with an empty body and cutting at the system marker guarantees this
    /// is a true prefix of every prompt, rather than a separately-tokenized string
    /// that might not align on a token boundary.
    /// </summary>
    private int[] BuildSystemPrefixTokens()
    {
        string systemOnly = ChatTemplate.SystemOpen + SystemPrompt + ChatTemplate.SystemClose;
        return [.. _model.Tokenizer.Encode(systemOnly, addSpecial: true)];
    }

    /// <summary>
    /// Positions the KV cache so that only the uncached suffix has to run, and
    /// returns how many leading tokens are already resident.
    /// </summary>
    private int PrepareCache(int[] tokens)
    {
        if (_prefix is not null && _prefix.IsPrefixOf(tokens))
        {
            _prefix.RestoreInto(_model);
            return _prefix.TokenCount;
        }
        _model.ResetKvCache();
        return 0;
    }

    private int[] ResolveVerdictTokens(string[] forms)
    {
        var ids = new List<int>();
        for (int id = 0; id < _model.Tokenizer.VocabSize; id++)
        {
            string piece = Normalize(_model.Tokenizer.Decode(id));
            if (Array.IndexOf(forms, piece) >= 0) ids.Add(id);
        }
        return [.. ids];
    }

    private static string Normalize(string piece)
        => piece.Trim().Trim('"', '\'', '.').ToLowerInvariant();

    /// <summary>
    /// Renormalises the best "yes" against the best "no".
    ///
    /// Rather than scanning the top-k as the model card's snippet does, the yes/no
    /// token ids are resolved once up front and read directly — same answer, but it
    /// cannot fail on a prompt where neither form makes the top 20, and it costs a
    /// couple of array lookups instead of a 131072-element partial sort.
    /// </summary>
    private ModerationResult Score(ReadOnlySpan<float> logits, int promptTokens, int prefilled)
    {
        float yes = Best(logits, _yesTokens);
        float no = Best(logits, _noTokens);

        float max = MathF.Max(yes, no);
        float eYes = MathF.Exp(yes - max);
        float eNo = MathF.Exp(no - max);
        return new ModerationResult(eYes / (eYes + eNo), yes, no, promptTokens, prefilled);

        static float Best(ReadOnlySpan<float> values, int[] ids)
        {
            float best = float.NegativeInfinity;
            foreach (int id in ids) if (values[id] > best) best = values[id];
            return best;
        }
    }

    /// <summary>
    /// The single most likely verdict token, decoded. Useful as a sanity check:
    /// anything other than "yes"/"no" means the prompt framing has drifted.
    /// </summary>
    public string TopVerdictToken(ModerationRequest request)
    {
        int[] tokens = Tokenize(request);
        int prefilled = PrepareCache(tokens);
        ReadOnlySpan<float> logits = _model.Forward(tokens.AsSpan(prefilled));
        return _model.Tokenizer.Decode(Kernels.ArgMax(logits));
    }

    public void Dispose()
    {
        if (_ownsModel) _model.Dispose();
    }
}
