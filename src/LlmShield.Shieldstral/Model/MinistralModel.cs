// Ministral-3 forward pass — the language backbone of Shieldstral 1.0 3B.
//
// Derived from TensorSharp's Mistral3Model (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause. See third-party/TensorSharp-LICENSE.
using System.Buffers;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Numerics;
using LlmShield.Shieldstral.Tokenization;

namespace LlmShield.Shieldstral.Model;

/// <summary>
/// A dense pre-norm transformer: RMSNorm → grouped-query attention with YaRN
/// RoPE → RMSNorm → SwiGLU MLP, 26 times, then a tied LM head.
///
/// Weights stay in their GGUF quantization inside the memory-mapped file and are
/// decoded one row at a time inside the matmul, so loading is near-instant and
/// the resident set is the file's size in shared page cache rather than that
/// plus a private float32 copy.
///
/// Not thread-safe: an instance owns one KV cache and one logits buffer. Use one
/// instance per concurrent request, or serialise calls.
/// </summary>
public sealed class MinistralModel : IDisposable
{
    private sealed class Layer
    {
        public required VectorParameter AttentionNorm { get; init; }
        public required WeightMatrix Q { get; init; }
        public required WeightMatrix K { get; init; }
        public required WeightMatrix V { get; init; }
        public required WeightMatrix O { get; init; }
        public required VectorParameter FfnNorm { get; init; }
        public required WeightMatrix Gate { get; init; }
        public required WeightMatrix Up { get; init; }
        public required WeightMatrix Down { get; init; }
    }

    private readonly GgufFile _gguf;
    private readonly bool _ownsFile;
    private readonly Layer[] _layers;
    private readonly WeightMatrix _tokenEmbeddings;
    private readonly WeightMatrix _lmHead;      // empty when embeddings are tied
    private readonly VectorParameter _outputNorm;
    private readonly Rope _rope;
    private readonly float[] _logits;

    public ModelConfig Config { get; }
    public TekkenTokenizer Tokenizer { get; }
    public KvCache KvCache { get; }
    public string ModelPath => _gguf.Path;

    /// <summary>Positions currently held in the KV cache.</summary>
    public int CachedTokenCount => KvCache.Length;

    public MinistralModel(string ggufPath, int initialCacheCapacity = 512)
        : this(new GgufFile(ggufPath), ownsFile: true, initialCacheCapacity) { }

    public MinistralModel(GgufFile gguf, bool ownsFile = false, int initialCacheCapacity = 512)
    {
        _gguf = gguf;
        _ownsFile = ownsFile;

        // A GGUF that is missing a tensor or disagrees with its own hyperparameters throws from
        // here, and then nobody holds the mapping this constructor opened. Windows still will not
        // let a mapped file be deleted, and deleting it is what the callers that catch these do —
        // ShieldstralModerator.CreateAsync drops a corrupt cached model and downloads it again.
        try
        {
            Config = ModelConfig.FromGguf(gguf);
            Tokenizer = TekkenTokenizer.FromGguf(gguf);
            _rope = new Rope(Config);

            _tokenEmbeddings = WeightMatrix.From(gguf, "token_embd.weight");
            _outputNorm = VectorParameter.From(gguf, "output_norm.weight");
            // Shieldstral ties the LM head to the embedding table, so `output.weight`
            // is legitimately absent; fall back to the embeddings in that case.
            _lmHead = WeightMatrix.From(gguf, "output.weight", required: false);

            _layers = new Layer[Config.LayerCount];
            for (int l = 0; l < Config.LayerCount; l++)
            {
                string p = $"blk.{l}.";
                _layers[l] = new Layer
                {
                    AttentionNorm = VectorParameter.From(gguf, p + "attn_norm.weight"),
                    Q = WeightMatrix.From(gguf, p + "attn_q.weight"),
                    K = WeightMatrix.From(gguf, p + "attn_k.weight"),
                    V = WeightMatrix.From(gguf, p + "attn_v.weight"),
                    O = WeightMatrix.From(gguf, p + "attn_output.weight"),
                    FfnNorm = VectorParameter.From(gguf, p + "ffn_norm.weight"),
                    Gate = WeightMatrix.From(gguf, p + "ffn_gate.weight"),
                    Up = WeightMatrix.From(gguf, p + "ffn_up.weight"),
                    Down = WeightMatrix.From(gguf, p + "ffn_down.weight"),
                };
            }

            ValidateShapes();

            KvCache = new KvCache(Config.LayerCount, Config.KvHeadCount, Config.HeadDim, initialCacheCapacity);
            _logits = new float[Config.VocabSize];
        }
        catch
        {
            if (ownsFile) gguf.Dispose();
            throw;
        }
    }

    private void ValidateShapes()
    {
        if (_tokenEmbeddings.Cols != Config.HiddenSize)
            throw new InvalidDataException(
                $"token_embd is {_tokenEmbeddings.Cols} wide but the model's hidden size is {Config.HiddenSize}.");
        Layer first = _layers[0];
        if (first.Q.Rows != Config.QDim)
            throw new InvalidDataException($"attn_q has {first.Q.Rows} rows; expected {Config.QDim}.");
        if (first.K.Rows != Config.KvDim)
            throw new InvalidDataException($"attn_k has {first.K.Rows} rows; expected {Config.KvDim}.");
        if (first.Gate.Rows != Config.FeedForwardSize)
            throw new InvalidDataException(
                $"ffn_gate has {first.Gate.Rows} rows; expected {Config.FeedForwardSize}.");
    }

    // ----------------------------------------------------------------- forward

    /// <summary>
    /// Appends <paramref name="tokens"/> to whatever is already cached and returns
    /// the logits for the final position. The returned memory is reused between
    /// calls — copy it if you need to keep it.
    /// <para>
    /// <paramref name="options"/> is handed straight to the row-parallel matmuls
    /// and the per-head attention loop, which between them are the only work this
    /// spreads across cores. Pass one with a bounded
    /// <see cref="ParallelOptions.MaxDegreeOfParallelism"/> to leave the rest of
    /// the machine alone.
    /// </para>
    /// </summary>
    public ValueTask<ReadOnlyMemory<float>> ForwardAsync(ReadOnlyMemory<int> tokens, ParallelOptions options)
        => ForwardAsync(tokens, capture: null, options);

    /// <summary>
    /// Runs <paramref name="tokens"/> through the model for their KV state only,
    /// skipping the 131072-row LM head. Used to warm a prefix whose logits nobody
    /// will look at — for Shieldstral that is the fixed system prompt, where the
    /// head would otherwise be a third of the work for a discarded result.
    /// </summary>
    public async ValueTask PrefillAsync(ReadOnlyMemory<int> tokens, ParallelOptions options)
        => await ForwardAsync(tokens, capture: null, options, computeLogits: false).ConfigureAwait(false);

    /// <summary>
    /// Forward pass with an optional observer for intermediate tensors, used by
    /// the parity tests to diff every layer against the Python reference.
    /// </summary>
    public ValueTask<ReadOnlyMemory<float>> ForwardAsync(
        ReadOnlyMemory<int> tokens, IActivationSink? capture, ParallelOptions options)
        => ForwardAsync(tokens, capture, options, computeLogits: true);

    private async ValueTask<ReadOnlyMemory<float>> ForwardAsync(
        ReadOnlyMemory<int> tokens, IActivationSink? capture, ParallelOptions options, bool computeLogits)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (tokens.Length == 0) throw new ArgumentException("No tokens to forward.", nameof(tokens));

        int seq = tokens.Length;
        int startPos = KvCache.Length;
        int hidden = Config.HiddenSize;
        int ff = Config.FeedForwardSize;

        KvCache.EnsureCapacity(startPos + seq);

        float[] states = ArrayPool<float>.Shared.Rent(seq * hidden);
        float[] normed = ArrayPool<float>.Shared.Rent(seq * hidden);
        float[] block = ArrayPool<float>.Shared.Rent(seq * hidden);
        float[] queries = ArrayPool<float>.Shared.Rent(seq * Config.QDim);
        float[] gate = ArrayPool<float>.Shared.Rent(seq * ff);
        float[] up = ArrayPool<float>.Shared.Rent(seq * ff);
        try
        {
            Memory<float> h = states.AsMemory(0, seq * hidden);
            Memory<float> norm = normed.AsMemory(0, seq * hidden);
            Memory<float> scratch = block.AsMemory(0, seq * hidden);
            Memory<float> g = gate.AsMemory(0, seq * ff);
            Memory<float> u = up.AsMemory(0, seq * ff);

            QuantMatMul.GatherRows(_tokenEmbeddings, tokens.Span, h.Span);
            capture?.Observe("embeddings", h.Span, seq, hidden);

            for (int l = 0; l < Config.LayerCount; l++)
            {
                Layer layer = _layers[l];

                NormRows(h.Span, layer.AttentionNorm, norm.Span, seq, hidden);
                capture?.Observe($"blk.{l}.attn_norm", norm.Span, seq, hidden);

                await AttentionAsync(layer, l, norm, queries, scratch, seq, startPos, capture, options).ConfigureAwait(false);
                capture?.Observe($"blk.{l}.attn_out", scratch.Span, seq, hidden);

                Kernels.Add(h.Span, scratch.Span);
                capture?.Observe($"blk.{l}.post_attn", h.Span, seq, hidden);

                NormRows(h.Span, layer.FfnNorm, norm.Span, seq, hidden);
                capture?.Observe($"blk.{l}.ffn_norm", norm.Span, seq, hidden);

                await QuantMatMul.ForwardAsync(layer.Gate, norm, seq, g, options).ConfigureAwait(false);
                await QuantMatMul.ForwardAsync(layer.Up, norm, seq, u, options).ConfigureAwait(false);
                Kernels.SwiGlu(g.Span, u.Span, g.Span);
                await QuantMatMul.ForwardAsync(layer.Down, g, seq, scratch, options).ConfigureAwait(false);
                capture?.Observe($"blk.{l}.ffn_out", scratch.Span, seq, hidden);

                Kernels.Add(h.Span, scratch.Span);
                capture?.Observe($"blk.{l}.output", h.Span, seq, hidden);
            }

            KvCache.Advance(seq);
            if (!computeLogits) return default;

            // Only the last position feeds the LM head: Shieldstral's verdict is a
            // single token, and the head is a 131072-row matmul we would rather not
            // run once per prompt token.
            Memory<float> last = norm[..hidden];
            Kernels.RmsNorm(h.Span.Slice((seq - 1) * hidden, hidden), _outputNorm, Config.RmsNormEps, last.Span);
            capture?.Observe("final_norm", last.Span, 1, hidden);

            WeightMatrix head = _lmHead.IsEmpty ? _tokenEmbeddings : _lmHead;
            await QuantMatMul.ForwardAsync(head, last, _logits, options).ConfigureAwait(false);
            capture?.Observe("logits", _logits, 1, Config.VocabSize);

            return _logits;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(states);
            ArrayPool<float>.Shared.Return(normed);
            ArrayPool<float>.Shared.Return(block);
            ArrayPool<float>.Shared.Return(queries);
            ArrayPool<float>.Shared.Return(gate);
            ArrayPool<float>.Shared.Return(up);
        }
    }

    private void NormRows(ReadOnlySpan<float> source, VectorParameter weight, Span<float> destination,
        int rows, int width)
    {
        for (int t = 0; t < rows; t++)
            Kernels.RmsNorm(source.Slice(t * width, width), weight, Config.RmsNormEps,
                destination.Slice(t * width, width));
    }

    /// <param name="output">Receives the attention block's contribution, seq × hidden.</param>
    private async ValueTask AttentionAsync(
        Layer layer, int layerIndex, ReadOnlyMemory<float> input,
        float[] queryBuffer, Memory<float> output, int seq, int startPos,
        IActivationSink? capture, ParallelOptions options)
    {
        int heads = Config.HeadCount, kvHeads = Config.KvHeadCount, headDim = Config.HeadDim;
        int group = Config.GroupSize, kvDim = Config.KvDim, qDim = Config.QDim;
        float scale = 1f / MathF.Sqrt(headDim);
        int cacheLength = startPos + seq;

        Memory<float> q = queryBuffer.AsMemory(0, seq * qDim);
        await QuantMatMul.ForwardAsync(layer.Q, input, seq, q, options).ConfigureAwait(false);
        RotateQueries(q.Span, heads, headDim, qDim, seq, startPos);
        capture?.Observe($"blk.{layerIndex}.q_rope", q.Span, seq, qDim);

        // K and V land straight in their cache slots, so the cache always holds
        // post-RoPE keys exactly as the score loop will read them back.
        float[] staging = ArrayPool<float>.Shared.Rent(seq * kvDim);
        try
        {
            Memory<float> kv = staging.AsMemory(0, seq * kvDim);

            await QuantMatMul.ForwardAsync(layer.K, input, seq, kv, options).ConfigureAwait(false);
            StoreKeys(kv.Span, layerIndex, kvHeads, headDim, kvDim, seq, startPos);
            capture?.Observe($"blk.{layerIndex}.k_rope", kv.Span, seq, kvDim);

            await QuantMatMul.ForwardAsync(layer.V, input, seq, kv, options).ConfigureAwait(false);
            StoreValues(kv.Span, layerIndex, kvHeads, headDim, kvDim, seq, startPos);
            capture?.Observe($"blk.{layerIndex}.v", kv.Span, seq, kvDim);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(staging);
        }

        // The attention context is heads x head_dim wide (4096 for Shieldstral),
        // which is *not* the model's hidden size (3072) — attn_output projects the
        // one down to the other. They coincide in most Llama-family models, so it
        // is an easy assumption to make and a confusing one to debug.
        float[] contextBuffer = ArrayPool<float>.Shared.Rent(seq * qDim);
        try
        {
            Memory<float> context = contextBuffer.AsMemory(0, seq * qDim);
            KvCache cache = KvCache;

            // Grouped-query attention. Heads are independent, so each rents its own
            // scores buffer for the duration and no synchronisation is needed. The
            // pool is what makes that free: a per-head cache on the model would have
            // to be sized for the widest fan-out any caller ever asks for, and would
            // pin every one of those buffers for the model's lifetime.
            await Parallel.ForAsync(0, heads, options, (head, _) =>
            {
                float[] scores = ArrayPool<float>.Shared.Rent(cacheLength);
                try
                {
                    int kvHead = head / group;
                    ReadOnlySpan<float> keys = cache.KeyHistory(layerIndex, kvHead, cacheLength);
                    ReadOnlySpan<float> values = cache.ValueHistory(layerIndex, kvHead, cacheLength);

                    for (int t = 0; t < seq; t++)
                    {
                        int limit = startPos + t + 1;                      // causal mask
                        ReadOnlySpan<float> query = q.Span.Slice(t * qDim + head * headDim, headDim);
                        Span<float> row = scores.AsSpan(0, limit);
                        for (int p = 0; p < limit; p++)
                            row[p] = Kernels.Dot(query, keys.Slice(p * headDim, headDim)) * scale;
                        Kernels.Softmax(row);

                        Span<float> sink = context.Span.Slice(t * qDim + head * headDim, headDim);
                        sink.Clear();
                        for (int p = 0; p < limit; p++)
                        {
                            float w = row[p];
                            if (w != 0f) Kernels.AddScaled(sink, values.Slice(p * headDim, headDim), w);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scores);
                }

                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);

            capture?.Observe($"blk.{layerIndex}.attn_weighted", context.Span, seq, qDim);

            await QuantMatMul.ForwardAsync(layer.O, context, seq, output, options).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(contextBuffer);
        }
    }

    private void RotateQueries(Span<float> q, int heads, int headDim, int qDim, int seq, int startPos)
    {
        for (int t = 0; t < seq; t++)
        {
            Span<float> row = q.Slice(t * qDim, qDim);
            _rope.Apply(row, heads, headDim, startPos + t);
            ApplyPositionScale(row, startPos + t);
        }
    }

    private void StoreKeys(Span<float> kv, int layerIndex, int kvHeads, int headDim, int kvDim, int seq, int startPos)
    {
        for (int t = 0; t < seq; t++)
        {
            Span<float> row = kv.Slice(t * kvDim, kvDim);
            _rope.Apply(row, kvHeads, headDim, startPos + t);
            for (int kh = 0; kh < kvHeads; kh++)
                row.Slice(kh * headDim, headDim).CopyTo(KvCache.Key(layerIndex, kh, startPos + t));
        }
    }

    private void StoreValues(Span<float> kv, int layerIndex, int kvHeads, int headDim, int kvDim, int seq, int startPos)
    {
        for (int t = 0; t < seq; t++)
            for (int kh = 0; kh < kvHeads; kh++)
                kv.Slice(t * kvDim + kh * headDim, headDim)
                    .CopyTo(KvCache.Value(layerIndex, kh, startPos + t));
    }

    /// <summary>
    /// Llama-4 attention temperature: <c>q *= 1 + beta·ln(1 + floor(pos / n_orig))</c>.
    /// Inside the 16384-token training window the floor is 0 and this is exactly 1,
    /// so it costs nothing on the prompts Shieldstral actually sees.
    /// </summary>
    private void ApplyPositionScale(Span<float> query, int position)
    {
        if (Config.RopeOriginalContextLength <= 0 || Config.AttentionTemperatureScale == 0f) return;
        float interval = MathF.Floor((float)position / Config.RopeOriginalContextLength);
        if (interval == 0f) return;
        Kernels.Scale(query, 1f + Config.AttentionTemperatureScale * MathF.Log(1f + interval));
    }

    // ------------------------------------------------------------------- state

    public void ResetKvCache() => KvCache.Reset();

    public void Dispose()
    {
        if (_ownsFile) _gguf.Dispose();
    }
}

/// <summary>
/// Receives intermediate tensors during a forward pass. Implemented by the tests
/// to diff every layer against the Python reference; production code passes null.
/// </summary>
public interface IActivationSink
{
    void Observe(string name, ReadOnlySpan<float> values, int rows, int columns);
}
