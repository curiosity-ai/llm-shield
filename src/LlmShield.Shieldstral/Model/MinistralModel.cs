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
public sealed unsafe class MinistralModel : IDisposable
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
    private readonly float[][] _scoreScratch;   // one per attention head

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
            _scoreScratch = new float[Config.HeadCount][];
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
    /// the logits for the final position. The returned span is reused between
    /// calls — copy it if you need to keep it.
    /// </summary>
    public ReadOnlySpan<float> Forward(ReadOnlySpan<int> tokens) => Forward(tokens, capture: null);

    /// <summary>
    /// Runs <paramref name="tokens"/> through the model for their KV state only,
    /// skipping the 131072-row LM head. Used to warm a prefix whose logits nobody
    /// will look at — for Shieldstral that is the fixed system prompt, where the
    /// head would otherwise be a third of the work for a discarded result.
    /// </summary>
    public void Prefill(ReadOnlySpan<int> tokens) => Forward(tokens, capture: null, computeLogits: false);

    /// <summary>
    /// Forward pass with an optional observer for intermediate tensors, used by
    /// the parity tests to diff every layer against the Python reference.
    /// </summary>
    public ReadOnlySpan<float> Forward(ReadOnlySpan<int> tokens, IActivationSink? capture)
        => Forward(tokens, capture, computeLogits: true);

    private ReadOnlySpan<float> Forward(ReadOnlySpan<int> tokens, IActivationSink? capture, bool computeLogits)
    {
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
            Span<float> h = states.AsSpan(0, seq * hidden);
            Span<float> norm = normed.AsSpan(0, seq * hidden);
            Span<float> scratch = block.AsSpan(0, seq * hidden);

            QuantMatMul.GatherRows(_tokenEmbeddings, tokens, h);
            capture?.Observe("embeddings", h, seq, hidden);

            for (int l = 0; l < Config.LayerCount; l++)
            {
                Layer layer = _layers[l];

                NormRows(h, layer.AttentionNorm, norm, seq, hidden);
                capture?.Observe($"blk.{l}.attn_norm", norm, seq, hidden);

                Attention(layer, l, norm, queries, scratch, seq, startPos, capture);
                capture?.Observe($"blk.{l}.attn_out", scratch, seq, hidden);

                Kernels.Add(h, scratch);
                capture?.Observe($"blk.{l}.post_attn", h, seq, hidden);

                NormRows(h, layer.FfnNorm, norm, seq, hidden);
                capture?.Observe($"blk.{l}.ffn_norm", norm, seq, hidden);

                Span<float> g = gate.AsSpan(0, seq * ff);
                Span<float> u = up.AsSpan(0, seq * ff);
                QuantMatMul.Forward(layer.Gate, norm, seq, g);
                QuantMatMul.Forward(layer.Up, norm, seq, u);
                Kernels.SwiGlu(g, u, g);
                QuantMatMul.Forward(layer.Down, g, seq, scratch);
                capture?.Observe($"blk.{l}.ffn_out", scratch, seq, hidden);

                Kernels.Add(h, scratch);
                capture?.Observe($"blk.{l}.output", h, seq, hidden);
            }

            KvCache.Advance(seq);
            if (!computeLogits) return default;

            // Only the last position feeds the LM head: Shieldstral's verdict is a
            // single token, and the head is a 131072-row matmul we would rather not
            // run once per prompt token.
            Span<float> last = norm[..hidden];
            Kernels.RmsNorm(h.Slice((seq - 1) * hidden, hidden), _outputNorm, Config.RmsNormEps, last);
            capture?.Observe("final_norm", last, 1, hidden);

            WeightMatrix head = _lmHead.IsEmpty ? _tokenEmbeddings : _lmHead;
            QuantMatMul.Forward(head, last, _logits);
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
    private void Attention(
        Layer layer, int layerIndex, ReadOnlySpan<float> input,
        float[] queryBuffer, Span<float> output, int seq, int startPos, IActivationSink? capture)
    {
        int heads = Config.HeadCount, kvHeads = Config.KvHeadCount, headDim = Config.HeadDim;
        int group = Config.GroupSize, kvDim = Config.KvDim, qDim = Config.QDim, hidden = Config.HiddenSize;
        float scale = 1f / MathF.Sqrt(headDim);
        int cacheLength = startPos + seq;

        Span<float> q = queryBuffer.AsSpan(0, seq * qDim);
        QuantMatMul.Forward(layer.Q, input, seq, q);
        for (int t = 0; t < seq; t++)
        {
            Span<float> row = q.Slice(t * qDim, qDim);
            _rope.Apply(row, heads, headDim, startPos + t);
            ApplyPositionScale(row, startPos + t);
        }
        capture?.Observe($"blk.{layerIndex}.q_rope", q, seq, qDim);

        // K and V land straight in their cache slots, so the cache always holds
        // post-RoPE keys exactly as the score loop will read them back.
        float[] staging = ArrayPool<float>.Shared.Rent(seq * kvDim);
        try
        {
            Span<float> kv = staging.AsSpan(0, seq * kvDim);

            QuantMatMul.Forward(layer.K, input, seq, kv);
            for (int t = 0; t < seq; t++)
            {
                Span<float> row = kv.Slice(t * kvDim, kvDim);
                _rope.Apply(row, kvHeads, headDim, startPos + t);
                for (int kh = 0; kh < kvHeads; kh++)
                    row.Slice(kh * headDim, headDim).CopyTo(KvCache.Key(layerIndex, kh, startPos + t));
            }
            capture?.Observe($"blk.{layerIndex}.k_rope", kv, seq, kvDim);

            QuantMatMul.Forward(layer.V, input, seq, kv);
            for (int t = 0; t < seq; t++)
                for (int kh = 0; kh < kvHeads; kh++)
                    kv.Slice(t * kvDim + kh * headDim, headDim)
                        .CopyTo(KvCache.Value(layerIndex, kh, startPos + t));
            capture?.Observe($"blk.{layerIndex}.v", kv, seq, kvDim);
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
            Span<float> context = contextBuffer.AsSpan(0, seq * qDim);

            // Grouped-query attention. Heads are independent, so each gets its own
            // scores buffer and no synchronisation is needed.
            fixed (float* qp = q)
            fixed (float* cp = context)
            {
                float* queryBase = qp, contextBase = cp;
                KvCache cache = KvCache;
                float[][] scratch = _scoreScratch;

                Parallel.For(0, heads, head =>
                {
                    float[] scores = scratch[head];
                    if (scores is null || scores.Length < cacheLength)
                        scores = scratch[head] = new float[Math.Max(cacheLength, 256)];

                    int kvHead = head / group;
                    ReadOnlySpan<float> keys = cache.KeyHistory(layerIndex, kvHead, cacheLength);
                    ReadOnlySpan<float> values = cache.ValueHistory(layerIndex, kvHead, cacheLength);

                    for (int t = 0; t < seq; t++)
                    {
                        int limit = startPos + t + 1;                      // causal mask
                        var query = new ReadOnlySpan<float>(queryBase + t * qDim + head * headDim, headDim);
                        Span<float> row = scores.AsSpan(0, limit);
                        for (int p = 0; p < limit; p++)
                            row[p] = Kernels.Dot(query, keys.Slice(p * headDim, headDim)) * scale;
                        Kernels.Softmax(row);

                        var sink = new Span<float>(contextBase + t * qDim + head * headDim, headDim);
                        sink.Clear();
                        for (int p = 0; p < limit; p++)
                        {
                            float w = row[p];
                            if (w != 0f) Kernels.AddScaled(sink, values.Slice(p * headDim, headDim), w);
                        }
                    }
                });
            }
            capture?.Observe($"blk.{layerIndex}.attn_weighted", context, seq, qDim);

            QuantMatMul.Forward(layer.O, context, seq, output);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(contextBuffer);
        }
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
