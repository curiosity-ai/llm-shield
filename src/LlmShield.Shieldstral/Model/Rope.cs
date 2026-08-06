// YaRN-scaled rotary embeddings, matching ggml's rope_yarn.
//
// Derived from TensorSharp's Mistral3Model (https://github.com/zhongkaifu/TensorSharp),
// BSD-3-Clause. See third-party/TensorSharp-LICENSE.
namespace LlmShield.Shieldstral.Model;

/// <summary>
/// Precomputed rotary frequencies plus the single magnitude factor YaRN applies
/// on top of them.
///
/// Two properties matter for correctness and are easy to get subtly wrong:
///
/// 1. Ministral rotates <em>adjacent</em> pairs (x[2i], x[2i+1]) — ggml's "norm"
///    mode, the layout the Mistral-format checkpoint is trained in. HF's Llama
///    code rotates (x[i], x[i+d/2]) instead and its converter permutes the
///    weights to compensate; converting from Mistral format must not.
/// 2. Prefill and decode must use the <em>same</em> frequencies and the same
///    magnitude factor. If they disagree, a freshly generated token is rotated
///    differently from the prompt already in the KV cache and attention silently
///    degrades. Keeping one table for both is what makes that impossible here.
/// </summary>
public sealed class Rope
{
    private readonly float[] _freqs;   // one per rotated pair
    private readonly float _mscale;
    private readonly int _ropeDim;

    /// <summary>Magnitude correction folded into cos/sin (1.0 when YaRN says "apply_scale": false).</summary>
    public float MagnitudeScale => _mscale;

    public ReadOnlySpan<float> Frequencies => _freqs;

    public Rope(ModelConfig config)
    {
        _ropeDim = config.RopeDim;
        int half = _ropeDim / 2;
        _freqs = new float[half];

        float freqScale = 1f / config.RopeScaleFactor;
        bool yarn = config.YarnActive;

        if (!yarn)
        {
            for (int i = 0; i < half; i++)
                _freqs[i] = freqScale / MathF.Pow(config.RopeFreqBase, 2f * i / _ropeDim);
            _mscale = config.RopeAttnFactor;
            return;
        }

        YarnCorrectionDims(_ropeDim, config.RopeOriginalContextLength, config.RopeFreqBase,
            config.YarnBetaFast, config.YarnBetaSlow, out float low, out float high);

        for (int i = 0; i < half; i++)
        {
            float extrapolated = 1f / MathF.Pow(config.RopeFreqBase, 2f * i / _ropeDim);
            float interpolated = freqScale * extrapolated;

            // Ramp is linear in the dimension-pair index, exactly as ggml does it.
            float y = (i - low) / MathF.Max(0.001f, high - low);
            float mix = (1f - Math.Clamp(y, 0f, 1f)) * config.YarnExtFactor;

            _freqs[i] = interpolated * (1f - mix) + extrapolated * mix;
        }

        _mscale = ComputeMagnitudeScale(config);
    }

    /// <summary>
    /// Net magnitude scaling YaRN applies to q and k.
    ///
    /// llama.cpp splits this into an "attention factor" and a fixed
    /// <c>1 + 0.1·ln(factor)</c> term inside its RoPE kernel; the two multiply
    /// back to the ratio below. For a Mistral-format checkpoint with
    /// <c>"apply_scale": false</c> both mscale terms are 1.0, they cancel, and the
    /// result is exactly 1 — no magnitude boost, which is what the reference
    /// implementation does.
    /// </summary>
    private static float ComputeMagnitudeScale(ModelConfig config)
    {
        float factor = config.RopeScaleFactor;
        float mscale = config.YarnMscale != 0f ? config.YarnMscale : 1f;   // llama.cpp's default
        float ratio = config.YarnMscaleAllDim != 0f
            ? Mscale(factor, mscale) / Mscale(factor, config.YarnMscaleAllDim)
            : Mscale(factor, 1f);
        return ratio * config.RopeAttnFactor;

        static float Mscale(float scale, float m) => scale <= 1f ? 1f : 0.1f * m * MathF.Log(scale) + 1f;
    }

    /// <summary>Mirror of ggml's <c>ggml_rope_yarn_corr_dims</c>.</summary>
    private static void YarnCorrectionDims(
        int dims, int originalContext, float freqBase, float betaFast, float betaSlow,
        out float low, out float high)
    {
        if (betaFast == 0f && betaSlow == 0f)
        {
            low = float.MaxValue;
            high = dims / 2f - 1f;
            return;
        }
        low = MathF.Max(0, MathF.Floor(CorrDim(betaFast)));
        high = MathF.Min(dims / 2f - 1f, MathF.Ceiling(CorrDim(betaSlow)));

        float CorrDim(float rotations)
            => dims * MathF.Log(originalContext / (rotations * 2 * MathF.PI)) / (2 * MathF.Log(freqBase));
    }

    /// <summary>
    /// Rotates every head of one token in place. <paramref name="data"/> is
    /// <paramref name="heads"/> × <paramref name="headDim"/> contiguous floats.
    /// </summary>
    public void Apply(Span<float> data, int heads, int headDim, int position)
    {
        int half = _freqs.Length;
        Span<float> cos = stackalloc float[half];
        Span<float> sin = stackalloc float[half];
        for (int i = 0; i < half; i++)
        {
            float theta = position * _freqs[i];
            cos[i] = MathF.Cos(theta) * _mscale;
            sin[i] = MathF.Sin(theta) * _mscale;
        }

        for (int h = 0; h < heads; h++)
        {
            Span<float> head = data.Slice(h * headDim, headDim);
            for (int i = 0; i < half; i++)
            {
                float x0 = head[2 * i];
                float x1 = head[2 * i + 1];
                head[2 * i] = x0 * cos[i] - x1 * sin[i];
                head[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
            }
        }
    }
}
