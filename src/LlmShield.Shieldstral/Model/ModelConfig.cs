using LlmShield.Shieldstral.Gguf;

namespace LlmShield.Shieldstral.Model;

/// <summary>
/// Hyperparameters of the Ministral-3 backbone, read from GGUF metadata.
///
/// Shieldstral publishes these in <c>params.json</c>; the converter copies them
/// into the GGUF under llama.cpp's key names so a checkpoint produced by either
/// converter loads identically.
/// </summary>
public sealed record ModelConfig
{
    public required string Architecture { get; init; }
    public required int LayerCount { get; init; }
    public required int HiddenSize { get; init; }
    public required int HeadCount { get; init; }
    public required int KvHeadCount { get; init; }
    public required int HeadDim { get; init; }
    public required int FeedForwardSize { get; init; }
    public required int VocabSize { get; init; }
    public required float RmsNormEps { get; init; }
    public required int ContextLength { get; init; }

    // ---- RoPE / YaRN ----
    public required float RopeFreqBase { get; init; }
    public required int RopeDim { get; init; }
    /// <summary>YaRN factor; 1 means no context extension.</summary>
    public required float RopeScaleFactor { get; init; }
    public required string RopeScalingType { get; init; }
    /// <summary>Training window YaRN interpolates from (16384 for Shieldstral).</summary>
    public required int RopeOriginalContextLength { get; init; }
    public required float YarnBetaFast { get; init; }
    public required float YarnBetaSlow { get; init; }
    public required float YarnExtFactor { get; init; }
    public required float YarnMscale { get; init; }
    /// <summary>
    /// YaRN's <c>mscale_all_dim</c>. llama.cpp's writer stores this under
    /// <c>rope.scaling.yarn_log_multiplier</c>; Mistral-format checkpoints set it
    /// to 1.0 when <c>params.json</c> says <c>"apply_scale": false</c>, which makes
    /// the magnitude correction cancel to exactly 1.
    /// </summary>
    public required float YarnMscaleAllDim { get; init; }
    public required float RopeAttnFactor { get; init; }

    /// <summary>
    /// Llama-4-style attention temperature: q is scaled by
    /// <c>1 + beta·ln(1 + floor(pos / original_context))</c>, which is a no-op
    /// inside the training window and grows slowly beyond it.
    /// </summary>
    public required float AttentionTemperatureScale { get; init; }

    /// <summary>True when the frequency ramp and magnitude correction are in effect.</summary>
    public bool YarnActive =>
        RopeScalingType == "yarn" && RopeOriginalContextLength > 0
        && YarnExtFactor != 0f && RopeScaleFactor > 1f;

    public int GroupSize => HeadCount / KvHeadCount;
    public int QDim => HeadCount * HeadDim;
    public int KvDim => KvHeadCount * HeadDim;

    public static ModelConfig FromGguf(GgufFile gguf)
    {
        string arch = gguf.GetString("general.architecture")
            ?? throw new InvalidDataException($"{gguf.Path} has no general.architecture.");
        if (arch is not ("mistral3" or "ministral3" or "llama"))
            throw new NotSupportedException(
                $"{gguf.Path} is architecture '{arch}'; this runtime implements the Ministral-3 " +
                "backbone that Shieldstral uses (GGUF architecture \"mistral3\").");

        int headCount = (int)gguf.GetUInt32($"{arch}.attention.head_count");
        int hidden = (int)gguf.GetUInt32($"{arch}.embedding_length");
        int headDim = (int)gguf.GetUInt32($"{arch}.attention.key_length", (uint)(hidden / Math.Max(1, headCount)));

        var config = new ModelConfig
        {
            Architecture = arch,
            LayerCount = (int)gguf.GetUInt32($"{arch}.block_count"),
            HiddenSize = hidden,
            HeadCount = headCount,
            KvHeadCount = (int)gguf.GetUInt32($"{arch}.attention.head_count_kv", (uint)headCount),
            HeadDim = headDim,
            FeedForwardSize = (int)gguf.GetUInt32($"{arch}.feed_forward_length"),
            VocabSize = gguf.GetStringArray("tokenizer.ggml.tokens")?.Length
                        ?? (int)gguf.GetUInt32($"{arch}.vocab_size"),
            RmsNormEps = gguf.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            ContextLength = (int)gguf.GetUInt32($"{arch}.context_length", 32768),

            RopeFreqBase = gguf.GetFloat32($"{arch}.rope.freq_base", 10000f),
            RopeDim = (int)gguf.GetUInt32($"{arch}.rope.dimension_count", (uint)headDim),
            RopeScaleFactor = gguf.GetFloat32($"{arch}.rope.scaling.factor", 1f),
            RopeScalingType = gguf.GetString($"{arch}.rope.scaling.type", "") ?? "",
            RopeOriginalContextLength = (int)gguf.GetUInt32($"{arch}.rope.scaling.original_context_length", 0),
            YarnBetaFast = gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_fast",
                           gguf.GetFloat32($"{arch}.rope.scaling.beta_fast", 32f)),
            YarnBetaSlow = gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_slow",
                           gguf.GetFloat32($"{arch}.rope.scaling.beta_slow", 1f)),
            YarnExtFactor = gguf.GetFloat32($"{arch}.rope.scaling.yarn_ext_factor", 1f),
            YarnMscale = gguf.GetFloat32($"{arch}.rope.scaling.mscale", 0f),
            YarnMscaleAllDim = gguf.GetFloat32($"{arch}.rope.scaling.yarn_log_multiplier",
                               gguf.GetFloat32($"{arch}.rope.scaling.mscale_all_dim", 0f)),
            RopeAttnFactor = gguf.GetFloat32($"{arch}.rope.scaling.attn_factor",
                             gguf.GetFloat32($"{arch}.rope.scaling.yarn_attn_factor", 1f)),
            AttentionTemperatureScale = gguf.GetFloat32($"{arch}.attention.temperature_scale",
                                        gguf.GetFloat32($"{arch}.rope.scaling_beta", 0f)),
        };

        if (config.HeadCount % config.KvHeadCount != 0)
            throw new InvalidDataException(
                $"{config.HeadCount} attention heads do not divide into {config.KvHeadCount} KV heads.");
        return config;
    }

    public override string ToString() =>
        $"{Architecture}: {LayerCount}L d={HiddenSize} heads={HeadCount}/{KvHeadCount}x{HeadDim} " +
        $"ffn={FeedForwardSize} vocab={VocabSize} rope={RopeFreqBase}" +
        (YarnActive ? $" yarn(x{RopeScaleFactor} over {RopeOriginalContextLength})" : "");
}
