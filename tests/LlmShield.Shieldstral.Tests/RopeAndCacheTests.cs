using LlmShield.Shieldstral.Model;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// The YaRN rotary embedding and the KV cache, checked without needing weights.
/// </summary>
public class RopeAndCacheTests
{
    private readonly ITestOutputHelper _output;
    public RopeAndCacheTests(ITestOutputHelper output) => _output = output;

    private static ModelConfig ShieldstralLikeConfig(bool yarn = true) => new()
    {
        Architecture = "mistral3",
        LayerCount = 2,
        HiddenSize = 3072,
        HeadCount = 32,
        KvHeadCount = 8,
        HeadDim = 128,
        FeedForwardSize = 9216,
        VocabSize = 131072,
        RmsNormEps = 1e-5f,
        ContextLength = 262144,
        RopeFreqBase = 1_000_000f,
        RopeDim = 128,
        RopeScaleFactor = yarn ? 16f : 1f,
        RopeScalingType = yarn ? "yarn" : "",
        RopeOriginalContextLength = 16384,
        YarnBetaFast = 32f,
        YarnBetaSlow = 1f,
        YarnExtFactor = 1f,
        YarnMscale = 1f,
        YarnMscaleAllDim = 1f,
        RopeAttnFactor = 1f,
        AttentionTemperatureScale = 0.1f,
    };

    /// <summary>
    /// Shieldstral's params.json sets <c>"apply_scale": false</c>, which in
    /// llama.cpp's encoding means mscale == mscale_all_dim and the magnitude
    /// correction cancels to exactly 1. Getting this wrong scales every q and k by
    /// ~1.277 and skews attention everywhere at once — the kind of bug that still
    /// produces fluent-looking output.
    /// </summary>
    [Fact]
    public void MagnitudeScaleIsOneWhenApplyScaleIsOff()
    {
        var rope = new Rope(ShieldstralLikeConfig());
        Assert.Equal(1f, rope.MagnitudeScale, 6);
    }

    [Fact]
    public void MagnitudeScaleFallsBackWhenNoLogMultiplierIsStored()
    {
        // An older GGUF that never wrote mscale_all_dim: llama.cpp then applies the
        // plain 1 + 0.1*ln(factor).
        ModelConfig withoutLogMul = ShieldstralLikeConfig() with { YarnMscale = 0f, YarnMscaleAllDim = 0f };
        var rope = new Rope(withoutLogMul);
        Assert.Equal(1f + 0.1f * MathF.Log(16f), rope.MagnitudeScale, 5);
    }

    /// <summary>
    /// YaRN blends interpolated and extrapolated frequencies over a ramp: the
    /// fastest bands stay extrapolated (unscaled), the slowest are fully
    /// interpolated (divided by the factor).
    /// </summary>
    [Fact]
    public void YarnRampInterpolatesTheSlowBandsOnly()
    {
        var yarn = new Rope(ShieldstralLikeConfig(yarn: true));
        var plain = new Rope(ShieldstralLikeConfig(yarn: false));

        ReadOnlySpan<float> scaled = yarn.Frequencies;
        ReadOnlySpan<float> unscaled = plain.Frequencies;
        Assert.Equal(64, scaled.Length);

        // Highest-frequency band: pure extrapolation, so unchanged.
        Assert.Equal(unscaled[0], scaled[0], 6);

        // Lowest-frequency band: fully interpolated, so divided by the factor.
        Assert.Equal(unscaled[^1] / 16f, scaled[^1], 10);

        // And the ramp is monotone in between.
        for (int i = 1; i < scaled.Length; i++)
            Assert.True(scaled[i] <= scaled[i - 1],
                $"frequency band {i} ({scaled[i]:E3}) exceeds band {i - 1} ({scaled[i - 1]:E3})");

        _output.WriteLine($"band 0 {scaled[0]:E4} (unscaled {unscaled[0]:E4}), " +
                          $"band 63 {scaled[^1]:E4} (unscaled {unscaled[^1]:E4})");
    }

    /// <summary>
    /// Position 0 is the identity rotation, and the rotation preserves the norm of
    /// each rotated pair — which is what distinguishes a rotation from a rescale.
    /// </summary>
    [Fact]
    public void RotationIsNormPreservingAndIdentityAtZero()
    {
        var rope = new Rope(ShieldstralLikeConfig());
        var rng = new Random(5);
        var original = new float[128];
        for (int i = 0; i < original.Length; i++) original[i] = (float)(rng.NextDouble() * 2 - 1);

        var atZero = (float[])original.Clone();
        rope.Apply(atZero, heads: 1, headDim: 128, position: 0);
        Numeric.Close(original, atZero, 1e-6, "rotation at position 0");

        var rotated = (float[])original.Clone();
        rope.Apply(rotated, heads: 1, headDim: 128, position: 137);
        for (int i = 0; i < 64; i++)
        {
            double before = Math.Sqrt(original[2 * i] * (double)original[2 * i]
                                    + original[2 * i + 1] * (double)original[2 * i + 1]);
            double after = Math.Sqrt(rotated[2 * i] * (double)rotated[2 * i]
                                   + rotated[2 * i + 1] * (double)rotated[2 * i + 1]);
            Numeric.Close(before, after, 1e-5, $"pair {i} magnitude");
        }
    }

    /// <summary>
    /// Rotation must pair adjacent elements (x[2i], x[2i+1]), not (x[i], x[i+d/2]).
    /// The Mistral-format checkpoint is stored for the former; using the latter
    /// loads and runs and is completely wrong.
    /// </summary>
    [Fact]
    public void RotationPairsAdjacentElements()
    {
        var rope = new Rope(ShieldstralLikeConfig());
        var x = new float[128];
        x[0] = 1f;                       // first element of the first pair

        rope.Apply(x, heads: 1, headDim: 128, position: 1);

        // Only its partner may have picked up energy.
        Assert.NotEqual(0f, x[1]);
        for (int i = 2; i < x.Length; i++)
            Assert.Equal(0f, x[i]);
    }

    [Fact]
    public void HeadsRotateIndependently()
    {
        var rope = new Rope(ShieldstralLikeConfig());
        var rng = new Random(9);
        var two = new float[256];
        for (int i = 0; i < 128; i++) two[i] = two[128 + i] = (float)(rng.NextDouble() * 2 - 1);

        rope.Apply(two, heads: 2, headDim: 128, position: 42);
        for (int i = 0; i < 128; i++)
            Assert.Equal(two[i], two[128 + i], 6);
    }

    // ---------------------------------------------------------------- KV cache

    [Fact]
    public void CacheGrowsWithoutLosingHistory()
    {
        var cache = new KvCache(layers: 2, kvHeads: 2, headDim: 4, initialCapacity: 2);
        Assert.Equal(2, cache.Capacity);

        for (int position = 0; position < 5; position++)
        {
            cache.EnsureCapacity(position + 1);
            for (int layer = 0; layer < 2; layer++)
                for (int head = 0; head < 2; head++)
                {
                    Fill(cache.Key(layer, head, position), layer, head, position, 0);
                    Fill(cache.Value(layer, head, position), layer, head, position, 100);
                }
            cache.Advance(1);
        }

        Assert.True(cache.Capacity >= 5);
        Assert.Equal(5, cache.Length);
        for (int position = 0; position < 5; position++)
            for (int layer = 0; layer < 2; layer++)
                for (int head = 0; head < 2; head++)
                {
                    AssertFilled(cache.Key(layer, head, position), layer, head, position, 0);
                    AssertFilled(cache.Value(layer, head, position), layer, head, position, 100);
                }

        static void Fill(Span<float> span, int layer, int head, int position, int bias)
        {
            for (int d = 0; d < span.Length; d++) span[d] = bias + layer * 1000 + head * 100 + position * 10 + d;
        }

        static void AssertFilled(Span<float> span, int layer, int head, int position, int bias)
        {
            for (int d = 0; d < span.Length; d++)
                Assert.Equal(bias + layer * 1000 + head * 100 + position * 10 + d, span[d]);
        }
    }

    [Fact]
    public void SnapshotAndRestoreRoundTrip()
    {
        var cache = new KvCache(layers: 3, kvHeads: 2, headDim: 4, initialCapacity: 8);
        var rng = new Random(17);
        for (int position = 0; position < 6; position++)
        {
            for (int layer = 0; layer < 3; layer++)
                for (int head = 0; head < 2; head++)
                {
                    Randomize(cache.Key(layer, head, position));
                    Randomize(cache.Value(layer, head, position));
                }
            cache.Advance(1);
        }

        float[] snapshot = cache.Snapshot(4);
        float[] expectedKeys = cache.KeyHistory(1, 1, 4).ToArray();

        // Overwrite everything, then restore and check the first four positions came back.
        for (int position = 0; position < 6; position++)
            for (int layer = 0; layer < 3; layer++)
                for (int head = 0; head < 2; head++)
                {
                    cache.Key(layer, head, position).Fill(-1f);
                    cache.Value(layer, head, position).Fill(-1f);
                }

        cache.Restore(snapshot, 4);
        Assert.Equal(4, cache.Length);
        Assert.Equal(expectedKeys, cache.KeyHistory(1, 1, 4).ToArray());

        void Randomize(Span<float> span)
        {
            for (int i = 0; i < span.Length; i++) span[i] = (float)rng.NextDouble();
        }
    }

    [Fact]
    public void RestoreRejectsAMissizedSnapshot()
    {
        var cache = new KvCache(layers: 2, kvHeads: 2, headDim: 4, initialCapacity: 4);
        cache.Advance(2);
        float[] snapshot = cache.Snapshot(2);
        Assert.Throws<ArgumentException>(() => cache.Restore(snapshot, 3));
    }

    [Fact]
    public void SnapshotRejectsMoreThanIsCached()
    {
        var cache = new KvCache(layers: 1, kvHeads: 1, headDim: 4, initialCapacity: 4);
        cache.Advance(1);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Snapshot(2));
    }
}
