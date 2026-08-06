using System.Text.Json;
using LlmShield.Shieldstral.Model;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Layer-by-layer comparison against the NumPy reference on the same prompt.
///
/// An end-to-end score check tells you *that* something is wrong; this tells you
/// *where*. The tensors are recorded in forward order, so the first one to
/// diverge names the broken step — a RoPE pairing mistake shows up at
/// <c>q_rope</c>, a transposed weight at <c>attn_out</c>, a wrong norm epsilon at
/// <c>attn_norm</c> — and everything downstream of it is just the same error
/// carried forward.
///
/// The reference runs on unquantized weights while the runtime reads a quantized
/// GGUF, so the tolerance widens with depth as quantization error accumulates
/// through the residual stream. That is expected; a wiring bug moves values by
/// orders of magnitude more than this.
/// </summary>
public class ActivationParityTests
{
    private readonly ITestOutputHelper _output;
    public ActivationParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>Records what the model computes, keyed the same way the reference does.</summary>
    private sealed class Recorder : IActivationSink
    {
        public Dictionary<string, (float[] LastRow, double Checksum, int Rows, int Columns)> Tensors { get; } = [];

        public void Observe(string name, ReadOnlySpan<float> values, int rows, int columns)
        {
            double checksum = 0;
            for (int i = 0; i < values.Length; i++) checksum += (double)values[i] * values[i];
            Tensors[name] = (values.Slice((rows - 1) * columns, columns).ToArray(), checksum, rows, columns);
        }
    }

    [Fact]
    public void EveryRecordedTensorMatchesTheReference()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using JsonDocument fixture = Fixtures.Load("activations.json");
        int[] tokens = [.. fixture.RootElement.GetProperty("tokens").EnumerateArray().Select(v => v.GetInt32())];

        using var model = new MinistralModel(modelPath);

        // Tokenizing the fixture's prompt must reproduce the fixture's ids, or the
        // two runs are not comparing the same computation at all.
        int[] retokenized = [.. model.Tokenizer.Encode(
            fixture.RootElement.GetProperty("prompt").GetString()!, addSpecial: true)];
        Assert.Equal(tokens, retokenized);

        var recorder = new Recorder();
        model.ResetKvCache();
        model.Forward(tokens, recorder);

        // Relative L2 error, not worst element. The reference runs unquantized and
        // the runtime reads Q8_0, so individual elements legitimately differ by a
        // few percent of the tensor's own scale — RMS norm amplifies the embedding
        // table's quantization error by ~120x, which is arithmetic, not a bug. The
        // direction of the whole vector is what a wiring error changes, and it moves
        // that by orders of magnitude more than this.
        //
        // Measured on the Q8_0 build: every tensor lands at or below 3.4e-2, and the
        // error does not grow with depth (the last layer is the tightest at 3.0e-3).
        double tolerance = Fixtures.IsQ8Reference(modelPath) ? 0.05 : 0.15;

        var failures = new List<string>();
        int compared = 0;
        double worstOverall = 0;
        foreach (JsonProperty entry in fixture.RootElement.GetProperty("tensors").EnumerateObject())
        {
            string name = entry.Name;
            if (!recorder.Tensors.TryGetValue(name, out var actual))
            {
                failures.Add($"{name}: the runtime recorded no such tensor");
                continue;
            }

            int rows = entry.Value.GetProperty("rows").GetInt32();
            int columns = entry.Value.GetProperty("columns").GetInt32();
            Assert.Equal(columns, actual.Columns);
            Assert.Equal(rows, actual.Rows);

            float[] expected = [.. entry.Value.GetProperty("last_row").EnumerateArray().Select(v => v.GetSingle())];
            Assert.All(actual.LastRow, v => Assert.True(float.IsFinite(v),
                $"{name} contains {v}; a non-finite activation poisons every later layer"));

            double error = RelativeL2(expected, actual.LastRow);
            worstOverall = Math.Max(worstOverall, error);

            // The recorded checksum is the tensor's *squared* Frobenius norm, so
            // compare the norms: squaring doubles every relative error, and from
            // layer 2 on these tensors carry a handful of massive activations whose
            // magnitude dominates the sum entirely. The last row and the norm are
            // then on the same scale and can share a tolerance.
            double expectedNorm = Math.Sqrt(entry.Value.GetProperty("checksum").GetDouble());
            double actualNorm = Math.Sqrt(actual.Checksum);
            double normError = Math.Abs(actualNorm - expectedNorm) / Math.Max(1e-9, expectedNorm);

            _output.WriteLine($"{name,-24} [{rows,3}x{columns,6}] " +
                              $"L2 {error:E2}  norm {normError:E2}");

            if (error > tolerance)
                failures.Add($"{name}: last row differs by {error:E3} in relative L2 (tolerance {tolerance:E1})");
            if (normError > tolerance)
                failures.Add($"{name}: Frobenius norm differs by {normError:E3} " +
                             $"({actualNorm:E6} vs {expectedNorm:E6})");
            compared++;
        }
        _output.WriteLine($"worst relative L2 across {compared} tensors: {worstOverall:E2}");

        Assert.True(compared > 0, "the fixture recorded no tensors");
        Assert.True(compared >= 30,
            $"only {compared} tensors were compared; the fixture should cover the leading layers " +
            "plus the final norm — regenerate it (see CLAUDE.md).");
        Assert.True(failures.Count == 0,
            $"{failures.Count} of {compared} tensors diverged from the NumPy reference:\n  " +
            string.Join("\n  ", failures.Take(8)));
    }

    /// <summary>
    /// <c>‖actual - expected‖₂ / ‖expected‖₂</c> — how far the vector moved, relative
    /// to how big it is.
    /// </summary>
    private static double RelativeL2(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        double norm = 0, difference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            norm += (double)expected[i] * expected[i];
            double d = actual[i] - (double)expected[i];
            difference += d * d;
        }
        return norm == 0 ? Math.Sqrt(difference) : Math.Sqrt(difference / norm);
    }

    /// <summary>
    /// The verdict distribution itself: the same top-20 tokens in the same order.
    /// Ordering is a stricter check than the score, which can survive two logits
    /// swapping places if they are close.
    /// </summary>
    [Fact]
    public void TopVerdictTokensMatchTheReference()
    {
        string? modelPath = Fixtures.ModelPath;
        if (modelPath is null)
        {
            _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
            return;
        }

        using JsonDocument fixture = Fixtures.Load("activations.json");
        int[] tokens = [.. fixture.RootElement.GetProperty("tokens").EnumerateArray().Select(v => v.GetInt32())];
        int[] expected = [.. fixture.RootElement.GetProperty("logits").GetProperty("top20")
            .EnumerateArray().Select(v => v.GetInt32())];

        using var model = new MinistralModel(modelPath);
        model.ResetKvCache();
        float[] logits = model.Forward(tokens).ToArray();

        int[] actual = [.. Enumerable.Range(0, logits.Length)
            .OrderByDescending(i => logits[i]).Take(expected.Length)];

        _output.WriteLine($"reference: {string.Join(", ", expected.Take(8))}");
        _output.WriteLine($"runtime:   {string.Join(", ", actual.Take(8))}");

        // The top few carry the verdict; deep in the tail, quantization can
        // legitimately reorder near-tied logits.
        Assert.Equal(expected[..5], actual[..5]);
        Assert.True(expected.Intersect(actual).Count() >= expected.Length - 3,
            $"only {expected.Intersect(actual).Count()} of the reference's top {expected.Length} " +
            "tokens appear in the runtime's");
    }
}
