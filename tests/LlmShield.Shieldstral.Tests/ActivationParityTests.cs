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

        // Q8_0 keeps ~3 significant digits per weight; the residual stream accumulates
        // that over 26 layers, so a fixed tolerance would either pass everything or
        // fail the deep layers for no reason.
        double tolerance = Fixtures.IsQ8Reference(modelPath) ? 0.02 : 0.10;

        var failures = new List<string>();
        int compared = 0;
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
            double worst = Numeric.WorstRelativeError(expected, actual.LastRow);

            double expectedChecksum = entry.Value.GetProperty("checksum").GetDouble();
            double checksumError = Math.Abs(actual.Checksum - expectedChecksum)
                                 / Math.Max(1.0, Math.Abs(expectedChecksum));

            _output.WriteLine($"{name,-24} [{rows,3}x{columns,6}] " +
                              $"row {worst:E2}  checksum {checksumError:E2}");

            if (worst > tolerance)
                failures.Add($"{name}: last row differs by {worst:E3} (tolerance {tolerance:E1})");
            if (checksumError > tolerance)
                failures.Add($"{name}: sum of squares differs by {checksumError:E3} " +
                             $"({actual.Checksum:E6} vs {expectedChecksum:E6})");
            compared++;
        }

        Assert.True(compared > 0, "the fixture recorded no tensors");
        Assert.True(failures.Count == 0,
            $"{failures.Count} of {compared} tensors diverged from the NumPy reference:\n  " +
            string.Join("\n  ", failures.Take(8)));
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
