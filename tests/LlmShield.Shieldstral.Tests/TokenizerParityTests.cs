using System.Text.Json;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace LlmShield.Shieldstral.Tests;

/// <summary>
/// Token-for-token comparison against the Python reference.
///
/// The two implementations do not share an algorithm: the reference walks
/// tekken's ranked vocabulary the way <c>mistral-common</c> does, while the
/// runtime applies the merge table the converter derived from those same ranks.
/// Agreement therefore checks the derivation as well as the tokenizer — and a
/// single mis-derived merge shifts every token after it, so this is the cheapest
/// place to catch a whole class of "the model loads but answers nonsense" bugs.
/// </summary>
public class TokenizerParityTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly GgufFile? _gguf;
    private readonly TekkenTokenizer? _tokenizer;

    public TokenizerParityTests(ITestOutputHelper output)
    {
        _output = output;
        string? model = Fixtures.ModelPath;
        if (model is null) return;
        _gguf = new GgufFile(model);
        _tokenizer = TekkenTokenizer.FromGguf(_gguf);
    }

    private bool Skip()
    {
        if (_tokenizer is not null) return false;
        _output.WriteLine($"{Fixtures.ModelEnvironmentVariable} is not set; skipping");
        return true;
    }

    [Fact]
    public void EncodesReferencePromptsIdentically()
    {
        if (Skip()) return;
        using JsonDocument doc = Fixtures.Load("tokenizer.json");

        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = c.GetProperty("name").GetString()!;
            string prompt = c.GetProperty("prompt").GetString()!;
            int[] expected = [.. c.GetProperty("tokens").EnumerateArray().Select(v => v.GetInt32())];

            int[] actual = [.. _tokenizer!.Encode(prompt, addSpecial: true)];
            AssertSameTokens(name, expected, actual);
            _output.WriteLine($"{name,-22} {actual.Length} tokens");
        }
    }

    [Fact]
    public void EncodesPreTokenizerEdgeCasesIdentically()
    {
        if (Skip()) return;
        using JsonDocument doc = Fixtures.Load("tokenizer.json");

        foreach (JsonElement c in doc.RootElement.GetProperty("strings").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            int[] expected = [.. c.GetProperty("tokens").EnumerateArray().Select(v => v.GetInt32())];
            int[] actual = [.. _tokenizer!.Encode(text, addSpecial: false)];
            AssertSameTokens(JsonSerializer.Serialize(text), expected, actual);
        }
    }

    /// <summary>
    /// Which ids spell "yes" and "no" decides the safety score outright, so the
    /// runtime and the reference must agree on the whole set, not just the common
    /// spelling.
    /// </summary>
    [Fact]
    public void ResolvesTheSameVerdictTokens()
    {
        if (Skip()) return;
        using JsonDocument doc = Fixtures.Load("tokenizer.json");
        JsonElement verdicts = doc.RootElement.GetProperty("verdict_tokens");

        foreach (string form in (string[])["yes", "no"])
        {
            int[] expected = [.. verdicts.GetProperty(form).EnumerateArray().Select(v => v.GetInt32())];
            int[] actual = [.. Enumerable.Range(0, _tokenizer!.VocabSize)
                .Where(id => _tokenizer.Decode(id).Trim().Trim('"', '\'', '.').ToLowerInvariant() == form)];
            Assert.Equal(expected.Order(), actual.Order());
            _output.WriteLine($"'{form}' tokens: {string.Join(", ", actual)}");
        }
    }

    [Fact]
    public void RoundTripsDecodeOfEncode()
    {
        if (Skip()) return;
        foreach (string text in (string[])
                 ["hello world", "  leading space", "3.14159", "日本語のテキスト",
                  "emoji 🚨 test", "don't", "C:\\path\\to/file"])
        {
            List<int> ids = _tokenizer!.Encode(text, addSpecial: false);
            Assert.Equal(text, _tokenizer.Decode(ids));
        }
    }

    [Fact]
    public void ControlMarkersTokenizeToTheirOwnIds()
    {
        if (Skip()) return;
        // The prompt template's markers must never be split into text pieces.
        foreach (string marker in (string[])
                 ["[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]", "[IMG]"])
        {
            List<int> ids = _tokenizer!.Encode(marker, addSpecial: false);
            Assert.Single(ids);
            Assert.Equal(_tokenizer.Lookup(marker), ids[0]);
        }
    }

    [Fact]
    public void AddsExactlyOneBosToken()
    {
        if (Skip()) return;
        List<int> ids = _tokenizer!.Encode("hello", addSpecial: true);
        Assert.Equal(_tokenizer.BosTokenId, ids[0]);
        Assert.DoesNotContain(_tokenizer.BosTokenId, ids.Skip(1));
        Assert.DoesNotContain(_tokenizer.BosTokenId, _tokenizer.Encode("hello", addSpecial: false));
    }

    private static void AssertSameTokens(string label, int[] expected, int[] actual)
    {
        if (expected.SequenceEqual(actual)) return;

        int at = 0;
        while (at < expected.Length && at < actual.Length && expected[at] == actual[at]) at++;
        Assert.Fail(
            $"{label}: token streams diverge at index {at} " +
            $"(reference {(at < expected.Length ? expected[at] : -1)}, " +
            $"runtime {(at < actual.Length ? actual[at] : -1)}); " +
            $"lengths {expected.Length} vs {actual.Length}");
    }

    public void Dispose() => _gguf?.Dispose();
}
