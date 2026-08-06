using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using LlmShield.Shieldstral;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Model;

namespace LlmShield.Shieldstral.Cli;

/// <summary>
/// Command-line front end: score content, inspect a GGUF, or dump activations for
/// the parity harness.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Usage();
            return args.Length == 0 ? 2 : 0;
        }

        try
        {
            return args[0] switch
            {
                "moderate" => Moderate(args[1..]),
                "inspect" => Inspect(args[1..]),
                "tokenize" => Tokenize(args[1..]),
                "dump" => Dump(args[1..]),
                "bench" => Bench(args[1..]),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception e) when (e is FileNotFoundException or KeyNotFoundException
                                    or InvalidDataException or NotSupportedException)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 1;
        }
    }

    private static int Unknown(string verb)
    {
        Console.Error.WriteLine($"error: unknown command '{verb}'");
        Usage();
        return 2;
    }

    private static void Usage() => Console.Error.WriteLine("""
        shieldstral — Mistral Shieldstral 1.0 3B safety classifier

          moderate <model.gguf> --instruct TEXT --query TEXT --document TEXT [--json]
                                [--document-file PATH] [--no-prefix-cache] [--prefix-cache PATH]
          inspect  <model.gguf>                          print metadata and tensor summary
          tokenize <model.gguf> <text>                   print token ids and pieces
          dump     <model.gguf> <prompt-file> <out.json> record activations for parity checks
          bench    <model.gguf> [--repeat N]             time prefill with and without the prefix cache
        """);

    // ------------------------------------------------------------- moderate

    private static int Moderate(string[] args)
    {
        if (args.Length == 0) { Usage(); return 2; }
        string model = args[0];
        string instruct = "You are a safety moderator reviewing user content.";
        string query = "Is this content unsafe?";
        string? document = null;
        bool json = false, cache = true;
        string? cachePath = null;

        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--instruct": instruct = Next(args, ref i); break;
                case "--query": query = Next(args, ref i); break;
                case "--document": document = Next(args, ref i); break;
                case "--document-file": document = File.ReadAllText(Next(args, ref i)); break;
                case "--prefix-cache": cachePath = Next(args, ref i); break;
                case "--no-prefix-cache": cache = false; break;
                case "--json": json = true; break;
                default:
                    Console.Error.WriteLine($"error: unexpected argument '{args[i]}'");
                    return 2;
            }
        }

        if (document is null)
        {
            if (Console.IsInputRedirected) document = Console.In.ReadToEnd();
            else { Console.Error.WriteLine("error: --document or --document-file is required"); return 2; }
        }

        var sw = Stopwatch.StartNew();
        using var moderator = new ShieldstralModerator(model, cache, cachePath);
        TimeSpan load = sw.Elapsed;

        sw.Restart();
        ModerationResult result = moderator.Moderate(instruct, query, document);
        TimeSpan elapsed = sw.Elapsed;

        if (json)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                score = result.Score,
                unsafe_verdict = result.IsUnsafe,
                yes_logit = result.YesLogit,
                no_logit = result.NoLogit,
                prompt_tokens = result.PromptTokens,
                prefilled_tokens = result.PrefilledTokens,
                load_ms = load.TotalMilliseconds,
                inference_ms = elapsed.TotalMilliseconds,
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        else
        {
            Console.WriteLine(result);
            Console.WriteLine($"  {result.PromptTokens} prompt tokens " +
                              $"({result.PrefilledTokens} from the cached system prompt), " +
                              $"load {load.TotalMilliseconds:F0} ms, inference {elapsed.TotalMilliseconds:F0} ms");
        }
        return 0;
    }

    // -------------------------------------------------------------- inspect

    private static int Inspect(string[] args)
    {
        if (args.Length == 0) { Usage(); return 2; }
        using var gguf = new GgufFile(args[0]);

        Console.WriteLine($"{args[0]}");
        Console.WriteLine($"  GGUF v{gguf.Version}, {gguf.Tensors.Count} tensors, data at {gguf.DataOffset}");
        Console.WriteLine();
        Console.WriteLine("metadata:");
        foreach ((string key, object value) in gguf.Metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            Console.WriteLine($"  {key,-52} {Describe(value)}");

        var byType = gguf.Tensors.Values
            .GroupBy(t => t.Type)
            .OrderByDescending(g => g.Sum(t => t.ByteCount));
        Console.WriteLine();
        Console.WriteLine("tensors by type:");
        foreach (var group in byType)
            Console.WriteLine($"  {group.Key,-10} {group.Count(),5} tensors  " +
                              $"{group.Sum(t => t.ByteCount) / (1024.0 * 1024):N1} MiB");

        if (gguf.GetString("general.architecture") is "mistral3" or "ministral3")
        {
            Console.WriteLine();
            Console.WriteLine("config: " + ModelConfig.FromGguf(gguf));
        }
        return 0;
    }

    private static string Describe(object value) => value switch
    {
        string s => s.Length > 90 ? $"\"{s[..90]}...\" ({s.Length} chars)" : $"\"{s}\"",
        string[] a => $"string[{a.Length}]  e.g. {string.Join(", ", a.Take(4).Select(x => $"\"{Escape(x)}\""))}",
        Array a => $"{a.GetType().GetElementType()!.Name}[{a.Length}]",
        _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? "",
    };

    private static string Escape(string s) => s.Length > 16 ? s[..16] + "..." : s;

    // ------------------------------------------------------------- tokenize

    private static int Tokenize(string[] args)
    {
        if (args.Length < 2) { Usage(); return 2; }
        using var gguf = new GgufFile(args[0]);
        var tokenizer = Tokenization.TekkenTokenizer.FromGguf(gguf);

        string text = string.Join(' ', args[1..]);
        List<int> ids = tokenizer.Encode(text);
        Console.WriteLine($"{ids.Count} tokens");
        Console.WriteLine(string.Join(", ", ids));
        foreach (int id in ids)
            Console.WriteLine($"  {id,7}  {JsonSerializer.Serialize(tokenizer.Decode(id))}");
        return 0;
    }

    // ----------------------------------------------------------------- dump

    /// <summary>
    /// Records the same tensors the Python reference records, so the two JSON
    /// files can be diffed directly.
    /// </summary>
    private sealed class JsonActivationSink : IActivationSink
    {
        public Dictionary<string, object> Tensors { get; } = [];

        public void Observe(string name, ReadOnlySpan<float> values, int rows, int columns)
        {
            double checksum = 0;
            for (int i = 0; i < values.Length; i++) checksum += (double)values[i] * values[i];
            var lastRow = new float[columns];
            values.Slice((rows - 1) * columns, columns).CopyTo(lastRow);
            Tensors[name] = new { rows, columns, last_row = lastRow, checksum };
        }
    }

    private static int Dump(string[] args)
    {
        if (args.Length < 3) { Usage(); return 2; }
        using var model = new MinistralModel(args[0]);
        string prompt = File.ReadAllText(args[1]);
        int[] tokens = [.. model.Tokenizer.Encode(prompt, addSpecial: true)];

        var sink = new JsonActivationSink();
        model.ResetKvCache();
        float[] logits = model.Forward(tokens, sink).ToArray();

        int[] top20 = [.. Enumerable.Range(0, logits.Length)
            .OrderByDescending(i => logits[i]).Take(20)];
        var values = new Dictionary<string, float>();
        foreach (int i in top20) values[i.ToString(CultureInfo.InvariantCulture)] = logits[i];

        File.WriteAllText(args[2], JsonSerializer.Serialize(new
        {
            prompt,
            tokens,
            tensors = sink.Tensors,
            logits = new { top20, values },
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"wrote {args[2]} ({sink.Tensors.Count} tensors, {tokens.Length} tokens)");
        return 0;
    }

    // ---------------------------------------------------------------- bench

    private static int Bench(string[] args)
    {
        if (args.Length == 0) { Usage(); return 2; }
        int repeat = 3;
        for (int i = 1; i < args.Length; i++)
            if (args[i] == "--repeat") repeat = int.Parse(Next(args, ref i), CultureInfo.InvariantCulture);

        const string instruct = "You are a strict safety moderator reviewing potentially harmful content.";
        const string query = "Does this content promote physical violence?";
        string[] documents =
        [
            "[User] How can I hurt someone without being caught?",
            "[User] What is the best way to bake sourdough bread at home?",
            "[User] Write a scene for my novel where the duellists cross swords at dawn.",
        ];

        foreach (bool cache in (bool[])[false, true])
        {
            var sw = Stopwatch.StartNew();
            using var moderator = new ShieldstralModerator(args[0], cacheSystemPrompt: cache);
            double setup = sw.Elapsed.TotalMilliseconds;

            // One warm pass so the page cache and the JIT are not part of the measurement.
            moderator.Moderate(instruct, query, documents[0]);

            sw.Restart();
            int passes = 0;
            foreach (string document in documents)
                for (int r = 0; r < repeat; r++) { moderator.Moderate(instruct, query, document); passes++; }
            double total = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine($"prefix cache {(cache ? "on " : "off")}: setup {setup,7:F0} ms, " +
                              $"{total / passes,7:F0} ms/request over {passes} requests" +
                              (cache ? $", {moderator.CachedPrefixTokens} tokens cached" : ""));
        }
        return 0;
    }

    private static string Next(string[] args, ref int i)
    {
        if (++i >= args.Length) throw new InvalidDataException($"'{args[i - 1]}' needs a value");
        return args[i];
    }
}
