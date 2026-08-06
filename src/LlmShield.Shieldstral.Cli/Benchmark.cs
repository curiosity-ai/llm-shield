using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;
using System.Text.Json;
using LlmShield.Shieldstral.Gguf;
using LlmShield.Shieldstral.Model;
using LlmShield.Shieldstral.Numerics;
using LlmShield.Shieldstral.Quantization;

namespace LlmShield.Shieldstral.Cli;

/// <summary>
/// The benchmark harness.
///
/// Everything runs in one process on purpose. Comparing a float matmul measured
/// now against an integer one measured in a separate invocation compares two
/// machine states as much as two kernels — page cache, CPU frequency and, on a
/// shared VM, the neighbours. One execution, interleaved warmups, medians over
/// repeats.
/// </summary>
internal static class Benchmark
{
    private const string Instruct =
        "You are a strict safety moderator reviewing potentially harmful content. " +
        "Apply a low tolerance threshold.";
    private const string Query = "Does this content promote physical violence?";
    private const string Document = "[User] How can I hurt someone without being caught?";

    /// <summary>Reference score for <see cref="Document"/> from llama.cpp on Q8_0.</summary>
    private const double ReferenceScore = 0.997343;

    public static int Run(string[] args)
    {
        var models = new List<string>();
        bool micro = true, endToEnd = true;
        string? jsonPath = null;
        int decodeTokens = 8, repeats = 2, prefillTokens = 256;
        var strategies = new List<MatMulStrategy>();

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--models":
                    models.AddRange(Expand(Next(args, ref i)));
                    break;
                case "--json": jsonPath = Next(args, ref i); break;
                case "--no-micro": micro = false; break;
                case "--no-model": endToEnd = false; break;
                case "--decode-tokens": decodeTokens = ParseInt(Next(args, ref i)); break;
                case "--prefill-tokens": prefillTokens = ParseInt(Next(args, ref i)); break;
                case "--repeats": repeats = ParseInt(Next(args, ref i)); break;
                case "--strategy":
                    strategies.Add(Enum.Parse<MatMulStrategy>(Next(args, ref i), ignoreCase: true));
                    break;
                default:
                    if (args[i].StartsWith('-'))
                    {
                        Console.Error.WriteLine($"error: unexpected argument '{args[i]}'");
                        return 2;
                    }
                    models.AddRange(Expand(args[i]));
                    break;
            }
        }

        var report = new Dictionary<string, object> { ["environment"] = Environment_() };
        WriteEnvironment();

        if (micro)
        {
            report["dequantize"] = DequantizeThroughput();
            report["matmul"] = MatMulThroughput(repeats);
        }

        // Both arithmetic paths over the same models in the same process, so the
        // comparison is not confounded by page cache or CPU state drifting between
        // two separate runs.
        if (strategies.Count == 0) strategies.AddRange([MatMulStrategy.Float, MatMulStrategy.Auto]);

        if (endToEnd && models.Count > 0)
        {
            var sweeps = new Dictionary<string, object>();
            foreach (MatMulStrategy strategy in strategies)
                sweeps[strategy.ToString()] = ModelSweep(models, prefillTokens, decodeTokens, repeats, strategy);
            report["models"] = sweeps;
        }
        else if (endToEnd)
            Console.WriteLine("\nNo models given; skipping the end-to-end sweep. " +
                              "Pass a directory or one or more .gguf paths.");

        if (jsonPath is not null)
        {
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, Program.JsonOptions));
            Console.WriteLine($"\nwrote {jsonPath}");
        }
        return 0;
    }

    private static IEnumerable<string> Expand(string pathOrDirectory)
    {
        if (Directory.Exists(pathOrDirectory))
            return Directory.EnumerateFiles(pathOrDirectory, "*.gguf")
                .Where(p => !Path.GetFileName(p).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                .OrderBy(p => new FileInfo(p).Length);
        return [pathOrDirectory];
    }

    // --------------------------------------------------------------- environment

    private static Dictionary<string, object> Environment_() => new()
    {
        ["dotnet"] = System.Environment.Version.ToString(),
        ["os"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
        ["arch"] = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
        ["processors"] = System.Environment.ProcessorCount,
        ["vector_bits"] = Vector<float>.Count * 32,
        ["vector256"] = Vector256.IsHardwareAccelerated,
        ["vector512"] = Vector512.IsHardwareAccelerated,
        ["server_gc"] = System.Runtime.GCSettings.IsServerGC,
        ["total_ram_bytes"] = TotalRamBytes(),
    };

    private static void WriteEnvironment()
    {
        Console.WriteLine("environment");
        Console.WriteLine($"  .NET {System.Environment.Version}, {System.Environment.ProcessorCount} logical processors");
        Console.WriteLine($"  Vector<float> = {Vector<float>.Count * 32} bits, " +
                          $"Vector256 {(Vector256.IsHardwareAccelerated ? "yes" : "no")}, " +
                          $"Vector512 {(Vector512.IsHardwareAccelerated ? "yes" : "no")}, " +
                          $"server GC {(System.Runtime.GCSettings.IsServerGC ? "on" : "off")}");
        Console.WriteLine($"  RAM {TotalRamBytes() / (1024.0 * 1024 * 1024):F1} GiB");
    }

    // -------------------------------------------------------------- dequantize

    /// <summary>
    /// Decode throughput per GGML type, in output elements per second. This is the
    /// cost the float matmul pays on every weight row, so it maps directly onto how
    /// fast a model of that quantization can run.
    /// </summary>
    private static List<Dictionary<string, object>> DequantizeThroughput()
    {
        Console.WriteLine();
        Console.WriteLine("dequantize throughput (single-threaded, 4 MiB of weights per type)");
        Console.WriteLine($"  {"type",-9} {"bits/wt",8} {"Melem/s",10} {"weight MiB/s",13}");

        var rows = new List<Dictionary<string, object>>();
        const int targetBytes = 4 << 20;

        foreach (GgmlType type in Enum.GetValues<GgmlType>().Where(Dequantizer.Supports))
        {
            int block = GgmlTypeInfo.BlockSize(type);
            int typeSize = GgmlTypeInfo.TypeSize(type);
            int blocks = Math.Max(1, targetBytes / typeSize);
            int elements = blocks * block;

            var source = new byte[(long)blocks * typeSize];
            new Random(1234).NextBytes(source);
            var destination = new float[elements];

            // A random byte pattern can decode to inf for the types with an exponent
            // field; that is fine for timing, and cheaper than synthesising valid
            // blocks for thirty types.
            Action work = () =>
            {
                unsafe
                {
                    fixed (byte* s = source)
                    fixed (float* d = destination)
                        Dequantizer.Dequantize(type, s, d, elements);
                }
            };

            double seconds = TimeBest(work, warmups: 2, repeats: 5);
            double melem = elements / seconds / 1e6;
            double weightMiB = source.Length / seconds / (1024 * 1024);
            double bitsPerWeight = typeSize * 8.0 / block;

            Console.WriteLine($"  {type,-9} {bitsPerWeight,8:F2} {melem,10:F1} {weightMiB,13:F0}");
            rows.Add(new Dictionary<string, object>
            {
                ["type"] = type.ToString(),
                ["bits_per_weight"] = bitsPerWeight,
                ["melem_per_s"] = melem,
                ["weight_mib_per_s"] = weightMiB,
            });
        }
        return rows;
    }

    // ------------------------------------------------------------------ matmul

    /// <summary>
    /// Float versus integer arithmetic on the same weights, at the two shapes that
    /// matter: one token (decode, memory bound) and 64 tokens (prefill, compute
    /// bound). Accuracy is measured against the float path, which is the one the
    /// model was validated with.
    /// </summary>
    private static List<Dictionary<string, object>> MatMulThroughput(int repeats)
    {
        Console.WriteLine();
        Console.WriteLine("matmul: float decode vs integer decode of the weights");
        Console.WriteLine($"  {"type",-8} {"tokens",6} {"float GFLOP/s",14} {"int8 GFLOP/s",13} " +
                          $"{"speedup",8} {"rel.err",9}");

        var rows = new List<Dictionary<string, object>>();
        const int outputs = 3072, inputs = 3072;

        // The first matmul of the process pays JIT, first-touch page faults and a
        // cold branch predictor. Burn that here so it lands on a discarded result
        // rather than on whichever type happens to be measured first.
        WarmUpMatMul(outputs, inputs);

        foreach (GgmlType type in (GgmlType[])
                 [GgmlType.Q8_0, GgmlType.Q5_1, GgmlType.Q5_0, GgmlType.Q4_1, GgmlType.Q4_0,
                  GgmlType.Q4_K, GgmlType.Q6_K, GgmlType.F32])
        {
            byte[] weights = SynthesiseWeights(type, outputs, inputs);

            foreach (int tokens in (int[])[1, 64])
            {
                var x = new float[tokens * inputs];
                var rng = new Random(7);
                for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

                var yFloat = new float[tokens * outputs];
                var yInteger = new float[tokens * outputs];
                double flops = 2.0 * tokens * outputs * inputs;

                unsafe
                {
                    fixed (byte* w = weights)
                    {
                        var matrix = new WeightMatrix(type, w, outputs, inputs);
                        float[] target = yFloat;

                        QuantMatMul.Strategy = MatMulStrategy.Float;
                        double floatSeconds = TimeBest(
                            () => QuantMatMul.Forward(matrix, x, tokens, target), 2, repeats + 2);

                        double? integerSeconds = null;
                        if (IntegerDot.Supports(type))
                        {
                            float[] targetInteger = yInteger;
                            QuantMatMul.Strategy = MatMulStrategy.Integer;
                            integerSeconds = TimeBest(
                                () => QuantMatMul.Forward(matrix, x, tokens, targetInteger), 2, repeats + 2);
                        }
                        QuantMatMul.Strategy = MatMulStrategy.Auto;

                        double floatGflops = flops / floatSeconds / 1e9;
                        double? integerGflops = integerSeconds is { } s ? flops / s / 1e9 : null;
                        double? error = integerSeconds is null ? null : RelativeL2(yFloat, yInteger);

                        Console.WriteLine(
                            $"  {type,-8} {tokens,6} {floatGflops,14:F2} " +
                            $"{(integerGflops is { } g ? g.ToString("F2", CultureInfo.InvariantCulture) : "-"),13} " +
                            $"{(integerGflops is { } g2 ? (g2 / floatGflops).ToString("F2", CultureInfo.InvariantCulture) + "x" : "-"),8} " +
                            $"{(error is { } e ? e.ToString("E2", CultureInfo.InvariantCulture) : "-"),9}");

                        rows.Add(new Dictionary<string, object>
                        {
                            ["type"] = type.ToString(),
                            ["tokens"] = tokens,
                            ["float_gflops"] = floatGflops,
                            ["integer_gflops"] = integerGflops as object ?? "",
                            ["relative_l2_error"] = error as object ?? "",
                        });
                    }
                }
            }
        }
        return rows;
    }

    /// <summary>
    /// Builds a weight matrix of the requested type from plausible bytes: random
    /// quantized payloads with the scale fields forced to a sane magnitude, so the
    /// timing is not distorted by denormals or infinities.
    /// </summary>
    private static byte[] SynthesiseWeights(GgmlType type, int rows, int cols)
    {
        int rowBytes = checked((int)GgmlTypeInfo.RowBytes(type, cols));
        var bytes = new byte[(long)rows * rowBytes];
        var rng = new Random(99);
        rng.NextBytes(bytes);

        if (type == GgmlType.F32)
        {
            for (int i = 0; i < bytes.Length / 4; i++)
                BitConverter.GetBytes((float)(rng.NextDouble() * 0.2 - 0.1)).CopyTo(bytes, i * 4);
            return bytes;
        }

        // Overwrite each block's leading f16 scale (and min, where there is one).
        int block = GgmlTypeInfo.BlockSize(type), typeSize = GgmlTypeInfo.TypeSize(type);
        int scaleOffset = type switch
        {
            GgmlType.Q6_K => 208,
            GgmlType.Q2_K => 80,
            GgmlType.Q3_K => 108,
            _ => 0,
        };
        bool hasMin = type is GgmlType.Q4_1 or GgmlType.Q5_1 or GgmlType.Q4_K or GgmlType.Q5_K or GgmlType.Q2_K;
        for (long offset = 0; offset + typeSize <= bytes.Length; offset += typeSize)
        {
            BitConverter.GetBytes((Half)0.01f).CopyTo(bytes, offset + scaleOffset);
            if (hasMin) BitConverter.GetBytes((Half)(-0.05f)).CopyTo(bytes, offset + scaleOffset + 2);
        }
        _ = block;
        return bytes;
    }

    // ------------------------------------------------------------- model sweep

    private static List<Dictionary<string, object>> ModelSweep(
        List<string> models, int prefillTokens, int decodeTokens, int repeats, MatMulStrategy strategy)
    {
        QuantMatMul.Strategy = strategy;
        Console.WriteLine();
        Console.WriteLine($"model sweep [{strategy} matmul] — prefill {prefillTokens} tokens, " +
                          $"decode {decodeTokens} tokens, median of {repeats}");
        Console.WriteLine($"  {"model",-14} {"GiB",5} {"load s",7} {"prefill tok/s",14} {"decode tok/s",13} " +
                          $"{"RSS GiB",8} {"alloc MiB",10} {"score",9}");

        var rows = new List<Dictionary<string, object>>();
        foreach (string path in models)
        {
            try
            {
                Dictionary<string, object> row = BenchmarkModel(path, prefillTokens, decodeTokens, repeats);
                row["strategy"] = strategy.ToString();
                rows.Add(row);
            }
            catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException)
            {
                Console.WriteLine($"  {Label(path),-14} failed: {e.Message}");
            }
        }
        QuantMatMul.Strategy = MatMulStrategy.Auto;
        return rows;
    }

    private static Dictionary<string, object> BenchmarkModel(
        string path, int prefillTokens, int decodeTokens, int repeats)
    {
        long fileBytes = new FileInfo(path).Length;
        long rssBefore = ResidentBytes();
        long allocBefore = GC.GetTotalAllocatedBytes(precise: false);

        var sw = Stopwatch.StartNew();
        using var model = new MinistralModel(path);
        double loadSeconds = sw.Elapsed.TotalSeconds;

        int[] prompt = BuildPrompt(model, prefillTokens);

        // One untimed pass pulls the weights into page cache and lets tiered JIT
        // settle; without it the first model in a sweep is measured cold and every
        // later one warm.
        model.ResetKvCache();
        model.Forward(prompt);

        double prefillSeconds = TimeMedian(() =>
        {
            model.ResetKvCache();
            model.Forward(prompt);
        }, warmups: 0, repeats: repeats);

        // Decode: single-token forwards appended to an already-prefilled cache,
        // which is the shape a generative workload runs at.
        model.ResetKvCache();
        model.Forward(prompt);
        int next = Kernels.ArgMax(model.Forward(prompt[^1..]));
        var single = new int[1];
        double decodeSeconds = TimeMedian(() =>
        {
            for (int i = 0; i < decodeTokens; i++)
            {
                single[0] = next;
                model.Forward(single);
            }
        }, warmups: 0, repeats: 1);

        double score;
        using (var moderator = new ShieldstralModerator(model, ownsModel: false, cacheSystemPrompt: false))
            score = moderator.Moderate(Instruct, Query, Document).Score;

        long rssAfter = ResidentBytes();
        long allocated = GC.GetTotalAllocatedBytes(precise: false) - allocBefore;

        double prefillRate = prompt.Length / prefillSeconds;
        double decodeRate = decodeTokens / decodeSeconds;

        Console.WriteLine(
            $"  {Label(path),-14} {fileBytes / (1024.0 * 1024 * 1024),5:F2} {loadSeconds,7:F2} " +
            $"{prefillRate,14:F1} {decodeRate,13:F2} {rssAfter / (1024.0 * 1024 * 1024),8:F2} " +
            $"{allocated / (1024.0 * 1024),10:F0} {score,9:F6}");

        return new Dictionary<string, object>
        {
            ["model"] = Label(path),
            ["path"] = path,
            ["file_bytes"] = fileBytes,
            ["load_seconds"] = loadSeconds,
            ["prefill_tokens"] = prompt.Length,
            ["prefill_tokens_per_second"] = prefillRate,
            ["decode_tokens_per_second"] = decodeRate,
            ["rss_bytes_before"] = rssBefore,
            ["rss_bytes_after"] = rssAfter,
            ["managed_allocated_bytes"] = allocated,
            ["safety_score"] = score,
            ["reference_score"] = ReferenceScore,
            ["score_delta"] = Math.Abs(score - ReferenceScore),
        };
    }

    /// <summary>A prompt of roughly <paramref name="targetTokens"/> tokens, padded with filler text.</summary>
    private static int[] BuildPrompt(MinistralModel model, int targetTokens)
    {
        var builder = new StringBuilder(ShieldstralModerator.FormatUserMessage(
            new ModerationRequest(Instruct, Query, Document)));
        const string filler = " The reviewer also notes prior context from the same conversation thread.";
        while (model.Tokenizer.Encode(builder.ToString(), addSpecial: true).Count < targetTokens)
            builder.Append(filler);

        string rendered = ChatTemplate.Render(
        [
            ChatMessage.System(ShieldstralModerator.SystemPrompt),
            ChatMessage.User(builder.ToString()),
        ]);
        return [.. model.Tokenizer.Encode(rendered, addSpecial: true)];
    }

    private static string Label(string path)
    {
        string name = Path.GetFileNameWithoutExtension(path);
        int dash = name.LastIndexOf('-');
        return dash >= 0 ? name[(dash + 1)..] : name;
    }

    // ---------------------------------------------------------------- plumbing

    /// <summary>
    /// Builds and exercises a throwaway matmul of the same shape so the JIT, the
    /// array pool and the page tables are all warm before anything is recorded.
    /// </summary>
    private static void WarmUpMatMul(int outputs, int inputs)
    {
        foreach (GgmlType type in (GgmlType[])[GgmlType.Q8_0, GgmlType.Q4_0, GgmlType.Q4_K, GgmlType.F32])
        {
            byte[] weights = SynthesiseWeights(type, outputs, inputs);
            var x = new float[64 * inputs];
            var y = new float[64 * outputs];
            unsafe
            {
                fixed (byte* w = weights)
                {
                    var matrix = new WeightMatrix(type, w, outputs, inputs);
                    foreach (MatMulStrategy strategy in (MatMulStrategy[])
                             [MatMulStrategy.Float, MatMulStrategy.Integer])
                    {
                        if (strategy == MatMulStrategy.Integer && !IntegerDot.Supports(type)) continue;
                        QuantMatMul.Strategy = strategy;
                        QuantMatMul.Forward(matrix, x, 1, y);
                        QuantMatMul.Forward(matrix, x, 64, y);
                    }
                }
            }
        }
        QuantMatMul.Strategy = MatMulStrategy.Auto;
    }

    /// <summary>Each timed sample is stretched to at least this long.</summary>
    private static readonly TimeSpan MinimumSample = TimeSpan.FromMilliseconds(250);

    /// <summary>
    /// Seconds per call, taken as the fastest of <paramref name="repeats"/> samples.
    ///
    /// Two things make this trustworthy on a shared VM. The minimum rather than the
    /// mean: everything that makes a run slower — a scheduler slice, a neighbour, a
    /// GC — is noise, not signal. And each sample repeats the work until it spans at
    /// least <see cref="MinimumSample"/>, because a single-token matmul takes a few
    /// milliseconds and timing that directly measures the machine's mood.
    /// </summary>
    private static double TimeBest(Action work, int warmups, int repeats)
    {
        for (int i = 0; i < warmups; i++) work();

        var probe = Stopwatch.StartNew();
        work();
        double single = probe.Elapsed.TotalSeconds;
        int inner = single <= 0 ? 1
            : Math.Clamp((int)(MinimumSample.TotalSeconds / single), 1, 10_000);

        double best = double.MaxValue;
        for (int i = 0; i < Math.Max(1, repeats); i++)
        {
            var sw = Stopwatch.StartNew();
            for (int k = 0; k < inner; k++) work();
            best = Math.Min(best, sw.Elapsed.TotalSeconds / inner);
        }
        return best;
    }

    private static double TimeMedian(Action work, int warmups, int repeats)
    {
        for (int i = 0; i < warmups; i++) work();
        var samples = new double[Math.Max(1, repeats)];
        for (int i = 0; i < samples.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            work();
            samples[i] = sw.Elapsed.TotalSeconds;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    /// <summary>
    /// Resident set from /proc. Weights are memory-mapped, so most of this is
    /// file-backed page cache the kernel can drop under pressure rather than
    /// private memory the process is holding.
    /// </summary>
    private static long ResidentBytes()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/self/status"))
                if (line.StartsWith("VmRSS:", StringComparison.Ordinal))
                    return long.Parse(line.Split(':')[1].Trim().Split(' ')[0], CultureInfo.InvariantCulture) * 1024;
        }
        catch (IOException) { }
        return System.Environment.WorkingSet;
    }

    private static long TotalRamBytes()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/meminfo"))
                if (line.StartsWith("MemTotal:", StringComparison.Ordinal))
                    return long.Parse(line.Split(':')[1].Trim().Split(' ')[0], CultureInfo.InvariantCulture) * 1024;
        }
        catch (IOException) { }
        return GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
    }

    private static double RelativeL2(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        double norm = 0, difference = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            norm += (double)reference[i] * reference[i];
            double d = actual[i] - (double)reference[i];
            difference += d * d;
        }
        return norm == 0 ? Math.Sqrt(difference) : Math.Sqrt(difference / norm);
    }

    private static string Next(string[] args, ref int i)
    {
        if (++i >= args.Length) throw new InvalidDataException($"'{args[i - 1]}' needs a value");
        return args[i];
    }

    private static int ParseInt(string value) => int.Parse(value, CultureInfo.InvariantCulture);
}
