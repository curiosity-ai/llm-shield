# CLAUDE.md

Guidance for working in this repository.

## What this is

A managed .NET 10 runtime for one model: Mistral's Shieldstral 1.0 3B safety
classifier. It is a reduction of
[TensorSharp](https://github.com/zhongkaifu/TensorSharp) (BSD-3-Clause) down to
what this single architecture needs, with no native dependency and no backend
abstraction — one CPU path, written around `System.Numerics.Tensors` and
`Vector<T>`.

Files carrying logic derived from TensorSharp say so in their header. Keep that
attribution when you move code between files; `third-party/TensorSharp-LICENSE`
is the licence it is carried under.

## Layout

```
src/LlmShield.Shieldstral/
  Gguf/            GgmlType.cs (block geometry), GgufFile.cs (mmap reader)
  Quantization/    Dequantizer.cs (every GGML type), QuantGrids.g.cs (generated),
                   IntegerGemm.cs (the prefill kernel), IntegerDot.cs (its fallback)
  Numerics/        Kernels.cs, QuantMatMul.cs, WeightMatrix.cs
  Tokenization/    TekkenTokenizer.cs
  Model/           ModelConfig.cs, Rope.cs, KvCache.cs, MinistralModel.cs
  ChatTemplate.cs, SystemPromptCache.cs, ShieldstralModerator.cs, ModelDownloader.cs
src/LlmShield.Shieldstral.Cli/    the `shieldstral` command
tests/LlmShield.Shieldstral.Tests/
tests/fixtures/                   generated oracles (JSON), committed
tools/                            Python: conversion, reference impl, fixtures
```

Dependency direction is one way: `Gguf` → `Quantization` → `Numerics` →
`Model` → the top-level API. Nothing lower reaches up.

## Invariants worth knowing before changing anything

These are the things that are easy to "fix" into being wrong. Each has a test;
the test names are given so you can see what would catch you.

**RoPE pairs adjacent elements.** `(x[2i], x[2i+1])`, ggml's default mode — not
Hugging Face's `(x[i], x[i+d/2])`. The Mistral-format checkpoint is stored for
the former. Introducing a permutation in the converter, or `rotate_half` in the
model, produces a model that loads and runs and is completely wrong.
→ `RopeAndCacheTests.RotationPairsAdjacentElements`

**YaRN's magnitude scale is exactly 1.0 here.** `params.json` says
`"apply_scale": false`, which llama.cpp encodes as `mscale == mscale_all_dim`, so
the two terms cancel. llama.cpp splits this into an "attention factor" and a
fixed `1 + 0.1·ln(factor)` inside its RoPE kernel; `Rope` computes the product
directly instead. Getting it wrong scales every q and k by ~1.277.
→ `RopeAndCacheTests.MagnitudeScaleIsOneWhenApplyScaleIsOff`

**Prefill and decode share one frequency table.** If they ever diverge, a newly
generated token is rotated differently from the prompt already in the KV cache
and attention degrades silently. `Rope` exists to make that impossible.

**The attention context is `heads × head_dim`, not `hidden_size`.** For
Shieldstral those are 4096 and 3072; `attn_output` projects one to the other.
They coincide in most Llama-family models, which is why assuming they are equal
is a natural mistake and a confusing one to debug.

**The LM head is tied to the embedding table.** `output.weight` is legitimately
absent. It is evaluated only at the last position — 131072 rows is a third of a
forward pass and every other position's logits are discarded. When scoring, it
is evaluated only for the yes/no rows (`QuantMatMul.ForwardRowsAsync`): the
verdict reads two logits, and the full head streams a quarter of a gigabyte of
weights to produce the other 131070. The subset must be bit-identical to the
same entries of the full head, not merely close.
→ `SystemPromptCacheTests.ScoringReadsExactlyWhatTheFullHeadWouldHave`,
`MatMulTilingTests.ARowSubsetIsBitIdenticalToTheFullProduct`

**Control tokens are matched literally, before any BPE.** `[SYSTEM_PROMPT]` and
friends must land on their own ids. The `_specialTokens` list is sorted
longest-first so `[/SYSTEM_PROMPT]` is not shadowed by a prefix of itself.

**The prompt is bytes the model was trained on.** `ChatTemplate` reproduces the
published `chat_template.jinja` for the roles Shieldstral accepts. An extra
space is not cosmetic.
→ `ChatTemplateTests`

**The prefix cache must be a pure optimisation.** Cached and uncached paths
produce bit-identical logits, and the tests assert that over all 131072 values
rather than over the score. The two runs put the same token at different
positions in the batch, so this is also why every register-blocked kernel must
compute an output with exactly the arithmetic of its tail kernel: a 4-token tile
and the 1-token remainder cannot sum in different orders.
→ `SystemPromptCacheTests.CachedAndUncachedProduceIdenticalLogits`,
`MatMulTilingTests.ATokensResultDoesNotDependOnTheBatchItIsIn`

**A file at the download destination is always complete.** Everything else treats
a path that exists as a file worth memory-mapping, so `ModelDownloader` writes to
a `.download` sidecar and renames only at the end. Its resume path is the part
that needs care: a partial file is reused only when the recorded size *and* entity
tag still match the server, because resuming into a republished model yields a
GGUF of exactly the right length that is wrong from the resume point on — which no
loader can detect. models.curiosity.ai answers HEAD with 405, so the size, the
tag and range support all come from a one-byte ranged GET; drop that fallback and
nothing fails, downloads just silently stop resuming.
→ `ModelDownloaderTests`, against a loopback socket rather than the real host

**Softmax subtracts the maximum; `TensorPrimitives.SoftMax` does not.** That one
evaluates `exp(x) / Σexp(x)` directly. Attention scores here reach the 90s in the
deepest layers, `exp` overflows float32, and the division returns NaN — for the
positions that mattered most, in layer 24 of 26, on prompts long enough to have
sharp attention. It survives short prompts and shallow layers, which is precisely
what makes it dangerous. `Kernels.Softmax` does the max-shift; do not "simplify"
it back.
→ `KernelTests.SoftmaxSurvivesLargeLogits`

## Testing

```bash
dotnet test                                          # no weights needed
SHIELDSTRAL_MODEL=/path/to/model.gguf dotnet test    # + model-backed parity
```

Tests that change `QuantMatMul.Strategy` must join
`[Collection(MatMulStrategyCollection.Name)]` *and* restore it. The strategy is
process-wide and xunit runs test classes in parallel, so restoring alone is not
enough — one class pinning it to Float while another pins it to Integer makes both
measure whatever the scheduler left behind, and it fails intermittently, which is
worse than failing. The collection disables parallelism between them.

`SHIELDSTRAL_MODEL` may be a `.gguf` or a directory to search. Model-backed tests
write a line explaining the skip and pass when it is unset — keep that pattern
for new ones, so a checkout without a 3.4 GiB download stays green.

Comparisons use `Numeric.Close` (relative error), never xunit's decimal-places
overload: that one compares rounded strings, so two float32 values a single ULP
apart can straddle a boundary and "fail" at four decimal places while agreeing to
seven significant figures.

## Benchmarking

```bash
dotnet run --project src/LlmShield.Shieldstral.Cli -c Release -- \
  bench /path/to/models --json benchmark.json
```

One process does everything: environment, per-type dequantize throughput, the
float-versus-integer matmul comparison, and an end-to-end sweep over every GGUF
it finds, once per strategy. That is not a convenience — comparing a number
measured now against one measured in a separate invocation compares two machine
states as much as two kernels, and on a shared VM the machine state moves.

Two habits keep the micro-numbers honest: each timed sample repeats the work
until it spans at least 250 ms (a one-token matmul takes a few milliseconds, and
timing that directly measures the scheduler), and the reported figure is the
*fastest* sample, since everything that makes a run slower is noise.

## Regenerating fixtures

`tests/fixtures/*.json` are committed oracles. They are generated, not
hand-written; regenerate rather than edit.

```bash
pip install gguf numpy regex

# the source weights, if you do not already have them (~7.2 GiB, not the 15 GB
# the full repository would be — see the script's header for what it skips)
python3 tools/download_shieldstral.py models/shieldstral

# i-quant codebooks -> Quantization/QuantGrids.g.cs
python3 tools/gen_quant_grids.py

# per-type dequantization oracle
python3 tools/gen_quant_fixtures.py

# tokenizer.json, scores.json, activations.json (needs the original weights)
python3 tools/reference_shieldstral.py fixtures /path/to/models/shieldstral
```

`tools/reference_shieldstral.py` is the NumPy reference: an independent
implementation of the architecture reading the original bfloat16 weights. It
deliberately shares no code with the runtime — including its tokenizer, which
walks tekken's ranked vocabulary the way `mistral-common` does rather than
applying the derived merge table. Keep it that way; the value of the fixtures is
entirely in that independence.

`activations.json` records the last row and the sum of squares of each tensor for
the first few layers. Full rows for 26 layers would be tens of megabytes, and the
last row is the one every downstream value depends on.

## Parallelism

Every loop that spreads across cores — the row chunks in `QuantMatMul` and the
attention work items in `MinistralModel` — takes its fan-out from a `ParallelOptions`
threaded down from the caller, so a host bounds the whole runtime with one
setting. That is why the forward path is asynchronous: `Parallel.ForAsync` is
what carries the options, and `await` cannot appear in an `unsafe` context, which
is in turn why `QuantMatMul` is no longer an `unsafe` class — only the individual
pointer-using helpers are, and they are called from inside the loop body rather
than wrapping it.

Two consequences worth remembering before "simplifying" any of it back:

- **A `fixed` region cannot span an `await`.** The parallel regions therefore pass
  `Memory<T>` (which a lambda *can* capture) and take their spans inside the body,
  or pin with a `GCHandle` where a raw pointer is genuinely needed — see
  `PinnedMatMul` in the tests and `Benchmark.MatMulWork`.
- **Scratch buffers come from `ArrayPool`, not from a per-head field.** The
  attention scores buffer used to be a `float[][]` on the model, one slot per head.
  That only works while the fan-out is fixed at construction; renting per iteration
  costs nothing measurable and stops the model pinning a buffer per head forever.

The result must not depend on the fan-out — row chunks write to disjoint slices,
so no accumulation can be reordered.
→ `KernelTests.MaxDegreeOfParallelismDoesNotChangeTheResult`

## Performance notes

The hot loop is `QuantMatMul.ForwardAsync`, and it is a GEMM, not a dot product
per output. Weights dominate both the memory traffic and the unpack cost, so
each weight row is unpacked once per call into a cache-resident scratch buffer,
and then a register-blocked kernel sweeps every token past it. A dot per
(row, token) streams both operands for every single output; it is load-bound at
a small fraction of the machine and was the original bottleneck (~70 GFLOP/s on
4 cores against ~700 now). Changing the loop order will usually make it slower;
measure with `shieldstral bench`.

There are two arithmetic paths, selected by `QuantMatMul.Strategy`:

- **Float** decodes two weight rows to float32 and dots them against four tokens
  at a time (`Dot2x4`): eight accumulators fed by six loads. Works for every one
  of the thirty-odd types. `TokenTileBytes` bounds the activation slab so it
  stays in L2 while every row pair streams past it.
- **Integer** quantizes the activations to 8 bits and multiplies in 8 bits. Only
  for types with a single scale (plus optional offset) per 32-weight block —
  `IntegerDot.Supports` is the predicate. The k-quants and i-quants carry
  per-sub-block scales and stay on the float path.

`Auto` picks integer where there is a kernel. On x86 with AVX2+FMA that kernel is
`IntegerGemm`; the details that make it fast are all load-bearing:

1. **Eight rows interleaved, four bytes at a time.** `PackGroup` unpacks eight
   rows and transposes them so lane `j` of a 256-bit vector holds four
   consecutive weights of row `j`. Broadcasting four activation bytes against it
   makes one `vpdpbusd` (`AvxVnni`) produce partial sums for eight rows, already
   in their lanes. There is no horizontal reduction anywhere — the accumulator
   *is* the eight outputs — and each block's scale is one vector multiply for
   eight rows rather than eight scalar ones.
2. **Four tokens per packed vector.** `Multiply4` reuses each weight load across
   a tile of four tokens; `Multiply1` is the tail and does per lane exactly what
   `Multiply4` does.
3. **Unsigned weights, zero point in the integer domain.** `vpdpbusd` and
   `vpmaddubsw` want an unsigned first operand, so weights unpack to `w + z`
   (z = 8, 16, 128 for Q4_0, Q5_0, Q8_0) and each block's accumulator starts at
   `−z·Σa`, which is exact. Q4_1/Q5_1 are unsigned already and add `m·d_a·Σa`.
   Without VNNI the fallback is `vpmaddubsw` + `vpmaddwd`, whose int16 pair sums
   hold 31·127·2 but not 255·127·2 — so Q8_0 needs VNNI here and otherwise takes
   the `IntegerDot` path.
4. **Packing is per weight, so it is specialised.** At a few tokens it is most of
   the matmul. `PackGroup<TBlock>` hoists the type dispatch out of the block loop
   and converts the eight rows' f16 scales in one vector (`HalfToSingle`, exact
   for every bit pattern) — together a 3.4× cut, and 2× on a one-token matmul.

`IntegerDot` is the portable fallback (and the Q8_0 path without VNNI): unpack
once per row, one `Vector256<float>` accumulator per dot, `vpmaddubsw` with the
`|w| · sign(w)·a` trick. It is what the GEMM replaced, and it is kept because it
runs on anything AVX2 or wider without VNNI.

Activation scales stay float32 in the GEMM (`IntegerGemm.QuantizeActivations`)
rather than being rounded to f16 as a stored Q8_0 block would be. The cost of the
integer path is accuracy: about 3e-3 of relative L2 error per matmul, well inside
the tolerance the model-level tests use. `IntegerDotTests` bounds both the noise
and — separately and much more tightly — any systematic bias, since an unpacking
error shows up as a shift rather than as noise.

Attention is parallel over (KV head × run of query tokens), and evaluates the
four query heads that share a KV head together (`Kernels.Dot4`,
`Kernels.AddScaled4`), so each cached key and value row is loaded once per group
rather than once per head.

## Releasing

`.devops/azure-pipelines.yml` builds `main`, runs the tests and pushes
`LlmShield.Shieldstral` to nuget.org through the `nuget-curiosity-org` service
connection. Versions are CalVer — `yy.M.<buildId mod 65536>`, stamped by
`/p:Version` at build time, the modulo because the build counter has to fit an
int16. `Directory.Build.props` keeps `IsPackable` false, so a new project is not
published by accident: opt in on the project *and* add a build step, or it will
never leave the agent.

## Conventions

- `unsafe` and pointers are fine in `Gguf`, `Quantization` and `Numerics`; the
  model layer should stay in spans where it can. Put `unsafe` on the member, not
  the class, so the file can still contain an `await`.
- Buffers come from `ArrayPool<float>.Shared`. Return them in `finally`.
- A `Span<T>` cannot be captured by a lambda, and cannot live across an `await` —
  the parallel regions hold `Memory<T>` and take `.Span` inside the body.
- Public API is documented with `///`; explain *why*, not what the code says.
- Errors name the file and what to do: "re-download this file", not "bad header".
