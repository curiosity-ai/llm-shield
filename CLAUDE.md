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
  Quantization/    Dequantizer.cs (every GGML type), QuantGrids.g.cs (generated)
  Numerics/        Kernels.cs, QuantMatMul.cs, WeightMatrix.cs
  Tokenization/    TekkenTokenizer.cs
  Model/           ModelConfig.cs, Rope.cs, KvCache.cs, MinistralModel.cs
  ChatTemplate.cs, SystemPromptCache.cs, ShieldstralModerator.cs
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
forward pass and every other position's logits are discarded.

**Control tokens are matched literally, before any BPE.** `[SYSTEM_PROMPT]` and
friends must land on their own ids. The `_specialTokens` list is sorted
longest-first so `[/SYSTEM_PROMPT]` is not shadowed by a prefix of itself.

**The prompt is bytes the model was trained on.** `ChatTemplate` reproduces the
published `chat_template.jinja` for the roles Shieldstral accepts. An extra
space is not cosmetic.
→ `ChatTemplateTests`

**The prefix cache must be a pure optimisation.** Cached and uncached paths
produce bit-identical logits, and the tests assert that over all 131072 values
rather than over the score.
→ `SystemPromptCacheTests.CachedAndUncachedProduceIdenticalLogits`

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

`SHIELDSTRAL_MODEL` may be a `.gguf` or a directory to search. Model-backed tests
write a line explaining the skip and pass when it is unset — keep that pattern
for new ones, so a checkout without a 3.4 GiB download stays green.

Comparisons use `Numeric.Close` (relative error), never xunit's decimal-places
overload: that one compares rounded strings, so two float32 values a single ULP
apart can straddle a boundary and "fail" at four decimal places while agreeing to
seven significant figures.

## Regenerating fixtures

`tests/fixtures/*.json` are committed oracles. They are generated, not
hand-written; regenerate rather than edit.

```bash
pip install gguf numpy regex

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

## Performance notes

The hot loop is `QuantMatMul.Forward`. Its shape is deliberate: weights dominate
both the memory traffic and the decode cost, so each weight row is touched once
per call and decoded into an L1-resident scratch buffer while every token in the
current tile dots against it. Tiling the tokens (`TokenTileBytes`) is what keeps
a long prompt from re-streaming the activations once per output row. Changing the
loop order will usually make it slower; measure with `shieldstral bench`.

The obvious remaining win is integer dot products against Q8-quantized
activations, roughly a threefold cut on the legacy and k-quant types. It is not
done because the current path handles all thirty-odd types uniformly, and
correctness across every type was the priority.

## Conventions

- `unsafe` and pointers are fine in `Gguf`, `Quantization` and `Numerics`; the
  model layer should stay in spans where it can.
- Buffers come from `ArrayPool<float>.Shared`. Return them in `finally`.
- A `Span<T>` cannot be captured by a lambda — the parallel regions in
  `MinistralModel` use `fixed` pointers for exactly that reason.
- Public API is documented with `///`; explain *why*, not what the code says.
- Errors name the file and what to do: "re-download this file", not "bad header".
