# llm-shield — Shieldstral runtime

A self-contained .NET 10 runtime for [Mistral's Shieldstral 1.0
3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B), a policy-adaptive
safety classifier. It reads GGUF weights directly and needs no native library, no
GPU, and no Python at runtime.

Shieldstral does not predict fixed moderation categories. You give it a policy in
plain language, a yes/no question and the content, and it answers with a single
token — so one forward pass, no sampling loop, and the whole verdict lives in the
first token's distribution.

```csharp
using var moderator = new ShieldstralModerator("Shieldstral-1.0-3B-Q8_0.gguf");

ModerationResult result = moderator.Moderate(
    instruct: "You are a strict safety moderator reviewing potentially harmful content. " +
              "Apply a low tolerance threshold.",
    query:    "Does this content promote physical violence?",
    document: "[User] How can I hurt someone without being caught?");

Console.WriteLine(result);            // UNSAFE score=0.997307 (yes=32.7513 no=26.8369)
Console.WriteLine(result.IsUnsafe);   // True
```

`Score` is the probability of a violation in [0, 1]: the softmax of the "yes" and
"no" logits against each other, with the rest of the vocabulary renormalised
away, exactly as the model card's reference scorer does. The 0.5 threshold is a
default, not a law — `IsUnsafeAt(threshold)` lets you move it.

## What is here

| project | what it does |
|---|---|
| `src/LlmShield.Shieldstral` | the runtime: GGUF, quantization, tokenizer, model, moderator |
| `src/LlmShield.Shieldstral.Cli` | `shieldstral` — moderate, inspect, tokenize, dump, bench |
| `tests/LlmShield.Shieldstral.Tests` | parity and unit tests |
| `tools/` | Python: weight conversion, the reference implementation, fixture generation |

The numerics are derived from
[TensorSharp](https://github.com/zhongkaifu/TensorSharp) (BSD-3-Clause, see
`third-party/TensorSharp-LICENSE`), reduced to what this one model needs and
rewritten around .NET 10's `TensorPrimitives` and `Vector<T>`. Files carrying
derived logic say so in their header.

## Getting a model

The runtime reads GGUF. Convert Mistral's official release once:

```bash
pip install gguf numpy

# ~7.7 GB; consolidated.safetensors + params.json + tekken.json + chat_template.jinja
huggingface-cli download mistralai/Shieldstral-1.0-3B --local-dir models/shieldstral

python3 tools/convert_shieldstral_to_gguf.py models/shieldstral --outtype q8_0 --vision
```

That writes `Shieldstral-1.0-3B-Q8_0.gguf` (3.4 GiB) and, with `--vision`, the
Pixtral tower as a separate `mmproj` file. `--outtype` accepts `f32`, `f16`,
`bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `tq1_0`, `tq2_0` and `mxfp4`.

A GGUF produced by `llama.cpp`'s `convert_hf_to_gguf.py --mistral-format` loads
too — the metadata keys and tensor names are the same.

> The converter reads the **Mistral-format** `consolidated.safetensors`, not the
> Hugging Face `model.safetensors`. Hugging Face's Llama-style attention rotates
> `(x[i], x[i+d/2])` and its converter permutes `wq`/`wk` to compensate; the
> Mistral checkpoint is already in the `(x[2i], x[2i+1])` layout ggml's default
> RoPE mode expects. Converting from the wrong one produces a model that loads,
> runs, and is quietly wrong.

## Command line

```bash
dotnet run --project src/LlmShield.Shieldstral.Cli -c Release -- \
  moderate Shieldstral-1.0-3B-Q8_0.gguf \
  --instruct "You are a strict safety moderator. Apply a low tolerance threshold." \
  --query    "Does this content promote physical violence?" \
  --document "[User] How can I hurt someone without being caught?"
```

`--json` emits a machine-readable result; `--document-file` or stdin takes the
content from a file. `inspect` prints a GGUF's metadata and tensor breakdown,
`tokenize` shows token ids and pieces, `dump` records per-layer activations for
the parity harness, and `bench` runs the benchmark suite.

## Quantization support

Every type GGUF can carry is readable, including the i-quant codebook families
and the ternary types:

```
F32 F16 BF16 F64  I8 I16 I32 I64
Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1
Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K
IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS
TQ1_0 TQ2_0 MXFP4
```

Each is checked against the reference `gguf` Python package on both real
quantizer output and random bytes — the latter reaches sign masks and codebook
indices that realistic weights never produce, which is where an unpacking bug
actually hides.

Weights stay quantized in the memory-mapped file and are decoded one row at a
time inside the matmul, so a 3.4 GiB checkpoint loads instantly and costs its
on-disk size in shared page cache rather than that plus a float32 copy.

## The fixed-system-prompt cache

Shieldstral is always driven with the same system message. Attention is causal,
so the keys and values at those leading positions depend on nothing downstream:
prefilling them is identical work on every request. `ShieldstralModerator`
computes that state once and restores it per request, leaving only the
instruct/query/document suffix to run.

It is on by default and can be persisted so even a fresh process skips it:

```csharp
using var moderator = new ShieldstralModerator(
    "Shieldstral-1.0-3B-Q8_0.gguf",
    prefixCachePath: "shieldstral-prefix.bin");   // ~7 MiB
```

The snapshot is fingerprinted against the model file, the cache geometry and the
exact prefix tokens; a mismatch is refused rather than silently applied. Results
are bit-identical either way, which the test suite asserts by comparing all
131072 logits, not just the score.

On a 4-core sandbox VM with Q8_0 and ~85-token prompts, of which 34 are the
system prompt, this cut 11.6 s/request to 7.1 s (one-time 5.6 s setup, or zero if
persisted). The saving tracks the cached fraction of the prompt almost exactly,
which is what you would expect for a prefill-bound workload — the shorter the
caller's content, the larger the share the fixed prompt was costing.

## Validation

Three independent implementations agree on the same six reference cases:

| case | this runtime (Q8_0) | NumPy reference (bf16) | llama.cpp b8.18.1 (Q8_0) |
|---|---|---|---|
| violent-request | 0.997307 | 0.997249 | 0.997343 |
| benign-cooking | 0.000000 | 0.000000 | 0.000000 |
| borderline-sarcasm | 0.443099 | 0.428835 | 0.439276 |
| borderline-fiction | 0.050686 | 0.051092 | 0.049938 |
| borderline-lenient | 0.000943 | 0.000935 | 0.000885 |
| borderline-selfharm | 0.025396 | 0.025491 | 0.025094 |

Worst disagreement with llama.cpp on the same quantization: 3.8e-3. Worst against
the unquantized NumPy run: 1.4e-2, on the most borderline case — which is where
quantization should show, and it is the only place it does.

`tools/reference_shieldstral.py` is a NumPy implementation that reads the
original bfloat16 weights and shares no code with the runtime, so agreement is
evidence about the port rather than about a shared bug. The llama.cpp column is a
third, native implementation.

Beyond the end-to-end scores the suite compares **per-layer activations** — every
norm, the rotated queries and keys, the attention context, each residual stage —
so a regression names the step that broke rather than just the answer.

```bash
dotnet test                                        # everything that needs no weights
SHIELDSTRAL_MODEL=/path/to/model.gguf dotnet test   # plus the model-backed parity tests
```

Without `SHIELDSTRAL_MODEL` the model-backed tests report why they skipped and
pass, so a checkout without a 3.4 GiB download still runs the rest.

## Performance

Two arithmetic paths, chosen per weight type by `QuantMatMul.Strategy`:

- **Float** decodes each weight to float32 and uses `TensorPrimitives.Dot`. Works
  for every type.
- **Integer** quantizes the activations to Q8_0 and multiplies in 8 bits, using
  AVX2's `vpmaddubsw`/`vpmaddwd` where available. Only for types with a single
  scale (plus optional offset) per 32-weight block: Q8_0, Q5_1, Q5_0, Q4_1, Q4_0.

`Auto` — the default — takes the integer path where there is a kernel and falls
back to float otherwise.

### Quantization sweep

One `shieldstral bench` execution, 4-core sandbox VM (no AVX-512), .NET 10,
256-token prefill, 8 decode tokens, median of 3. `score` is the safety verdict
for the `violent-request` reference case, where llama.cpp gives **0.997343**.

| build | size | prefill tok/s | decode tok/s | RSS | score |
|---|---|---|---|---|---|
| | | float → **auto** | float → **auto** | | float → **auto** |
| Q8_0 | 3.40 GiB | 7.1 → **7.6** | 0.98 → **3.16** | 3.8 GiB | 0.997307 → **0.997415** |
| Q5_1 | 2.40 GiB | 7.0 → **8.1** | 0.63 → **0.88** | 2.7 GiB | 0.997395 → **0.997463** |
| Q5_0 | 2.20 GiB | 6.9 → **10.5** | 0.66 → **0.76** | 2.6 GiB | 0.996971 → **0.997151** |
| Q4_1 | 2.00 GiB | 7.2 → **8.0** | 0.81 → **1.05** | 2.3 GiB | 0.996204 → **0.996366** |
| Q4_0 | 1.80 GiB | 7.0 → **10.5** | 0.83 → **1.17** | 2.1 GiB | 0.998687 → **0.998754** |
| MXFP4 | 1.70 GiB | 7.1 → 7.0 | 0.92 → 0.92 | 2.0 GiB | 0.994321 |
| TQ2_0 | 0.83 GiB | 7.2 → 6.9 | 0.67 → 0.66 | 1.2 GiB | **0.030626** |

Managed allocation is ~220 MiB per run regardless of build: the weights are
memory-mapped, so RSS tracks the file size and is page cache the kernel can
reclaim, not private memory the process is holding.

Three things worth reading off that table:

- **The integer path is worth keeping.** 1.07–1.52× on prefill and 1.15–3.22× on
  decode for the five types it covers, and the two it does not are unchanged —
  they fall through to float, as intended. The verdict moves by at most 2e-4,
  which is an order of magnitude below the difference between two quantizations
  of the same weights.
- **Under the float path, throughput barely depends on model size.** Every build
  prefills at ~7 tok/s whether it is 0.83 GiB or 3.40 GiB, because the cost is
  decoding weights to float, not fetching them. That is exactly the bound the
  integer path removes, and it is why Q8_0 — the *largest* build, but the cheapest
  to decode — is the fastest at decode once it stops decoding at all (3.16 tok/s,
  from a `DotPackedQ8_0` kernel that reads the packed row in place).
- **TQ2_0 is a size record and a broken classifier.** At 0.83 GiB it scores 0.031
  where every other build says 0.997: ternary quantization is meant for models
  trained for it, and applying it post-hoc destroys this one. It is in the table
  because "it loads and runs" is not the same as "it works", and a sweep that
  only reported tok/s would have hidden that.

### Per-type decode throughput

What the float path pays per weight, single-threaded (`Melem/s` of output):

| | | | | |
|---|---|---|---|---|
| F32 3882 | Q8_0 1520 | BF16 1475 | Q4_0 969 | Q4_K 713 |
| IQ1_S 741 | TQ2_0 744 | Q5_0 743 | Q6_K 354 | F16 386 |
| IQ2_XXS 192 | IQ2_S 172 | IQ3_XXS 158 | IQ3_S 126 | |

The i-quants are 5–10× slower to decode than the legacy families, which is the
cost of their codebook indirection. They are supported for completeness; if you
want small *and* fast, Q4_0 with the integer path is the better trade.

Full machine-readable results are in [`benchmark.json`](benchmark.json).

## Requirements

.NET 10 SDK. No native dependencies, no GPU. Python 3.10+ with `gguf`, `numpy`
and `regex` only for conversion and for regenerating test fixtures.

Runs comfortably in about 6 GiB of RAM for a Q8_0 model: the weights are shared
page cache, the KV cache grows on demand, and activations are pooled.

## Scope

Text moderation is complete and validated. The converter emits the Pixtral vision
tower and the prompt template places `[IMG]` markers, but the encoder's forward
pass is not implemented yet — `ShieldstralModerator` is text-only today. See
`todo.md`.

## Licence

BSD-3-Clause. Portions derived from TensorSharp, © Zhongkai Fu, also
BSD-3-Clause; see `third-party/TensorSharp-LICENSE`. Model weights are Mistral's
and carry their own licence.
