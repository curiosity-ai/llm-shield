# todo — Shieldstral runtime for llm-shield

Working notes for the port. Checked items are done and covered by tests; the
"verified by" column names what would catch a regression.

## 1. Project scaffold

- [x] `net10.0` library `LlmShield.Shieldstral`, no native dependencies
- [x] xunit test project, fixtures copied to the test output
- [x] `shieldstral` CLI (`moderate`, `inspect`, `tokenize`, `dump`, `bench`)
- [x] `LlmShield.slnx`, `Directory.Build.props` (server GC, tiered PGO, unsafe on)
- [x] TensorSharp's BSD-3-Clause licence carried in `third-party/TensorSharp-LICENSE`,
      and every file derived from it says so in its header

## 2. GGUF

- [x] Memory-mapped reader — metadata, tensor table, all GGUF value types
- [x] Truncated-file detection up front (a partial download otherwise fails as an
      access violation inside a dequantizer)
- [x] Block geometry table for every `ggml_type` — verified by `GgufReaderTests`
- [x] Weights read straight out of the mapping; nothing is copied at load

## 3. Quantization — every type GGUF can carry

Reading is complete. Writing is the converter's job and covers what the reference
`gguf` package can write (see §5).

| family | types | dequantize | verified by |
|---|---|---|---|
| float | F32, F16, BF16, F64 | [x] | `DequantizerParityTests` |
| integer | I8, I16, I32, I64 | [x] | `DequantizerParityTests.DequantizeIntegerTypes` |
| legacy | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | [x] | parity + synthetic blocks |
| k-quant | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | [x] | parity + synthetic blocks |
| i-quant | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | [x] | parity + synthetic blocks |
| ternary | TQ1_0, TQ2_0 | [x] | parity + synthetic blocks |
| microscaling | MXFP4 | [x] | parity + synthetic blocks |

- [x] i-quant codebooks generated from the reference package rather than
      transcribed by hand (`tools/gen_quant_grids.py`)
- [x] Fixtures cover both real quantizer output and random bytes, so sign masks
      and codebook indices no realistic weight would produce are still exercised
- [ ] Optional: k-quant/i-quant *writing*, so the repo can quantize without the
      Python `gguf` package. Not needed to run the model; reading is what matters.

## 4. Tokenizer

- [x] Tekken pre-tokenizer regex (`\p{N}` singly, punctuation runs absorbing `/`)
- [x] GPT-2 byte alphabet encode/decode
- [x] Control markers matched literally, longest-first, before any merging
- [x] Priority-queue BPE with stale-candidate rejection
- [x] Token-identical to `mistral-common`-style segmentation on the reference
      prompts and on pre-tokenizer edge cases — verified by `TokenizerParityTests`

## 5. Conversion

- [x] `tools/convert_shieldstral_to_gguf.py`: Mistral format → GGUF, no
      permutation of wq/wk (the checkpoint is already in ggml's RoPE layout)
- [x] Tekken vocabulary → GGUF tokens/types + derived merge table
- [x] YaRN and llama-4 metadata written under llama.cpp's key names
- [x] Pixtral vision tower → `mmproj` GGUF (`--vision`)
- [x] Output types: f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, tq1_0, tq2_0, mxfp4

## 6. Model

- [x] RMSNorm (float64 reduction), SwiGLU, softmax, grouped-query attention
- [x] YaRN RoPE — one frequency table shared by prefill and decode
- [x] `mscale`/`mscale_all_dim` cancelling to 1.0 for `"apply_scale": false`
- [x] Llama-4 attention temperature
- [x] Tied LM head, evaluated only at the last position
- [x] Grow-on-demand KV cache
- [x] Layer-by-layer parity against the NumPy reference — `ActivationParityTests`
- [x] Scores match NumPy *and* llama.cpp on all six reference cases —
      `ModerationScoreTests`

## 7. Fixed-system-prompt cache

- [x] `SystemPromptCache`: capture, restore, save, load
- [x] Fingerprint over (model file identity, cache geometry, prefix tokens), so a
      stale snapshot is refused rather than silently applied
- [x] On by default in `ShieldstralModerator`; optional on-disk persistence
- [x] Cached and uncached paths produce *bit-identical* logits —
      `SystemPromptCacheTests`

## 8. Performance

- [x] Weight-stationary GEMM: each weight row decoded once per call, token-tiled
      so the activations stay in cache
- [x] `TensorPrimitives` for dot/sigmoid/softmax/elementwise; `Vector<T>` for the
      fused shapes it has no single call for
- [x] Attention parallel over heads, GEMM parallel over row blocks
- [ ] Integer dot products against Q8-quantized activations. Would cut the
      dequantize-then-FMA cost roughly threefold on the legacy and k-quant types;
      the current path is simpler and covers every type uniformly.
- [ ] Persist the prefix cache next to the model by default rather than opt-in.

## 9. Vision

- [x] Converter emits the Pixtral tower and projector
- [x] `[IMG]` placement in the prompt template — `ChatTemplateTests`
- [ ] Pixtral encoder forward pass (patch conv, 2-D RoPE, patch merger, projector)
- [ ] Image preprocessing (resize to longest edge 1540, CLIP normalisation)
- [ ] Expanding `[IMG]` into patch tokens and injecting the embeddings

Text moderation is complete and validated; the vision path is converted but not
yet executed. `ShieldstralModerator` is text-only today.

## 10. Documentation

- [x] `README.md` — what it is, how to convert, how to run
- [x] `CLAUDE.md` — layout, invariants, how to regenerate fixtures
- [x] This file
