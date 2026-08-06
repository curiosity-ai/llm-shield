#!/usr/bin/env python3
"""Generate tests/fixtures/quantization.json — the dequantizer oracle.

For every GGML type the runtime claims to read, this writes a run of blocks and
the float32 the reference `gguf` package decodes them to. Two kinds of block:

  quantized  real output of gguf's quantizer on pseudo-random weights, i.e. the
             bit patterns a converted checkpoint actually contains
  synthetic  pseudo-random *bytes* interpreted as blocks, which covers the whole
             encoding space including sign masks, codebook indices and scale
             fields no real quantizer would emit

The synthetic case is what makes this a strong check: a nibble swapped in an
i-quant sign mask is invisible on realistic weights and obvious here. Blocks
whose reference output is not finite (a random exponent byte can decode to inf)
are dropped rather than compared.

Usage:  python3 tools/gen_quant_fixtures.py [tests/fixtures/quantization.json]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T
from gguf.quants import dequantize, quantize, _type_traits

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "tests/fixtures/quantization.json"

# Every type LlmShield.Shieldstral.Quantization.Dequantizer.Supports() returns true
# for. F32/F16/BF16 and the integer types are handled by gguf directly.
TYPES = [
    T.F32, T.F16, T.BF16,
    T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0,
    T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K,
    T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S,
    T.IQ1_S, T.IQ1_M, T.IQ4_NL, T.IQ4_XS,
    T.TQ1_0, T.TQ2_0, T.MXFP4,
]

BLOCKS = 8


def synthetic_blocks(qtype: T, rng: np.random.Generator) -> tuple[bytes, list[float]] | None:
    """Random bytes read as blocks, keeping only the ones that decode finitely."""
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    kept_bytes: list[bytes] = []
    kept_values: list[np.ndarray] = []

    for _ in range(200):
        if len(kept_bytes) >= BLOCKS:
            break
        raw = rng.integers(0, 256, size=type_size, dtype=np.uint8)
        try:
            values = dequantize(raw.reshape(1, type_size), qtype).reshape(-1)
        except NotImplementedError:
            return None
        if not np.all(np.isfinite(values)):
            continue
        # Reject blocks whose magnitude would swamp a float32 comparison; they
        # test nothing the moderate-magnitude ones do not.
        if np.max(np.abs(values)) > 1e6:
            continue
        kept_bytes.append(raw.tobytes())
        kept_values.append(values.astype(np.float32))

    if not kept_bytes:
        return None
    return b"".join(kept_bytes), [float(x) for x in np.concatenate(kept_values)]


def quantized_blocks(qtype: T, rng: np.random.Generator) -> tuple[bytes, list[float]] | None:
    """What gguf's own quantizer emits for pseudo-random weights."""
    block_size, _ = GGML_QUANT_SIZES[qtype]
    source = rng.standard_normal(BLOCKS * block_size).astype(np.float32) * 0.35
    try:
        raw = quantize(source, qtype)
    except (NotImplementedError, KeyError):
        return None
    values = dequantize(raw, qtype).reshape(-1).astype(np.float32)
    if not np.all(np.isfinite(values)):
        return None
    return raw.tobytes(), [float(x) for x in values]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DEFAULT
    rng = np.random.default_rng(20260806)

    cases = []
    for qtype in TYPES:
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        entry: dict = {
            "type": qtype.name,
            "ggml_type": int(qtype),
            "block_size": block_size,
            "type_size": type_size,
        }

        if qtype in (T.F32, T.F16, T.BF16):
            source = rng.standard_normal(64).astype(np.float32)
            raw = quantize(source, qtype)
            entry["quantized"] = {
                "bytes": base64.b64encode(raw.tobytes()).decode(),
                "expected": [float(x) for x in dequantize(raw, qtype).reshape(-1)],
            }
        else:
            if (q := quantized_blocks(qtype, rng)) is not None:
                entry["quantized"] = {
                    "bytes": base64.b64encode(q[0]).decode(),
                    "expected": q[1],
                }
            if (s := synthetic_blocks(qtype, rng)) is not None:
                entry["synthetic"] = {
                    "bytes": base64.b64encode(s[0]).decode(),
                    "expected": s[1],
                }

        if "quantized" not in entry and "synthetic" not in entry:
            print(f"  skipping {qtype.name}: gguf can neither quantize nor dequantize it")
            continue
        cases.append(entry)
        kinds = "+".join(k for k in ("quantized", "synthetic") if k in entry)
        print(f"  {qtype.name:9s} block={block_size:3d} size={type_size:3d}  {kinds}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, indent=1))
    print(f"wrote {out} ({len(cases)} types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
