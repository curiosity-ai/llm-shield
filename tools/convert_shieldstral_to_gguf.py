#!/usr/bin/env python3
"""Convert mistralai/Shieldstral-1.0-3B from Mistral format to GGUF.

Input is the *Mistral*-format release (`consolidated.safetensors` + `params.json`
+ `tekken.json`), not the Hugging Face one. That choice matters: HF's Llama-style
attention rotates (x[i], x[i+d/2]) and its converter permutes wq/wk to compensate,
while the Mistral checkpoint is stored in the (x[2i], x[2i+1]) layout that ggml's
default RoPE mode expects. Converting from Mistral format means no permutation —
and applying one anyway is the classic way to get a model that loads, runs, and
quietly produces garbage.

Outputs, by default alongside the input:
  Shieldstral-1.0-3B-<TYPE>.gguf         the language model
  Shieldstral-1.0-3B-mmproj-<TYPE>.gguf  the Pixtral vision tower (--vision)

Usage:
  python3 tools/convert_shieldstral_to_gguf.py MODEL_DIR [--outtype q8_0] [--vision]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType, RopeScalingType
from gguf.quants import quantize

ARCH = "mistral3"
VISION_ARCH = "clip"

# gguf-py can only *write* these; the k-quants and i-quants are read-only there
# (and in this repo's C# writer). Reading every GGUF type is separately covered.
OUT_TYPES = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "bf16": GGMLQuantizationType.BF16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q4_0": GGMLQuantizationType.Q4_0,
    "tq2_0": GGMLQuantizationType.TQ2_0,
    "tq1_0": GGMLQuantizationType.TQ1_0,
    "mxfp4": GGMLQuantizationType.MXFP4,
}

# Tensors that stay float regardless of --outtype. Norm weights are one value per
# channel — quantizing them saves nothing and costs accuracy everywhere at once.
KEEP_F32_SUFFIXES = ("_norm.weight", "norm.weight")


# --------------------------------------------------------------- safetensors

class SafetensorsFile:
    """
    Minimal safetensors reader that yields float32.

    The `safetensors` numpy backend refuses BF16 (numpy has no such dtype), and
    Shieldstral ships entirely in BF16 — so widen it here by the definition:
    a bfloat16 is the top 16 bits of the float32 with the same value.
    """

    _DTYPES = {
        "F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "F16": np.dtype("<f2"),
        "I64": np.dtype("<i8"), "I32": np.dtype("<i4"), "I16": np.dtype("<i2"),
        "I8": np.dtype("i1"), "U8": np.dtype("u1"), "BOOL": np.dtype("?"),
    }

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            self.header = json.loads(f.read(header_len))
        self._data_start = 8 + header_len
        self._map = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def get_tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        start, end = meta["data_offsets"]
        raw = self._map[self._data_start + start : self._data_start + end]
        shape = tuple(meta["shape"])

        if meta["dtype"] == "BF16":
            bits = raw.view(np.uint16).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(shape)

        dtype = self._DTYPES.get(meta["dtype"])
        if dtype is None:
            raise TypeError(f"{name}: unsupported safetensors dtype {meta['dtype']}")
        return raw.view(dtype).reshape(shape).astype(np.float32, copy=False)


# --------------------------------------------------------------------- vocab

def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte alphabet: every byte to a printable code point."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENCODER = bytes_to_unicode()


def encode_token(raw: bytes) -> str:
    return "".join(_BYTE_ENCODER[b] for b in raw)


def bpe_split(ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    """Greedy BPE over `token`, refusing any merge at or above `max_rank`."""
    parts = [bytes([b]) for b in token]
    while True:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and (best_rank is None or rank < best_rank):
                best_i, best_rank = i, rank
        if best_rank is None or best_rank >= max_rank:
            return parts
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]


def build_vocab(tekken: dict) -> tuple[list[str], list[int], list[str]]:
    """
    Returns (tokens, token_types, merges) in llama.cpp's GGUF conventions.

    tekken.json numbers its 1000 control tokens 0..999 and its byte-level entries
    from 0 again, so a vocabulary id is `rank + num_special_tokens` for the latter.
    """
    specials = tekken["special_tokens"]
    vocab = tekken["vocab"]
    n_special = len(specials)
    size = n_special + len(vocab)

    tokens: list[str] = [""] * size
    types: list[int] = [1] * size          # LLAMA_TOKEN_TYPE_NORMAL

    for entry in specials:
        rank = entry["rank"]
        tokens[rank] = entry["token_str"]
        types[rank] = 3 if entry.get("is_control", True) else 4   # CONTROL / USER_DEFINED

    import base64

    raw_by_id: dict[int, bytes] = {}
    ranks: dict[bytes, int] = {}
    for entry in vocab:
        token_id = entry["rank"] + n_special
        raw = base64.b64decode(entry["token_bytes"])
        tokens[token_id] = encode_token(raw)
        raw_by_id[token_id] = raw
        # Merge priority follows tekken's own rank, which is what makes the derived
        # merge list reproduce tiktoken's segmentation exactly.
        ranks.setdefault(raw, entry["rank"])

    # Derive the merge table: a token merges from exactly the two pieces greedy BPE
    # leaves when every merge at or above its own rank is forbidden.
    merges: list[str] = []
    for entry in vocab:
        raw = base64.b64decode(entry["token_bytes"])
        if len(raw) < 2:
            continue
        parts = bpe_split(ranks, raw, entry["rank"])
        if len(parts) == 2:
            merges.append(f"{encode_token(parts[0])} {encode_token(parts[1])}")

    return tokens, types, merges


# -------------------------------------------------------------------- tensors

MISTRAL_TO_GGUF = {
    "tok_embeddings.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
}

LAYER_MAP = {
    "attention_norm.weight": "attn_norm.weight",
    "attention.wq.weight": "attn_q.weight",
    "attention.wk.weight": "attn_k.weight",
    "attention.wv.weight": "attn_v.weight",
    "attention.wo.weight": "attn_output.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "feed_forward.w1.weight": "ffn_gate.weight",
    "feed_forward.w2.weight": "ffn_down.weight",
    "feed_forward.w3.weight": "ffn_up.weight",
}

VISION_LAYER_MAP = {
    "attention_norm.weight": "ln1.weight",
    "attention.wq.weight": "attn_q.weight",
    "attention.wk.weight": "attn_k.weight",
    "attention.wv.weight": "attn_v.weight",
    "attention.wo.weight": "attn_out.weight",
    "ffn_norm.weight": "ln2.weight",
    "feed_forward.w1.weight": "ffn_gate.weight",
    "feed_forward.w2.weight": "ffn_down.weight",
    "feed_forward.w3.weight": "ffn_up.weight",
}


def map_text_name(name: str) -> str | None:
    if name in MISTRAL_TO_GGUF:
        return MISTRAL_TO_GGUF[name]
    if name.startswith("layers."):
        _, index, rest = name.split(".", 2)
        if rest in LAYER_MAP:
            return f"blk.{index}.{LAYER_MAP[rest]}"
        raise KeyError(f"unmapped layer tensor: {name}")
    return None   # vision / projector, handled separately


def map_vision_name(name: str) -> str | None:
    if name == "vision_encoder.patch_conv.weight":
        return "v.patch_embd.weight"
    if name == "vision_encoder.ln_pre.weight":
        return "v.pre_ln.weight"
    if name.startswith("vision_encoder.transformer.layers."):
        rest = name[len("vision_encoder.transformer.layers.") :]
        index, tail = rest.split(".", 1)
        if tail in VISION_LAYER_MAP:
            return f"v.blk.{index}.{VISION_LAYER_MAP[tail]}"
        raise KeyError(f"unmapped vision tensor: {name}")
    # The multimodal projector: RMS norm, patch merger, then two linear layers.
    return {
        "pre_mm_projector_norm.weight": "mm.input_norm.weight",
        "patch_merger.merging_layer.weight": "mm.patch_merger.weight",
        "vision_language_adapter.w_in.weight": "mm.1.weight",
        "vision_language_adapter.w_out.weight": "mm.2.weight",
    }.get(name)


def pick_type(name: str, requested: GGMLQuantizationType, shape: tuple[int, ...]) -> GGMLQuantizationType:
    if requested in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        return requested
    if any(name.endswith(s) for s in KEEP_F32_SUFFIXES) or len(shape) != 2:
        return GGMLQuantizationType.F32
    # A quantized row must be a whole number of blocks.
    from gguf.constants import GGML_QUANT_SIZES

    block, _ = GGML_QUANT_SIZES[requested]
    return requested if shape[-1] % block == 0 else GGMLQuantizationType.F16


def add_tensor(writer: GGUFWriter, name: str, array: np.ndarray, qtype: GGMLQuantizationType) -> None:
    # The writer reverses the numpy shape on the way out, so a (out, in) weight
    # lands in the file as ne = [in, out] — one output feature per contiguous row,
    # which is what llama.cpp and this repo's reader both expect. For a quantized
    # type it derives the logical shape from the byte shape itself, so hand it the
    # packed array and nothing else.
    data = quantize(array.astype(np.float32, copy=False), qtype)
    writer.add_tensor(name, data, raw_dtype=qtype)


# ----------------------------------------------------------------- conversion

def convert_text_model(src: Path, out: Path, params: dict, tekken: dict,
                       qtype: GGMLQuantizationType, chat_template: str | None) -> None:
    writer = GGUFWriter(str(out), ARCH)

    n_layers = params["n_layers"]
    head_dim = params["head_dim"]
    yarn = params.get("yarn") or {}
    llama4 = params.get("llama_4_scaling") or {}

    writer.add_name("Shieldstral 1.0 3B")
    writer.add_description(
        "Mistral Shieldstral 1.0 3B — policy-adaptive safety classifier "
        "(Ministral-3 backbone), converted from the official Mistral-format release."
    )
    writer.add_file_type(qtype)

    writer.add_context_length(params["max_position_embeddings"])
    writer.add_embedding_length(params["dim"])
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(params["hidden_dim"])
    writer.add_head_count(params["n_heads"])
    writer.add_head_count_kv(params["n_kv_heads"])
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(params["norm_eps"])
    writer.add_vocab_size(params["vocab_size"])

    writer.add_rope_freq_base(params["rope_theta"])
    writer.add_rope_dimension_count(head_dim)

    if yarn:
        writer.add_rope_scaling_type(RopeScalingType.YARN)
        writer.add_rope_scaling_factor(float(yarn["factor"]))
        writer.add_rope_scaling_orig_ctx_len(int(yarn["original_max_position_embeddings"]))
        # params.json spells YaRN's beta/alpha; llama.cpp calls them beta_fast/beta_slow.
        writer.add_rope_scaling_yarn_beta_fast(float(yarn.get("beta", 32.0)))
        writer.add_rope_scaling_yarn_beta_slow(float(yarn.get("alpha", 1.0)))
        writer.add_rope_scaling_yarn_ext_factor(1.0)
        writer.add_key_value(f"{ARCH}.rope.scaling.mscale", 1.0, GGUFValueType.FLOAT32)
        # "apply_scale": false means the magnitude correction must cancel to 1.0,
        # which is what mscale_all_dim == mscale achieves. llama.cpp stores it
        # under the yarn_log_multiplier key.
        writer.add_rope_scaling_yarn_log_mul(0.0 if yarn.get("apply_scale", False) else 1.0)
        writer.add_rope_scaling_attn_factors(1.0)

    if llama4:
        writer.add_attn_temperature_scale(float(llama4["beta"]))
        writer.add_attn_temperature_length(int(llama4["original_max_position_embeddings"]))

    tokens, types, merges = build_vocab(tekken)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("tekken")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_pad_token_id(11)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    if chat_template:
        writer.add_chat_template(chat_template)

    tied = params.get("tied_embeddings", True)
    print(f"  tied embeddings: {tied} (no separate output.weight)" if tied else "  separate lm_head")

    f = SafetensorsFile(src)
    if True:
        names = sorted(f.keys())
        for name in names:
            target = map_text_name(name)
            if target is None:
                continue
            array = f.get_tensor(name)
            t = pick_type(target, qtype, array.shape)
            add_tensor(writer, target, array, t)
            print(f"  {name:52s} -> {target:28s} {array.shape} {t.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def convert_vision_model(src: Path, out: Path, params: dict, qtype: GGMLQuantizationType) -> None:
    vision = params["vision_encoder"]
    writer = GGUFWriter(str(out), VISION_ARCH)

    writer.add_name("Shieldstral 1.0 3B vision tower")
    writer.add_file_type(qtype)
    writer.add_key_value("clip.has_vision_encoder", True, GGUFValueType.BOOL)
    writer.add_key_value("clip.projector_type", "pixtral", GGUFValueType.STRING)
    writer.add_key_value("clip.vision.image_size", int(vision["image_size"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.patch_size", int(vision["patch_size"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.embedding_length", int(vision["hidden_size"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.feed_forward_length", int(vision["intermediate_size"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.block_count", int(vision["num_hidden_layers"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.attention.head_count", int(vision["num_attention_heads"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.attention.layer_norm_epsilon", float(params["norm_eps"]), GGUFValueType.FLOAT32)
    writer.add_key_value("clip.vision.rope.freq_base", float(vision["rope_theta"]), GGUFValueType.FLOAT32)
    writer.add_key_value("clip.vision.spatial_merge_size", int(vision["spatial_merge_size"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.projection_dim", int(params["dim"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.image_token_id", int(vision["image_token_id"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.image_break_token_id", int(vision["image_break_token_id"]), GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.image_end_token_id", int(vision["image_end_token_id"]), GGUFValueType.UINT32)

    f = SafetensorsFile(src)
    if True:
        for name in sorted(f.keys()):
            target = map_vision_name(name)
            if target is None:
                continue
            array = f.get_tensor(name)
            t = pick_type(target, qtype, array.shape)
            add_tensor(writer, target, array, t)
            print(f"  {name:52s} -> {target:28s} {array.shape} {t.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="directory holding consolidated.safetensors etc.")
    ap.add_argument("--outtype", default="q8_0", choices=sorted(OUT_TYPES),
                    help="quantization for 2-D weights (default: q8_0)")
    ap.add_argument("--outfile", type=Path, default=None)
    ap.add_argument("--vision", action="store_true", help="also write the Pixtral mmproj")
    ap.add_argument("--vision-only", action="store_true")
    args = ap.parse_args()

    d = args.model_dir
    params = json.loads((d / "params.json").read_text())
    src = d / "consolidated.safetensors"
    if not src.exists():
        print(f"error: {src} not found (this converter needs the Mistral-format release)", file=sys.stderr)
        return 2

    qtype = OUT_TYPES[args.outtype]
    suffix = args.outtype.upper()

    if not args.vision_only:
        tekken = json.loads((d / "tekken.json").read_text())
        template_path = d / "chat_template.jinja"
        template = template_path.read_text() if template_path.exists() else None
        out = args.outfile or d / f"Shieldstral-1.0-3B-{suffix}.gguf"
        print(f"Writing {out}")
        convert_text_model(src, out, params, tekken, qtype, template)

    if args.vision or args.vision_only:
        out = d / f"Shieldstral-1.0-3B-mmproj-{suffix}.gguf"
        print(f"Writing {out}")
        convert_vision_model(src, out, params, qtype)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
