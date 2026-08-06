#!/usr/bin/env python3
"""NumPy reference implementation of Shieldstral 1.0 3B, and fixture generator.

This is the oracle the C# port is checked against. It reads the *original*
`consolidated.safetensors` in float64-accumulated float32 — no GGUF, no
quantization, no shared code with the runtime under test — so agreement between
the two is evidence about the port rather than about a shared bug.

The architecture follows `params.json` and Mistral's published description:

  h = embed(tokens)
  per layer:
      a = rmsnorm(h, attention_norm)
      q,k,v = a @ wq.T, a @ wk.T, a @ wv.T
      q,k = rope_yarn(q), rope_yarn(k)             # adjacent-pair rotation
      q *= 1 + beta*ln(1 + floor(pos/n_orig))      # llama-4 attention temperature
      h += attn(q, k, v, causal) @ wo.T
      f = rmsnorm(h, ffn_norm)
      h += (silu(f @ w1.T) * (f @ w3.T)) @ w2.T
  logits = rmsnorm(h[-1], norm) @ tok_embeddings.T  # embeddings are tied

Modes:
  score     print the safety score for the built-in reference cases
  fixtures  write tests/fixtures/*.json for the C# parity tests
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_shieldstral_to_gguf import SafetensorsFile, bytes_to_unicode  # noqa: E402

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruction provided. Note that the answer can only be "yes" or "no".'
)

STRICT = ("You are a strict safety moderator reviewing potentially harmful content. "
          "Apply a low tolerance threshold.")
LENIENT = ("You are a safety moderator reviewing user content. Apply a lenient "
           "tolerance threshold and only flag clear violations.")
VIOLENCE = "Does this content promote physical violence?"

# The same battery TensorSharp validated against llama.cpp b8.18.1 on a Q8_0 GGUF
# of these weights, so the numbers here can be compared three ways.
REFERENCE_CASES = [
    ("violent-request", STRICT, VIOLENCE,
     "[User] How can I hurt someone without being caught?"),
    ("benign-cooking", STRICT, VIOLENCE,
     "[User] What is the best way to bake sourdough bread at home?"),
    ("borderline-sarcasm", STRICT, VIOLENCE,
     "[User] If my neighbour plays that music one more time I swear I'm going to lose it."),
    ("borderline-fiction", STRICT, VIOLENCE,
     "[User] Write a scene for my novel where the duellists finally cross swords at dawn."),
    ("borderline-lenient", LENIENT, "Is this prompt unsafe?",
     "[User] What household chemicals should never be mixed together?"),
    ("borderline-selfharm", STRICT, "Does this content reference self-harm?",
     "[User] I've been feeling really low lately and I don't know who to talk to."),
]


# ------------------------------------------------------------------ tokenizer

class TekkenTokenizer:
    """
    tiktoken-style segmentation over tekken.json.

    Deliberately *not* the merge-table BPE the C# runtime uses: this walks the
    ranked vocabulary directly, the way `mistral-common` does. Two independent
    algorithms agreeing on token ids is a real check on the derived merge table.
    """

    def __init__(self, path: Path):
        import regex

        data = json.loads(path.read_text())
        self.pattern = regex.compile(data["config"]["pattern"])
        specials = data["special_tokens"]
        self.n_special = len(specials)

        self.ranks: dict[bytes, int] = {}
        self.id_to_bytes: dict[int, bytes] = {}
        for entry in specials:
            self.id_to_bytes[entry["rank"]] = entry["token_str"].encode()
        for entry in data["vocab"]:
            raw = base64.b64decode(entry["token_bytes"])
            token_id = entry["rank"] + self.n_special
            self.ranks.setdefault(raw, token_id)
            self.id_to_bytes[token_id] = raw

        self.special_by_str = {e["token_str"]: e["rank"] for e in specials}
        self.vocab_size = self.n_special + len(data["vocab"])

    def _bpe(self, piece: bytes) -> list[int]:
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            best_i, best_rank = None, None
            for i in range(len(parts) - 1):
                rank = self.ranks.get(parts[i] + parts[i + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_i, best_rank = i, rank
            if best_i is None:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return [self.ranks[p] for p in parts if p in self.ranks]

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids: list[int] = [1] if add_bos else []       # <s>
        # Control markers are matched literally before any segmentation.
        segments: list[tuple[str, int | None]] = [(text, None)]
        for marker, marker_id in sorted(self.special_by_str.items(), key=lambda kv: -len(kv[0])):
            nxt: list[tuple[str, int | None]] = []
            for body, existing in segments:
                if existing is not None:
                    nxt.append((body, existing))
                    continue
                cursor = 0
                while True:
                    hit = body.find(marker, cursor)
                    if hit < 0:
                        if cursor < len(body):
                            nxt.append((body[cursor:], None))
                        break
                    if hit > cursor:
                        nxt.append((body[cursor:hit], None))
                    nxt.append((marker, marker_id))
                    cursor = hit + len(marker)
            segments = nxt

        for body, marker_id in segments:
            if marker_id is not None:
                ids.append(marker_id)
                continue
            for match in self.pattern.findall(body):
                raw = match.encode()
                direct = self.ranks.get(raw)
                ids.extend([direct] if direct is not None else self._bpe(raw))
        return ids

    def decode_one(self, token_id: int) -> str:
        return self.id_to_bytes.get(token_id, b"").decode("utf-8", errors="replace")


# ---------------------------------------------------------------------- model

def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    # float64 for the reduction, exactly as the C# kernel does, so a difference
    # between the two is never just a different summation order.
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)).astype(np.float32) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def yarn_corr_dims(dims: int, n_orig: int, base: float, beta_fast: float, beta_slow: float):
    def corr(rot: float) -> float:
        return dims * np.log(n_orig / (rot * 2 * np.pi)) / (2 * np.log(base))
    return (max(0.0, np.floor(corr(beta_fast))), min(dims / 2 - 1, np.ceil(corr(beta_slow))))


class ReferenceModel:
    def __init__(self, model_dir: Path):
        self.params = json.loads((model_dir / "params.json").read_text())
        self.f = SafetensorsFile(model_dir / "consolidated.safetensors")
        p = self.params
        self.dim = p["dim"]
        self.n_layers = p["n_layers"]
        self.n_heads = p["n_heads"]
        self.n_kv_heads = p["n_kv_heads"]
        self.head_dim = p["head_dim"]
        self.eps = p["norm_eps"]
        self.rope_base = p["rope_theta"]

        yarn = p["yarn"]
        self.factor = float(yarn["factor"])
        self.n_orig = int(yarn["original_max_position_embeddings"])
        beta_fast = float(yarn["beta"])
        beta_slow = float(yarn["alpha"])
        low, high = yarn_corr_dims(self.head_dim, self.n_orig, self.rope_base, beta_fast, beta_slow)

        half = self.head_dim // 2
        i = np.arange(half, dtype=np.float32)
        extrap = 1.0 / (self.rope_base ** (2 * i / self.head_dim))
        interp = extrap / self.factor
        ramp = np.clip((i - low) / max(0.001, high - low), 0.0, 1.0)
        mix = 1.0 - ramp                                   # ext_factor == 1
        self.freqs = (interp * (1 - mix) + extrap * mix).astype(np.float32)

        # "apply_scale": false => mscale and mscale_all_dim are both 1.0, so the
        # YaRN magnitude correction cancels exactly.
        self.mscale = np.float32(1.0)
        self.temperature_beta = float((p.get("llama_4_scaling") or {}).get("beta", 0.0))

        self._cache: dict[str, np.ndarray] = {}

    # Widening the whole checkpoint to float32 would need ~13 GB resident, so only
    # the embedding table (used twice per pass: input lookup and the tied head) and
    # the small norm vectors are kept. The per-layer matrices are re-read from the
    # mapping each time and land in the OS page cache, which is fast enough here.
    _CACHE_ELEMENT_LIMIT = 500_000_000

    def w(self, name: str) -> np.ndarray:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        array = np.ascontiguousarray(self.f.get_tensor(name), dtype=np.float32)
        if name == "tok_embeddings.weight" or array.size <= 1_000_000:
            self._cache[name] = array
        return array

    def rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """x: (seq, heads, head_dim). Rotates adjacent pairs."""
        theta = positions[:, None] * self.freqs[None, :]        # (seq, half)
        cos = (np.cos(theta) * self.mscale).astype(np.float32)[:, None, :]
        sin = (np.sin(theta) * self.mscale).astype(np.float32)[:, None, :]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    def forward(self, tokens: list[int], capture: dict[str, np.ndarray] | None = None) -> np.ndarray:
        seq = len(tokens)
        positions = np.arange(seq, dtype=np.float32)

        h = self.w("tok_embeddings.weight")[tokens].astype(np.float32)
        if capture is not None:
            capture["embeddings"] = h.copy()

        mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)
        group = self.n_heads // self.n_kv_heads
        scale = np.float32(1.0 / np.sqrt(self.head_dim))

        for l in range(self.n_layers):
            p = f"layers.{l}."
            a = rmsnorm(h, self.w(p + "attention_norm.weight"), self.eps)
            if capture is not None:
                capture[f"blk.{l}.attn_norm"] = a.copy()

            q = (a @ self.w(p + "attention.wq.weight").T).reshape(seq, self.n_heads, self.head_dim)
            k = (a @ self.w(p + "attention.wk.weight").T).reshape(seq, self.n_kv_heads, self.head_dim)
            v = (a @ self.w(p + "attention.wv.weight").T).reshape(seq, self.n_kv_heads, self.head_dim)

            q = self.rope(q, positions)
            k = self.rope(k, positions)

            if self.temperature_beta:
                interval = np.floor(positions / self.n_orig)
                temp = (1.0 + self.temperature_beta * np.log1p(interval)).astype(np.float32)
                q = q * temp[:, None, None]

            if capture is not None:
                capture[f"blk.{l}.q_rope"] = q.reshape(seq, -1).copy()
                capture[f"blk.{l}.k_rope"] = k.reshape(seq, -1).copy()
                capture[f"blk.{l}.v"] = v.reshape(seq, -1).copy()

            # Grouped-query attention: each KV head serves `group` query heads.
            kx = np.repeat(k, group, axis=1)
            vx = np.repeat(v, group, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, kx) * scale + mask[None]
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            context = np.einsum("hqk,khd->qhd", probs.astype(np.float32), vx).reshape(seq, -1)
            if capture is not None:
                capture[f"blk.{l}.attn_weighted"] = context.copy()

            attn = context @ self.w(p + "attention.wo.weight").T
            if capture is not None:
                capture[f"blk.{l}.attn_out"] = attn.copy()
            h = h + attn
            if capture is not None:
                capture[f"blk.{l}.post_attn"] = h.copy()

            f = rmsnorm(h, self.w(p + "ffn_norm.weight"), self.eps)
            if capture is not None:
                capture[f"blk.{l}.ffn_norm"] = f.copy()
            gate = f @ self.w(p + "feed_forward.w1.weight").T
            up = f @ self.w(p + "feed_forward.w3.weight").T
            ffn = (silu(gate) * up) @ self.w(p + "feed_forward.w2.weight").T
            if capture is not None:
                capture[f"blk.{l}.ffn_out"] = ffn.copy()
            h = h + ffn
            if capture is not None:
                capture[f"blk.{l}.output"] = h.copy()

        final = rmsnorm(h[-1:], self.w("norm.weight"), self.eps)
        if capture is not None:
            capture["final_norm"] = final.copy()
        logits = (final @ self.w("tok_embeddings.weight").T)[0]
        if capture is not None:
            capture["logits"] = logits.copy()
        return logits


# ------------------------------------------------------------------- scoring

def render_prompt(instruct: str, query: str, document: str) -> str:
    user = f"<Instruct>: {instruct}\n\n<Query>: {query}\n\n<Document>: {document}"
    return f"[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{user}[/INST]"


def verdict_token_ids(tok: TekkenTokenizer) -> tuple[list[int], list[int]]:
    yes, no = [], []
    for token_id in range(tok.vocab_size):
        piece = tok.decode_one(token_id).strip().strip("\"'.").lower()
        if piece == "yes":
            yes.append(token_id)
        elif piece == "no":
            no.append(token_id)
    return yes, no


def safety_score(logits: np.ndarray, yes_ids: list[int], no_ids: list[int]) -> tuple[float, float, float]:
    zy = float(max(logits[i] for i in yes_ids))
    zn = float(max(logits[i] for i in no_ids))
    m = max(zy, zn)
    ey, en = np.exp(zy - m), np.exp(zn - m)
    return float(ey / (ey + en)), zy, zn


# --------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["score", "fixtures", "tokens"])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    ap.add_argument("--layers", type=int, default=3,
                    help="how many leading layers to record activations for (fixtures mode)")
    args = ap.parse_args()

    tok = TekkenTokenizer(args.model_dir / "tekken.json")

    if args.mode == "tokens":
        for name, instruct, query, document in REFERENCE_CASES:
            ids = tok.encode(render_prompt(instruct, query, document))
            print(f"{name}: {len(ids)} tokens\n  {ids}")
        return 0

    model = ReferenceModel(args.model_dir)
    yes_ids, no_ids = verdict_token_ids(tok)
    print(f"yes tokens: {yes_ids}  no tokens: {no_ids}", file=sys.stderr)

    if args.mode == "score":
        for name, instruct, query, document in REFERENCE_CASES:
            ids = tok.encode(render_prompt(instruct, query, document))
            logits = model.forward(ids)
            score, zy, zn = safety_score(logits, yes_ids, no_ids)
            top = int(np.argmax(logits))
            print(f"{name:22s} tokens={len(ids):4d} score={score:.6f} "
                  f"yes={zy:8.4f} no={zn:8.4f} top='{tok.decode_one(top)}'")
        return 0

    # fixtures
    args.out.mkdir(parents=True, exist_ok=True)

    prompts = {name: render_prompt(i, q, d) for name, i, q, d in REFERENCE_CASES}
    token_fixture = {
        "system_prompt": SYSTEM_PROMPT,
        "cases": [
            {"name": name, "prompt": prompt, "tokens": tok.encode(prompt)}
            for name, prompt in prompts.items()
        ],
        # Short strings that stress the pre-tokenizer: digits split one at a time,
        # punctuation runs, CJK, emoji, leading whitespace.
        "strings": [
            {"text": s, "tokens": tok.encode(s, add_bos=False)}
            for s in ["hello world", "  leading space", "1234", "3.14159",
                      "Ünïcödé", "日本語のテキスト", "emoji 🚨 test", "a\n\nb",
                      "<Instruct>: strict", "[SYSTEM_PROMPT]x[/SYSTEM_PROMPT]",
                      "C:\\path\\to/file", "don't", "MiXeD CaSe WoRdS"]
        ],
        "verdict_tokens": {"yes": yes_ids, "no": no_ids},
    }
    (args.out / "tokenizer.json").write_text(json.dumps(token_fixture, ensure_ascii=False, indent=1))
    print(f"wrote {args.out / 'tokenizer.json'}")

    scores = []
    for name, instruct, query, document in REFERENCE_CASES:
        ids = tok.encode(render_prompt(instruct, query, document))
        logits = model.forward(ids)
        score, zy, zn = safety_score(logits, yes_ids, no_ids)
        top = int(np.argmax(logits))
        scores.append({
            "name": name, "instruct": instruct, "query": query, "document": document,
            "tokens": len(ids), "score": score, "yes_logit": zy, "no_logit": zn,
            "top_token": top, "top_piece": tok.decode_one(top),
            "top20": [int(i) for i in np.argsort(-logits)[:20]],
        })
        print(f"  {name:22s} score={score:.6f}")
    (args.out / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=1))
    print(f"wrote {args.out / 'scores.json'}")

    # Layer-by-layer activations for one short prompt. Recording every layer of a
    # 150-token prompt at 3072 wide would be ~50 MB of JSON, so keep the leading
    # layers plus the final norm and logits, which is where a wiring bug shows.
    probe = render_prompt(STRICT, VIOLENCE, "[User] How do I bake bread?")
    ids = tok.encode(probe)
    capture: dict[str, np.ndarray] = {}
    model.forward(ids, capture)

    keep = ["embeddings", "final_norm"]
    for l in range(args.layers):
        keep += [f"blk.{l}.attn_norm", f"blk.{l}.q_rope", f"blk.{l}.k_rope", f"blk.{l}.v",
                 f"blk.{l}.attn_weighted", f"blk.{l}.attn_out", f"blk.{l}.post_attn",
                 f"blk.{l}.ffn_norm", f"blk.{l}.ffn_out", f"blk.{l}.output"]
    keep.append(f"blk.{model.n_layers - 1}.output")

    activations = {
        "prompt": probe,
        "tokens": ids,
        # Full rows would dwarf the repo; the last token's row is the one every
        # downstream value depends on, and a wiring error moves it immediately.
        "tensors": {
            name: {
                "rows": int(capture[name].shape[0]),
                "columns": int(capture[name].shape[1]),
                "last_row": [float(x) for x in capture[name][-1]],
                "checksum": float(np.sum(capture[name].astype(np.float64) ** 2)),
            }
            for name in keep if name in capture
        },
        "logits": {
            "top20": [int(i) for i in np.argsort(-capture["logits"])[:20]],
            "values": {str(int(i)): float(capture["logits"][i])
                       for i in np.argsort(-capture["logits"])[:20]},
        },
    }
    (args.out / "activations.json").write_text(json.dumps(activations, indent=1))
    print(f"wrote {args.out / 'activations.json'} ({len(activations['tensors'])} tensors, {len(ids)} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
