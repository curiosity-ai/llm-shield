#!/usr/bin/env python3
"""Download only the Shieldstral files this repo's converter needs.

`huggingface-cli download mistralai/Shieldstral-1.0-3B` fetches the whole
repository — about 15 GB, half of which is the *same weights twice*:
`consolidated.safetensors` is the Mistral-format checkpoint and
`model.safetensors` is the Hugging Face one. The converter reads the Mistral
format (see convert_shieldstral_to_gguf.py for why that choice is not
interchangeable), so the HF copy is pure duplication. Same story for
`tokenizer.json`, which is a re-encoding of `tekken.json`.

This fetches ~7.7 GB instead.

Stdlib only, no huggingface_hub. Downloads resume, and every file is size-checked
against what the server reports before it is moved into place — a partial
download that merely *looks* finished is the failure mode worth engineering
against here, because it surfaces much later as an access violation inside a
dequantizer.

Usage:
  python3 tools/download_shieldstral.py models/shieldstral
  python3 tools/download_shieldstral.py models/shieldstral --check
"""
from __future__ import annotations

import argparse
import errno
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "mistralai/Shieldstral-1.0-3B"
BASE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"

# What the converter actually opens.
REQUIRED = [
    "params.json",              # hyperparameters: layers, heads, YaRN, llama-4 scaling
    "consolidated.safetensors",  # the weights, Mistral format
    "tekken.json",              # vocabulary and merge ranks
]

# Small, and worth having next to the weights: the prompt template the runtime
# reproduces, plus the model card and the configs that document the architecture.
OPTIONAL = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "README.md",
]

# Present in the repository, deliberately not fetched.
SKIPPED = {
    "model.safetensors": "the same weights in Hugging Face format (~7.7 GB duplicate)",
    "tokenizer.json": "a re-encoding of tekken.json for the HF tokenizers backend",
    "tokenizer_config.json": "configuration for that same HF tokenizer",
    ".gitattributes": "repository metadata",
}

CHUNK = 4 << 20
RETRIES = 8


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
    return f"{n:.1f} GiB"


def request(url: str, token: str | None, extra: dict[str, str] | None = None) -> urllib.request.Request:
    headers = {"User-Agent": "llm-shield/download_shieldstral"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers.update(extra or {})
    return urllib.request.Request(url, headers=headers)


def remote_size(url: str, token: str | None) -> int | None:
    """
    Size the server will actually send. Hugging Face answers a HEAD on an LFS file
    with a 1024-byte redirect stub, so `x-linked-size` — not `content-length` — is
    the number that matters.
    """
    try:
        with urllib.request.urlopen(request(url, token), timeout=60) as response:
            linked = response.headers.get("x-linked-size")
            if linked:
                return int(linked)
            length = response.headers.get("content-length")
            return int(length) if length else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download(url: str, destination: Path, expected: int | None, token: str | None) -> bool:
    """Fetch `url` to `destination`, resuming a partial `.part` file. True if it changed anything."""
    if destination.exists() and expected is not None and destination.stat().st_size == expected:
        print(f"  {destination.name:<28} {human(expected):>12}  already complete")
        return False

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if expected is not None and have > expected:
            # A previous run wrote past the end — two writers, or a server that
            # restarted the stream mid-file. Resuming would compound it.
            print(f"  {destination.name:<28} discarding {human(have)} of over-long partial data")
            partial.unlink()
            have = 0
        if expected is not None and have == expected:
            break

        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with urllib.request.urlopen(request(url, token, headers), timeout=300) as response, \
                    open(partial, "ab" if have else "wb") as out:
                total = expected
                written = have
                started = time.monotonic()
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if total and sys.stderr.isatty():
                        elapsed = max(1e-6, time.monotonic() - started)
                        rate = (written - have) / elapsed
                        print(f"\r  {destination.name:<28} {human(written):>12} / {human(total)}"
                              f"  {human(rate)}/s   ", end="", file=sys.stderr)
                if total and sys.stderr.isatty():
                    print(file=sys.stderr)
        except OSError as e:
            # A full disk is not a transient network hiccup, and retrying it eight
            # times just makes the user wait to be told the same thing. The partial
            # file is left in place, so freeing space and re-running resumes.
            if getattr(e, "errno", None) == errno.ENOSPC:
                have = partial.stat().st_size if partial.exists() else 0
                free = shutil.disk_usage(destination.parent).free
                raise IOError(
                    f"{destination.name}: out of disk space after {human(have)}"
                    f"{f' of {human(expected)}' if expected else ''}. "
                    f"{human(free)} free; about {human(max(0, (expected or 0) - have))} more is needed. "
                    "The partial download is kept — free some space and re-run to resume."
                ) from e
            if not isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError)):
                raise
            wait = min(30, 2 ** attempt)
            print(f"  {destination.name:<28} attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
            continue

        have = partial.stat().st_size if partial.exists() else 0
        if expected is None or have == expected:
            break
        print(f"  {destination.name:<28} short read: {human(have)} of {human(expected)}; resuming")

    final = partial.stat().st_size if partial.exists() else 0
    if expected is not None and final != expected:
        raise IOError(
            f"{destination.name}: got {final} bytes, expected {expected}. "
            "Re-run to resume; if it keeps failing the upstream file may have changed.")

    # Only now does it get its real name, so a interrupted run never leaves
    # something that looks like a complete file.
    partial.replace(destination)
    print(f"  {destination.name:<28} {human(final):>12}  done")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path, nargs="?", default=Path("models/shieldstral"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="Hugging Face token; only needed for a gated mirror (this repo is public)")
    ap.add_argument("--check", action="store_true",
                    help="verify what is already on disk without downloading")
    ap.add_argument("--required-only", action="store_true",
                    help="skip the small extras (chat template, configs, model card)")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    names = REQUIRED + ([] if args.required_only else OPTIONAL)

    print(f"{args.repo} @ {args.revision} -> {args.dest}")
    print(f"skipping {len(SKIPPED)} files that this repo does not need:")
    for name, why in SKIPPED.items():
        print(f"  {name:<28} {why}")
    print()

    sizes: dict[str, int | None] = {}
    for name in names:
        url = BASE.format(repo=args.repo, revision=args.revision, name=name)
        sizes[name] = remote_size(url, args.token)

    known = [s for s in sizes.values() if s]
    needed = sum(
        size - (path.stat().st_size if (path := args.dest / name).exists() else 0)
        for name, size in sizes.items() if size)
    print(f"fetching {len(names)} files, {human(sum(known))} total")

    free = shutil.disk_usage(args.dest).free
    if not args.check and needed > 0:
        print(f"{human(needed)} still to download, {human(free)} free on {args.dest}")
        if needed > free:
            print(f"error: not enough disk space — need about {human(needed - free)} more",
                  file=sys.stderr)
            return 2

    if args.check:
        missing = 0
        for name in names:
            path = args.dest / name
            expected = sizes[name]
            actual = path.stat().st_size if path.exists() else 0
            state = ("missing" if actual == 0 else
                     "OK" if expected is None or actual == expected else
                     f"MISMATCH (have {human(actual)}, want {human(expected)})")
            if state != "OK":
                missing += 1
            print(f"  {name:<28} {state}")
        return 1 if missing else 0

    for name in names:
        url = BASE.format(repo=args.repo, revision=args.revision, name=name)
        expected = sizes[name]
        if expected is None and name in REQUIRED:
            print(f"error: {name} is not present in {args.repo}", file=sys.stderr)
            return 2
        if expected is None:
            print(f"  {name:<28} not in this repository; skipping")
            continue
        download(url, args.dest / name, expected, args.token)

    print()
    print(f"Ready. Convert with:")
    print(f"  python3 tools/convert_shieldstral_to_gguf.py {args.dest} --outtype q8_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
