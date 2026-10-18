"""Rebuild s2tok_comfy/vendor from an upstream checkout and rewrite vendor/MANIFEST.sha256 (maintainer tool).

    python scripts/vendor_sync.py --upstream <Applied-Intuition-Open-Source/S2Tok checkout at UPSTREAM_COMMIT> [--check]

Files are copied byte for byte, except `s2tok/model.py`, which gets the edit of `patch()` (the change proposed upstream in
Applied-Intuition-Open-Source/S2Tok#1, plus a one-line change notice as CC BY-NC 4.0 requires). Not vendored: `s2tok/viewer.py` (the
interactive viser web viewer of demo.py; nothing in the package imports it), `demo.py` and `tools/`. tests/test_vendor.py rebuilds the
tree from the pinned upstream commit in CI and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "s2tok_comfy" / "vendor"
UPSTREAM_COMMIT = "96e8f5885bc3ed9e4f5c5ffb2c66523d92447132"
LICENSE_FILES = ["LICENSE", "NOTICE", "THIRD_PARTY_LICENSES", "WEIGHTS_LICENSE.md"]
NOT_VENDORED = {"s2tok/viewer.py"}
PATCHED = {"s2tok/model.py"}

NOTICE = "# Changed by ComfyUI-S2Tok (bf16 SDPA kernel fallback); see s2tok_comfy/vendor/UPSTREAM.md.\n"
_EDITS = [
    ('            precision: "fp32", or "bf16" (bf16 autocast with FlashAttention in the backbone).\n',
     '            precision: "fp32", or "bf16" (bf16 autocast with FlashAttention in the backbone, or the\n'
     '                memory-efficient SDPA kernel where FlashAttention is not available).\n'),
    ("            flash = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if bf16 else contextlib.nullcontext()\n",
     "            # FlashAttention where PyTorch has it; builds without it (e.g. the Windows wheels)\n"
     "            # fall back to the memory-efficient kernel.\n"
     "            flash = (\n"
     "                sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])\n"
     "                if bf16\n"
     "                else contextlib.nullcontext()\n"
     "            )\n"),
]


def patch(rel: str, text: str) -> str:
    """The pack's edit of a PATCHED upstream file (str -> str, LF line endings)."""
    if rel != "s2tok/model.py":
        raise KeyError(rel)
    for old, new in _EDITS:
        if text.count(old) != 1:
            raise ValueError(f"{rel}: the upstream text to patch was not found exactly once: {old.strip()[:60]}")
        text = text.replace(old, new)
    return NOTICE + text


def upstream_files(upstream: Path) -> list[str]:
    files = sorted(p.relative_to(upstream).as_posix() for p in (upstream / "s2tok").rglob("*.py"))
    return [f for f in files if f not in NOT_VENDORED] + LICENSE_FILES


def expected_bytes(upstream: Path, name: str) -> bytes:
    data = (upstream / name).read_bytes()
    return patch(name, data.decode("utf-8")).encode("utf-8") if name in PATCHED else data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    names = upstream_files(args.upstream)
    bad = [n for n in names if not (VENDOR / n).is_file() or (VENDOR / n).read_bytes() != expected_bytes(args.upstream, n)]
    if args.check:
        print("\n".join(f"DIFFERS {n}" for n in bad) or "vendor tree matches the upstream checkout")
        return 1 if bad else 0
    for n in bad:
        (VENDOR / n).parent.mkdir(parents=True, exist_ok=True)
        (VENDOR / n).write_bytes(expected_bytes(args.upstream, n))
        print("wrote", n)
    lines = [f"{hashlib.sha256((VENDOR / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (VENDOR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"MANIFEST.sha256: {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
