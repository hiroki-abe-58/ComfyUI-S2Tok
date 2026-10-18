"""Download the official S2Tok checkpoint into ComfyUI's models/s2tok and verify it.

    python tools/download_model.py --models-dir <ComfyUI>/models/s2tok

Source: https://huggingface.co/AppliedIntuitionResearch/S2Tok, pinned to the revision this node pack was verified with (about 6.9 GB).
LICENSE: CC BY-NC 4.0, non-commercial use only (WEIGHTS_LICENSE.md of that repository; also the terms of ZipSplat, DA3-Giant and the
training data). Run this only if your use is non-commercial. The file is checked against the SHA-256 in tools/model_files.sha256.
Needs `huggingface_hub` (installed with ComfyUI's dependencies). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "AppliedIntuitionResearch/S2Tok"
REVISION = "ddfa40260b30b6d3a4f52e5c7a4cf8a3a2e18fe0"
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{file name: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/s2tok folder")
    parser.add_argument("--verify-only", action="store_true", help="check an existing file, download nothing")
    args = parser.parse_args(argv)

    files = expected_files()
    if not args.verify_only:
        from huggingface_hub import hf_hub_download

        for name in files:
            hf_hub_download(repo_id=REPO, filename=name, revision=REVISION, local_dir=str(args.models_dir))
    ok = True
    for name, expected in files.items():
        path = args.models_dir / name
        good = path.is_file() and sha256(path) == expected
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {name}", flush=True)
    print("verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
