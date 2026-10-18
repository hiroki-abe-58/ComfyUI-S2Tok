"""Reconstruct frame folders through this pack's runtime (the code behind the nodes), in-process, and write what `demo.py` writes.

    python scripts/run_jobs.py --checkpoint s2tok_252.pt --out <dir> [--precision fp32] [--device cuda] <frames dir> [<frames dir> ...]

Each folder of PNG / JPEG frames is loaded the way ComfyUI's image loaders give it to the nodes (8-bit / 255, a [N, H, W, 3] batch),
then reconstructed. Writes <out>/<folder name>/scene.ply with the official PLY writer, frames.json, and one line per folder to
<out>/results.jsonl (SHA-256 of scene.ply, Gaussian count, time, peak VRAM), so scripts/compare_ply.py can compare it with a run of the
official demo.py on the same folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2tok_comfy import runtime  # noqa: E402

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_folder(folder: Path) -> torch.Tensor:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES)
    return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files])).float().div(255.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision", default="fp32", choices=runtime.PRECISIONS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("folders", type=Path, nargs="+")
    args = ap.parse_args(argv)
    loaded = runtime.acquire(runtime.make_handle(str(args.checkpoint), args.device, True))
    print(json.dumps({**loaded.info, "torch": torch.__version__}), flush=True)
    runtime.vendored()
    from s2tok.gaussians import write_gaussian_ply

    rows = []
    for folder in args.folders:
        images = load_folder(folder)
        frames = runtime.model_frames(images, loaded.model.config.image_size)
        out = runtime.reconstruct(loaded, frames, args.precision)
        target = args.out / folder.name
        write_gaussian_ply(target / "scene.ply", out["final"])
        report = out["report"]
        (target / "frames.json").write_text(json.dumps({"image_size": report["image_size"], "frames": report["per_frame"]}, indent=2))
        row = {"id": folder.name, "frames": report["frames"], "gaussians": report["gaussians"], "seconds": report["seconds"],
               "peak_vram_allocated_gib": report.get("peak_vram_allocated_gib"), "precision": args.precision,
               "ply_sha256": hashlib.sha256((target / "scene.ply").read_bytes()).hexdigest()[:16]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(args.out / "results.jsonl", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
