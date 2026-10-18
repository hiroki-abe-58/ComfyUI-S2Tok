"""Loading and streaming reconstruction with S2Tok (one resident model).

The steps are those of the official `demo.py`: every frame is converted to 8-bit RGB, center-cropped to a square and resized to the model
resolution with Lanczos filtering, then fed to `S2Tok.step()` one at a time; the reconstruction after the last frame is the result.
The model code is the vendored upstream package (vendor/UPSTREAM.md). Nothing here downloads anything.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parent / "vendor" / "s2tok"
PRECISIONS = ("fp32", "bf16")

_LOCK = threading.Lock()
_SLOT = None


def vendored():
    """Make the vendored package importable as `s2tok` (its modules import each other as `s2tok.*`). Its `__init__.py` only re-exports
    names and is not run; the pack imports the submodules it needs. sys.path is not changed."""
    existing = sys.modules.get("s2tok")
    if existing is not None:
        if [Path(p).resolve() for p in getattr(existing, "__path__", [])] != [PACKAGE_DIR]:
            raise RuntimeError("another package named 's2tok' is already imported in this Python process; it conflicts with this node pack")
        return existing
    package = types.ModuleType("s2tok", "S2Tok, vendored by ComfyUI-S2Tok (s2tok_comfy/vendor/UPSTREAM.md)")
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules["s2tok"] = package
    return package


def model_frames(images: torch.Tensor, size: int) -> list[torch.Tensor]:
    """ComfyUI IMAGE [N, H, W, 3] in 0..1 -> one [3, size, size] tensor per frame, prepared as `demo.py` prepares a frame it read from
    disk: 8-bit RGB, the central square (offsets rounded down), Lanczos resize, then / 255."""
    pixels = (images.detach().cpu().float().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    frames = []
    for frame in pixels:
        h, w = frame.shape[:2]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        square = Image.fromarray(frame).crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)
        frames.append(torch.from_numpy(np.array(square)).permute(2, 0, 1).float().div(255))
    return frames


@dataclass
class Loaded:
    key: tuple
    model: torch.nn.Module
    device: torch.device
    info: dict = field(default_factory=dict)


def make_handle(checkpoint: str, device: str, keep_loaded: bool) -> dict:
    return {"checkpoint": checkpoint, "device": resolve_device(device), "keep_loaded": bool(keep_loaded)}


def resolve_device(choice: str) -> str:
    if choice == "auto":
        try:
            import comfy.model_management as mm

            return str(mm.get_torch_device())
        except ImportError:  # outside ComfyUI
            return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["checkpoint"], handle["device"])
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        _SLOT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ComfyUI runs nodes under torch.inference_mode(); build the weights as ordinary tensors, as the official loader does.
        with torch.inference_mode(False):
            _SLOT = _load(key)
        return _SLOT


def _load(key) -> Loaded:
    checkpoint, device_name = key
    vendored()
    from s2tok.checkpoint import load_model

    started = time.perf_counter()
    model = load_model(checkpoint, device=device_name)  # official loader: torch.load(weights_only=True), strict state dict
    info = {
        "checkpoint": Path(checkpoint).name,
        "device": device_name,
        "load_seconds": round(time.perf_counter() - started, 2),
        "parameters_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
        "image_size": model.config.image_size,
        "max_scene_tokens": model.config.max_scene_tokens,
    }
    return Loaded(key=key, model=model, device=torch.device(device_name), info=info)


@torch.no_grad()
def reconstruct(loaded: Loaded, frames: list[torch.Tensor], precision: str = "fp32", keep_every_step: bool = False, progress=None) -> dict:
    """Run the frames through S2Tok in order. Returns {"final": Gaussians on the CPU, "steps": [Gaussians on the CPU] (only with
    keep_every_step), "report": dict}. precision "bf16" is the official bfloat16 autocast; it needs CUDA."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "bf16" and loaded.device.type != "cuda":
        raise ValueError("precision bf16 uses CUDA autocast; use fp32 on the CPU")
    if not frames:
        raise ValueError("no frames to reconstruct")
    model, device = loaded.model, loaded.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    memory = model.init_memory()
    records, steps = [], []
    for i, frame in enumerate(frames):
        out, memory = model.step(frame.to(device), memory, precision=precision)
        records.append({"frame": i, "log_scale": out.log_scale.item(), "admitted_patches": out.admitted.numel(),
                        "scene_tokens": memory.num_tokens})
        if keep_every_step:
            steps.append(out.gaussians.to("cpu"))
        if progress is not None:
            progress(i + 1)
    final = steps[-1] if keep_every_step else out.gaussians.to("cpu")
    report = {
        "frames": len(frames),
        "image_size": frames[0].shape[-1],
        "precision": precision,
        "gaussians": len(final),
        "scene_tokens": memory.num_tokens,
        "memory_full": memory.num_tokens >= model.config.max_scene_tokens,
        "seconds": round(time.perf_counter() - started, 3),
        "per_frame": records,
    }
    if device.type == "cuda":
        report["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    return {"final": final, "steps": steps, "report": report}


def splat_tensors(gaussians_list) -> dict:
    """S2Tok Gaussians (activated: linear scales, wxyz unit quaternions, opacities in 0..1, SH [N, K, 3]) -> the (B, N, ...) tensors of
    ComfyUI's SPLAT, zero-padded to the largest N with `counts` when the items differ in length. Coordinates stay in the camera frame of
    the first image (x right, y down, z forward), which is also the 3DGS convention of ComfyUI's splat nodes."""
    sizes = [len(g) for g in gaussians_list]
    n, k = max(sizes), gaussians_list[0].sh.shape[1]

    def stack(name, shape):
        out = torch.zeros((len(gaussians_list), n, *shape), dtype=torch.float32)
        for i, g in enumerate(gaussians_list):
            out[i, : sizes[i]] = getattr(g, name).float().reshape(sizes[i], *shape)
        return out

    rotations = stack("quats", (4,))
    # The official PLY writer (write_gaussian_ply) normalises the quaternions once more in numpy before writing; doing the same here makes
    # a .ply saved by ComfyUI's Save Splat byte-identical to demo.py's scene.ply (without it they differ by one float32 step).
    q = rotations.numpy()
    rotations = torch.from_numpy(q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12))
    return {
        "positions": stack("means", (3,)),
        "scales": stack("scales", (3,)),
        "rotations": rotations,
        "opacities": stack("opacities", (1,)),
        "sh": stack("sh", (k, 3)),
        "counts": None if len(set(sizes)) == 1 else torch.tensor(sizes, dtype=torch.long),
    }
