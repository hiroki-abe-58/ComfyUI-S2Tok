"""A tiny random S2Tok in the official checkpoint format, so the real loading and streaming code runs on the CPU in seconds.

The upstream model builds its ViT-Giant backbone with fixed sizes (`DA3Encoder` -> `DinoVisionTransformer()`); `tiny_backbone()` swaps
in a small one for as long as it is active. Everything else (scene memory, admission, decoder, Gaussian head, checkpoint format,
`load_model`) is the vendored upstream code."""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path

import torch

from s2tok_comfy import runtime

BACKBONE = {"embed_dim": 64, "depth": 6, "num_heads": 2, "alt_start": 2}
CONFIG = {"image_size": 56, "out_layers": (3, 4, 5), "decoder_heads": 2, "color_dim": 16, "gaussians_per_token": 4,
          "rope3d_pairs_per_axis": 4, "admission_dim": 16, "max_scene_tokens": 40}


@contextlib.contextmanager
def tiny_backbone():
    runtime.vendored()
    import s2tok.backbone as backbone

    original = backbone.DinoVisionTransformer
    backbone.DinoVisionTransformer = functools.partial(original, **BACKBONE)
    try:
        yield
    finally:
        backbone.DinoVisionTransformer = original


def make_tiny_checkpoint(path: Path, seed: int = 0) -> Path:
    runtime.vendored()
    from s2tok.checkpoint import save_checkpoint
    from s2tok.model import S2Tok, S2TokConfig

    torch.manual_seed(seed)
    config = S2TokConfig(**CONFIG)
    with tiny_backbone():
        model = S2Tok(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, model.state_dict(), config)
    return path
