"""Runtime on the CPU with a tiny random S2Tok in the official checkpoint format: preprocessing against the official demo.py, loading,
streaming, SPLAT conversion. The trained model is compared with the official demo in docs/VERIFICATION.md."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from s2tok_comfy import runtime


def official_center_crop_resize(upstream_dir: Path):
    """`center_crop_resize` from the upstream demo.py, taken from its source (demo.py itself imports the model package at the top)."""
    tree = ast.parse((upstream_dir / "demo.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "center_crop_resize")
    namespace = {"Image": Image}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "demo.py", "exec"), namespace)  # test-only: the upstream function, unchanged
    return namespace["center_crop_resize"]


@pytest.mark.parametrize("shape", [(1080, 1920), (1920, 1080), (300, 301), (252, 252), (101, 77)])
def test_frames_are_prepared_exactly_like_demo_py(upstream_dir, shape):
    center_crop_resize = official_center_crop_resize(upstream_dir)
    rng = np.random.default_rng(sum(shape))
    pixels = rng.integers(0, 256, size=(2, *shape, 3), dtype=np.uint8)
    images = torch.from_numpy(pixels).float().div(255.0)  # what ComfyUI's image loaders give the nodes
    ours = runtime.model_frames(images, 252)
    for frame, mine in zip(pixels, ours):
        expected = torch.from_numpy(np.array(center_crop_resize(Image.fromarray(frame), 252))).permute(2, 0, 1).float().div(255)
        assert torch.equal(mine, expected)


def test_out_of_range_image_values_are_clipped():
    frames = runtime.model_frames(torch.tensor([[[[1.5, -0.2, 0.5]]]]).expand(1, 14, 14, 3), 14)
    assert frames[0].min() >= 0 and frames[0].max() <= 1


@pytest.fixture(scope="module")
def loaded(tiny_checkpoint, tiny_backbone_active):
    yield runtime.acquire(runtime.make_handle(str(tiny_checkpoint), "cpu", True))
    runtime.release()


def frames(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return runtime.model_frames(torch.rand(n, 70, 90, 3, generator=g), 56)


def test_streaming_is_deterministic_and_reports_every_frame(loaded):
    a = runtime.reconstruct(loaded, frames(), "fp32", keep_every_step=True)
    b = runtime.reconstruct(loaded, frames(), "fp32")
    assert torch.equal(a["final"].means, b["final"].means) and torch.equal(a["final"].sh, b["final"].sh)
    report = a["report"]
    assert report["frames"] == 5 and len(report["per_frame"]) == 5 and len(a["steps"]) == 5
    assert report["per_frame"][0]["admitted_patches"] == (56 // 14) ** 2  # every patch of the first frame
    assert report["gaussians"] == report["scene_tokens"] * 4  # gaussians_per_token of the tiny config
    assert report["memory_full"] is True  # the tiny memory holds 40 tokens
    assert [len(s) for s in a["steps"]] == sorted(len(s) for s in a["steps"])


def test_other_frames_give_another_scene(loaded):
    a = runtime.reconstruct(loaded, frames(seed=0), "fp32")["final"]
    b = runtime.reconstruct(loaded, frames(seed=1), "fp32")["final"]
    assert len(a) != len(b) or not torch.equal(a.means, b.means)


def test_bad_requests(loaded):
    with pytest.raises(ValueError, match="CUDA"):
        runtime.reconstruct(loaded, frames(), "bf16")
    with pytest.raises(ValueError):
        runtime.reconstruct(loaded, frames(), "fp16")
    with pytest.raises(ValueError, match="no frames"):
        runtime.reconstruct(loaded, [], "fp32")


def test_splat_tensors_pad_and_count(loaded):
    steps = runtime.reconstruct(loaded, frames(), "fp32", keep_every_step=True)["steps"]
    s = runtime.splat_tensors(steps)
    sizes = [len(g) for g in steps]
    assert s["positions"].shape == (5, max(sizes), 3) and s["sh"].shape == (5, max(sizes), 4, 3) and s["opacities"].shape[-1] == 1
    assert s["counts"].tolist() == sizes
    first = sizes[0]
    assert torch.equal(s["positions"][0, :first], steps[0].means) and not s["positions"][0, first:].any()
    norms = s["rotations"][-1].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    assert runtime.splat_tensors([steps[-1]])["counts"] is None


def test_loading_inside_inference_mode_gives_ordinary_parameters(tiny_checkpoint, tiny_backbone_active):
    runtime.release()
    with torch.inference_mode():
        loaded = runtime.acquire(runtime.make_handle(str(tiny_checkpoint), "cpu", False))
    assert not any(p.is_inference() for p in loaded.model.parameters())
    runtime.release()


def test_vendored_package_is_registered_once():
    assert runtime.vendored() is runtime.vendored()
