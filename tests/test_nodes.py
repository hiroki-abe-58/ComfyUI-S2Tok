"""The node pack as ComfyUI loads it: registration, checkpoint discovery, ComfyUI's own validation of the shipped workflows, and node
execution with the tiny random model on the CPU (the trained model runs in the real-queue checks of docs/VERIFICATION.md)."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from sample_video import make_sample_video

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"S2TokLoader", "S2TokReconstruct", "S2TokUnload"}


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tiny_checkpoint, tiny_backbone_active, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    input_dir = base / "input"
    make_sample_video(input_dir / "example.mp4")  # the file name the shipped workflows use
    folder_paths.set_input_directory(str(input_dir))
    folder_paths.set_output_directory(str(base / "output"))
    models = tiny_checkpoint.parent
    shutil.copyfile(tiny_checkpoint, models / "s2tok_252.pt")  # the name the shipped workflows use
    (models / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
    folder_paths.add_model_folder_path("s2tok", str(models), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    return nodes


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `s2tok_comfy` the tests import)."""
    return sys.modules[comfy_nodes.NODE_CLASS_MAPPINGS["S2TokReconstruct"].__module__.rsplit(".", 1)[0] + "." + name]


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_only_checkpoint_files_are_listed_and_names_are_contained(comfy_nodes):
    paths = pack_module(comfy_nodes, "paths")
    assert paths.checkpoints() == ["s2tok_252.pt", "tiny_s2tok.pt"]
    for bad in ("notes.txt", "../s2tok_252.pt", "missing.pt", "..\\x.pt"):
        with pytest.raises(FileNotFoundError):
            paths.checkpoint_path(bad)


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def _images(n=6):
    return torch.rand(n, 70, 90, 3, generator=torch.Generator().manual_seed(n))


def _run(comfy_nodes, images, keep_loaded=True, **kw):
    handle = comfy_nodes.NODE_CLASS_MAPPINGS["S2TokLoader"].execute("s2tok_252.pt", "cpu", keep_loaded).args[0]
    args = {"frame_interval": 1, "max_frames": 0, "precision": "fp32", "every_frame": False}
    args.update(kw)
    return comfy_nodes.NODE_CLASS_MAPPINGS["S2TokReconstruct"].execute(handle, images, **args).args


def test_reconstruct_node_returns_a_comfy_splat(comfy_nodes):
    from comfy_api.latest import Types

    splat, report = _run(comfy_nodes, _images())
    parsed = json.loads(report)
    assert isinstance(splat, Types.SPLAT) and splat.positions.shape[0] == 1 and splat.counts is None
    assert splat.positions.shape[1] == parsed["gaussians"] and parsed["frames"] == 6 and parsed["input_frames"] == 6


def test_frame_interval_and_max_frames(comfy_nodes):
    images = _images(9)
    _, report = _run(comfy_nodes, images, frame_interval=2, max_frames=3)
    assert json.loads(report)["frames"] == 3
    picked, _ = _run(comfy_nodes, images[::2][:3])
    again, _ = _run(comfy_nodes, images, frame_interval=2, max_frames=3)
    assert torch.equal(picked.positions, again.positions)


def test_every_frame_gives_a_batch_whose_last_item_is_the_scene(comfy_nodes):
    images = _images()
    batch, report = _run(comfy_nodes, images, every_frame=True)
    final, _ = _run(comfy_nodes, images)
    last = int(batch.counts[-1]) if batch.counts is not None else batch.positions.shape[1]
    assert batch.positions.shape[0] == 6 and torch.equal(batch.positions[-1, :last], final.positions[0])


def test_saved_ply_equals_the_official_writer(comfy_nodes, tmp_path):
    """ComfyUI's own .ply writer (Create 3D File from a splat) gives the same bytes as upstream's write_gaussian_ply."""
    from comfy_extras import nodes_gaussian_splat

    runtime = pack_module(comfy_nodes, "runtime")
    splat, _ = _run(comfy_nodes, _images())
    core = nodes_gaussian_splat._gaussian_ply_bytes(splat.positions[0], splat.scales[0], splat.rotations[0], splat.opacities[0], splat.sh[0])
    runtime.vendored()
    from s2tok.gaussians import Gaussians, write_gaussian_ply

    loaded = runtime.acquire(runtime.make_handle(pack_module(comfy_nodes, "paths").checkpoint_path("s2tok_252.pt"), "cpu", True))
    out = runtime.reconstruct(loaded, runtime.model_frames(_images(), 56), "fp32")["final"]
    write_gaussian_ply(tmp_path / "official.ply", out)
    assert core == (tmp_path / "official.ply").read_bytes()
    assert isinstance(out, Gaussians)


def test_bf16_on_the_cpu_fails_before_loading(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    runtime.release()
    with pytest.raises(ValueError, match="CUDA"):
        _run(comfy_nodes, _images(), precision="bf16")
    assert runtime._SLOT is None


def test_keep_loaded_false_releases_and_unload_is_safe(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    _run(comfy_nodes, _images(), keep_loaded=False)
    assert runtime._SLOT is None
    unload = comfy_nodes.NODE_CLASS_MAPPINGS["S2TokUnload"]
    assert json.loads(unload.execute().args[0])["unloaded"] is False
    _run(comfy_nodes, _images())
    assert json.loads(unload.execute().args[0])["unloaded"] is True and runtime._SLOT is None


def test_interrupt_stops_between_frames(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] > 2:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        _run(comfy_nodes, _images())
    monkeypatch.undo()
    splat, _ = _run(comfy_nodes, _images())  # the resident model still works
    assert splat.positions.shape[1] > 0


def test_shipped_gui_workflows_use_known_nodes():
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        types = {n["type"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
        assert types <= NODE_IDS | {"LoadVideo", "GetVideoComponents", "SplatToFile3D", "SaveGaussianSplat", "CreateCameraInfo", "RenderSplat",
                                    "SaveImage", "PreviewAny", "CreateVideo", "SaveVideo"}, path.name
