"""ComfyUI nodes for S2Tok streaming 3D Gaussian reconstruction (runs in the ComfyUI process; nothing is downloaded at run time).

The output is ComfyUI's own SPLAT type, so the core nodes render it (Render Splat), save it as .ply (Create 3D File + Save Splat) or
preview it."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import ComfyExtension, Types, io

from . import paths, runtime

S2TokModel = io.Custom("S2TOK_MODEL")

NC_NOTE = "S2Tok code and weights are CC BY-NC 4.0: non-commercial use only."


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick(done: int):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(done, total), total)

    return tick


class S2TokLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="S2TokLoader",
            display_name="S2Tok Loader",
            category="loaders/s2tok",
            description=("Loads the official S2Tok checkpoint (1.72 B parameters, float32) from models/s2tok. One model stays resident; "
                         "loading another frees it. " + NC_NOTE),
            inputs=[
                io.Combo.Input("checkpoint", options=paths.checkpoints(), tooltip="s2tok_252.pt from AppliedIntuitionResearch/S2Tok"),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after the job. Turn off to free VRAM after every run."),
            ],
            outputs=[S2TokModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, checkpoint, device, keep_loaded) -> io.NodeOutput:
        return io.NodeOutput(runtime.make_handle(paths.checkpoint_path(checkpoint), device, keep_loaded))


class S2TokReconstruct(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="S2TokReconstruct",
            display_name="S2Tok Reconstruct",
            category="3d/s2tok",
            description=("Reconstructs a static scene as 3D Gaussians from the frames of a video, one frame at a time (no camera poses "
                         "needed). Frames are center-cropped to a square and resized to 252 x 252. The Gaussians are in the camera frame "
                         "of the first frame (x right, y down, z forward), up to a global scale. " + NC_NOTE),
            inputs=[
                S2TokModel.Input("model"),
                io.Image.Input("images", tooltip="Frames in order, e.g. from Load Video + Get Video Components"),
                io.Int.Input("frame_interval", default=1, min=1, max=1000, tooltip="Use every n-th frame (demo.py --frame_interval)."),
                io.Int.Input("max_frames", default=0, min=0, max=100000, tooltip="Stop after this many frames; 0 = all."),
                io.Combo.Input("precision", options=list(runtime.PRECISIONS), default="fp32",
                               tooltip="fp32 is the official default; bf16 is faster and its results differ slightly (CUDA only)."),
                io.Boolean.Input("every_frame", default=False,
                                 tooltip="Output the reconstruction after every frame as a batch (for animations) instead of only the last."),
            ],
            outputs=[io.Splat.Output(display_name="splat"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, images, frame_interval, max_frames, precision, every_frame) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        if precision == "bf16" and not model["device"].startswith("cuda"):
            raise ValueError("precision bf16 uses CUDA autocast; use fp32 on the CPU")  # before loading 6.9 GB for nothing
        picked = images[::frame_interval]
        if max_frames:
            picked = picked[:max_frames]
        loaded = runtime.acquire(model)
        info = loaded.info
        try:
            frames = runtime.model_frames(picked, loaded.model.config.image_size)
            out = runtime.reconstruct(loaded, frames, precision, keep_every_step=every_frame, progress=_progress(len(frames)))
        finally:
            del loaded  # drop this reference first, otherwise release() cannot free the weights
            if not model.get("keep_loaded", True):
                runtime.release()
        splat = Types.SPLAT(**runtime.splat_tensors(out["steps"] if every_frame else [out["final"]]))
        report = json.dumps({"model": {k: info[k] for k in ("checkpoint", "device")}, "input_frames": int(images.shape[0]),
                             "frame_interval": frame_interval, **out["report"]}, indent=1)
        return io.NodeOutput(splat, report)


class S2TokUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="S2TokUnload",
            display_name="S2Tok Unload",
            category="loaders/s2tok",
            description="Frees the resident S2Tok model (GPU and RAM).",
            inputs=[io.Splat.Input("after", optional=True, tooltip="Optional: connect a splat so this runs after it")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was, "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class S2TokExtension(ComfyExtension):
    async def get_node_list(self):
        return [S2TokLoader, S2TokReconstruct, S2TokUnload]


async def comfy_entrypoint() -> S2TokExtension:
    paths.register()
    return S2TokExtension()
