# Verification

All runs on one machine: Windows 11 Pro (build 26200), Intel Core Ultra 9 285K, 64 GB RAM, NVIDIA GeForce RTX 5090 (32 GB, driver
610.88); Python 3.12.13, torch 2.14.1+cu130, ComfyUI v0.38.0 (`6b747c0`). Model: `AppliedIntuitionResearch/S2Tok` `s2tok_252.pt` at
revision `ddfa40260b30b6d3a4f52e5c7a4cf8a3a2e18fe0` (SHA-256 `e1a63805…db7b`, equal to the Hub's LFS hash). Official code:
`Applied-Intuition-Open-Source/S2Tok` at `96e8f5885bc3ed9e4f5c5ffb2c66523d92447132`, run unchanged (`demo.py --no_viewer`). The
official requirements are unpinned (`torch>=2.5`, numpy, einops, pillow; opencv-python and viser only for video files and the viewer),
so the official runs used the same Python environment.

Test material: two CC BY-SA 4.0 videos by FakirNL on Wikimedia Commons (1920 x 1080, 30 fps, VP9 WebM): *Groei* (162 frames) and
*Pegasus* (276 frames), sources in `docs/images/README.md`; plus a 126 s CC BY-SA 4.0 library walkthrough by Asav (Theora, 59.94 fps,
7571 frames) for the long run. The videos are not part of the repository.

## 1. Against the official demo.py

The frames were decoded once with ComfyUI's own decoder (`VideoFromFile.get_components()`, what Load Video + Get Video Components give
the nodes), rounded to 8 bit and saved as PNG: every 4th frame of Groei (41 frames), every 6th of Pegasus (46 frames). The official
`demo.py` read those folders; the pack read the same frames.

| scene | official `scene.ply` (SHA-256, first 16) | Gaussians | pack in-process: `scene.ply` / `frames.json` | real queue: Save Splat `.ply` |
| --- | --- | --- | --- | --- |
| Groei, 41 frames | `163185a3de5b3701` | 82,976 | identical / identical | identical |
| Pegasus, 46 frames | `b946671465ba242a` | 83,776 | identical / identical | identical |

- *In-process* = `scripts/run_jobs.py` (the runtime behind the nodes, writing with the official `write_gaussian_ply`).
- *Real queue* = `workflows/api/reconstruct_video.json` through POST /prompt: Load Video (the original `.webm`) → Get Video Components →
  S2Tok Reconstruct (`frame_interval` 4 or 6) → Create 3D File (from Splat, ply) → Save Splat, i.e. written by **ComfyUI's** PLY writer.
  The first version differed from the official file in the quaternion fields by at most 1.2e-7 (one float32 step), because the official
  writer normalises the quaternions once more before writing; the pack now applies the same normalisation when it builds the `SPLAT`,
  and the files are byte-identical. `tests/test_nodes.py` checks that ComfyUI's writer and the official writer give the same bytes.
- The preprocessing (`runtime.model_frames`: 8-bit RGB, central square with offsets rounded down, Lanczos resize to 252) is compared
  with the official `center_crop_resize` function in CI for five frame shapes (`tests/test_runtime.py`).

## 2. bf16

The official `demo.py --precision bf16` fails on this machine:

```
  File "s2tok/modules/attention.py", line 59, in forward
    x = F.scaled_dot_product_attention(q, k, v)
RuntimeError: No available kernel. Aborting execution.
```

`S2Tok.step` enables only PyTorch's FlashAttention SDPA kernel for bf16, and the Windows wheels have none
(`torch.backends.cuda.is_flash_attention_available()` is False; the memory-efficient, cuDNN and math kernels work). With the
one-statement change proposed upstream (Applied-Intuition-Open-Source/S2Tok#1: enable FlashAttention and the memory-efficient kernel;
PyTorch's priority order FLASH, EFFICIENT, MATH, CUDNN still picks FlashAttention where it exists):

| | fp32 | bf16 |
| --- | --- | --- |
| official code, unchanged | Groei / Pegasus as above | `No available kernel` |
| official code + the change | byte-identical to the unchanged fp32 (both scenes) | 82,624 / 83,200 Gaussians (fp32: 82,976 / 83,776) |
| pack through the real queue (Groei) | identical to the official fp32 | byte-identical to the patched official bf16 |

Time for the 41 Groei frames through the queue: fp32 5.53 s, bf16 3.32 s; peak VRAM 6.85 / 6.78 GiB (the weights stay float32).

## 3. Other real-model runs

- 190 frames (every 40th frame of the library walkthrough, decoded as a stream with PyAV and saved as PNG; in-process): 42.9 s, peak
  VRAM 7.07 GiB, 113,856 Gaussians, 3558 of 4000 memory tokens after the last frame (1834 after 25 frames, 2390 after 50, 3367
  after 100).
- CPU, fp32, the first 10 Groei frames: 14.2 s (GPU: 1.22 s), the same 31,552 Gaussians; values differ by floating-point rounding
  (positions at most 0.008).
- Real queue: interrupt during a 276-frame job (status interrupted; the next job runs normally), keep-loaded off (the next Unload
  reports nothing loaded), Unload, bf16 with the CPU device (clear error before the model is loaded), every-frame output → Render Splat
  → Create Video → Save Video (`workflows/api/reconstruction_progress.json`).
- `docs/images`: renders from the first camera (Create Camera Info, quaternion mode, identity) and from two cameras 20° to each side
  of a point 0.5 units in front of it (look-at mode), 512 x 512.

## 4. A ComfyUI frame-count bug found on the way

Load Video → Video Frame Sample (uniform) → … failed on both test videos with `KeyError: 162` (Groei) because
`VideoFromFile.get_frame_count()` returns one frame too many for MKV and WebM files: it counts the empty flush packet that PyAV's
`demux()` yields at the end. Reproduced on ComfyUI master (2026-10-18) for VP9, VP8, AV1, H.264 and FFV1 in WebM / MKV (MP4 is
correct: its header has the frame count). Fix proposed: Comfy-Org/ComfyUI#17154 (with a regression test). For the Theora `.ogg`, the
metadata estimate is 7576 against 7571 decodable frames, which the same node also cannot handle; not addressed there.

## 5. Clean install

Fresh ComfyUI v0.38.0 checkout and Python 3.12 virtual environment (torch 2.14.1+cu130, ComfyUI's requirements, this pack's
`requirements.txt`), `git clone` of this repository into `custom_nodes`, the checkpoint downloaded with
`tools/download_model.py --models-dir ComfyUI/models/s2tok` (6.9 GB, 123 s including the SHA-256 check). `workflows/api/reconstruct_video.json`
and `reconstruction_progress.json` ran unchanged on an `example.mp4` (the Groei clip transcoded to H.264 with ComfyUI's `save_to`), and
`reconstruct_video.json` on the original Groei `.webm` (`frame_interval` 4) saved a `.ply` byte-identical to the official `scene.ply`.
`comfy node validate` passes; the archive of `comfy node pack` has 37 files (200 kB) and no environment-variable, network or process
code. The pack is not published to the Comfy Registry (the code and weights it runs are non-commercial).

## 6. Automated tests (CI)

`tests/` (29 tests) runs on the CPU with a tiny random S2Tok in the official checkpoint format (`save_checkpoint` / `load_model` of the
vendored code; the ViT-Giant backbone is swapped for a 6-layer one while the tests run): preprocessing against `demo.py`, loading
(including inside `torch.inference_mode()`), streaming (determinism, per-frame report, a full memory, padding of per-frame batches),
the nodes as ComfyUI loads them (registration, checkpoint listing and path containment, ComfyUI's validation of the shipped workflows,
frame interval / max frames, every-frame output, ComfyUI's PLY writer = official writer, keep-loaded, unload, interrupt, bf16 on the
CPU), the vendored code against the upstream commit (bytes, and syntax trees for the one changed method), and the absence of
environment-variable, network and process code in the package. `scripts/ci_queue_check.py` starts a real ComfyUI on the CPU, installs a
CI-only package that switches the backbone to the tiny one, and runs both shipped workflows through the queue (a second scene run after
the progress run must give a byte-identical `.ply`). CI matrix: Ubuntu 24.04 and Windows with Python 3.12 and torch 2.14.1, and Ubuntu with Python 3.10 and torch 2.8.0; all
three jobs pass.

## ComfyUI v0.41.0

Since 2026-10-19 the CI also runs the test suite and the real-queue check against ComfyUI v0.41.0 (commit
`98cc0bf84b7cb1383654c5b402e7612d875d6d22`, Ubuntu 24.04, Python 3.12 / torch 2.14.1), next to v0.38.0, the release the
real-model checks above used. Both pass.
