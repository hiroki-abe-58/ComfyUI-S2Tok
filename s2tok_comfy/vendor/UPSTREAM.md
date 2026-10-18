# Vendored upstream code — CC BY-NC 4.0, non-commercial use only

Source: https://github.com/Applied-Intuition-Open-Source/S2Tok, commit `96e8f5885bc3ed9e4f5c5ffb2c66523d92447132`.

License: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, Copyright 2026 The S2Tok Authors; see
`LICENSE` in this folder (the upstream file). Parts are derived from ZipSplat, DINOv2, Depth Anything 3 and timm (Apache-2.0); those
notices are in the source files, `NOTICE` and `THIRD_PARTY_LICENSES` (upstream files). The weights are licensed separately, also
CC BY-NC 4.0 (`WEIGHTS_LICENSE.md`, upstream file). This folder is **not** covered by the MIT license of the rest of the repository.

`scripts/vendor_sync.py` copies the files from an upstream checkout; `tests/test_vendor.py` compares every byte with the pinned commit
in CI, and `MANIFEST.sha256` lists the SHA-256 of every vendored file.

## Vendored unchanged (byte-identical)

- `s2tok/`: `__init__.py`, `admission.py`, `backbone.py`, `checkpoint.py`, `gaussians.py`,
  `modules/{__init__,attention,blocks,patch_embed,rope,rope3d}.py`
- `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES`, `WEIGHTS_LICENSE.md`

## Changed: `s2tok/model.py` (indicated at the top of the file)

`S2Tok.step(..., precision="bf16")` enabled only PyTorch's FlashAttention SDPA kernel for the backbone. PyTorch builds without that
kernel, among them the official Windows wheels, then fail with `RuntimeError: No available kernel` (the official `demo.py --precision
bf16` fails the same way on Windows; measured with torch 2.14.1+cu130). The change enables FlashAttention **and** the memory-efficient
kernel; PyTorch's priority order still uses FlashAttention wherever it exists, so other platforms are unaffected, and fp32 is not
touched at all. It is the change proposed upstream in https://github.com/Applied-Intuition-Open-Source/S2Tok/pull/1, byte for byte,
plus the one-line notice. The test compares the syntax trees and allows a difference only in `S2Tok.step`.

The node pack imports the submodules under the package name `s2tok` (`s2tok_comfy/runtime.py: vendored()`); the package's
`__init__.py`, which only re-exports names, is vendored but not executed.

## Not vendored

- `s2tok/viewer.py` (the interactive viser web viewer of the demo; nothing in the package imports it), `demo.py`, `tools/`,
  `assets/`. ComfyUI's own splat nodes render, preview and save the result instead.
