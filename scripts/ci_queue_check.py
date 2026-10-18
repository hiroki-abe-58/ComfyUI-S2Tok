"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed, and run the
shipped workflows through POST /prompt with a tiny random S2Tok in the official checkpoint format.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> [--workdir DIR] [--keep-pack]

The trained model (6.9 GB) cannot run in CI. The upstream model builds a fixed ViT-Giant backbone, so the check also installs a small
CI-only package (`zz_s2tok_ci_tiny_backbone`) that makes the vendored code build the tiny test backbone inside the server; it is removed
again at the end. This proves that the pack installs into a stock ComfyUI, that Load Video -> S2Tok Reconstruct -> Create 3D File ->
Save Splat and the render / progress-video workflows execute through the queue, that the saved .ply is reproducible, and that Unload
works. The trained-model runs are in docs/VERIFICATION.md. The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import queue_client  # noqa: E402
from sample_video import make_sample_video  # noqa: E402
from tiny_model import BACKBONE, make_tiny_checkpoint  # noqa: E402

NODE_IDS = {"S2TokLoader", "S2TokReconstruct", "S2TokUnload"}
HOOK = '''"""CI only (written by ComfyUI-S2Tok/scripts/ci_queue_check.py): build the tiny test backbone instead of ViT-Giant."""
import functools
import sys
import types

package = types.ModuleType("s2tok")
package.__path__ = [{vendor!r}]
sys.modules["s2tok"] = package
import s2tok.backbone as backbone  # noqa: E402

backbone.DinoVisionTransformer = functools.partial(backbone.DinoVisionTransformer, **{backbone!r})
NODE_CLASS_MAPPINGS = {{}}
'''


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="s2tok-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-S2Tok"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    hook = comfy / "custom_nodes" / "zz_s2tok_ci_tiny_backbone"
    hook.mkdir(exist_ok=True)
    (hook / "__init__.py").write_text(HOOK.format(vendor=str(pack / "s2tok_comfy" / "vendor" / "s2tok"), backbone=BACKBONE), encoding="utf-8")
    make_tiny_checkpoint(work / "models" / "s2tok" / "s2tok_252.pt")
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"s2tok_ci:\n  base_path: {(work / 'models').as_posix()}\n  s2tok: s2tok\n", encoding="utf-8")
    make_sample_video(comfy / "input" / "example.mp4")
    output = work / "output"

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(name: str, **overrides) -> dict | None:
            wf = json.loads((ROOT / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            result = queue_client.run(server, wf, None, 900)
            print(json.dumps({"workflow": name, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": result.get("error")}))
            if result["status"] != "success":
                failures.append(f"{name}: {result.get('error') or result['status']}")
                return None
            return result

        # scene, progress video, scene again: the progress workflow changes the inputs of node 4, so ComfyUI's cache cannot serve the
        # second scene run and the reconstruction executes again
        plys = []
        if run("reconstruct_video", **{"4.frame_interval": 1, "6.filename_prefix": "ci/scene_a", "8.width": 256, "8.height": 256}):
            plys += sorted((output / "ci").glob("scene_a*.ply"))
        if run("reconstruction_progress", **{"4.frame_interval": 1, "6.width": 128, "6.height": 128}) and not list((output / "s2tok").glob("progress*")):
            failures.append("no progress video saved")
        if run("reconstruct_video", **{"4.frame_interval": 1, "6.filename_prefix": "ci/scene_b", "8.width": 256, "8.height": 256}):
            plys += sorted((output / "ci").glob("scene_b*.ply"))
        if len(plys) != 2:
            failures.append(f"expected two saved .ply files, found {plys}")
        elif hashlib.sha256(plys[0].read_bytes()).digest() != hashlib.sha256(plys[1].read_bytes()).digest():
            failures.append("the two saved .ply files differ")
        if not list((output / "s2tok").glob("first_camera*.png")):
            failures.append("no render saved")
        unload = queue_client.run(server, {"1": {"class_type": "S2TokUnload", "inputs": {}}}, None, 120)
        print(json.dumps({"workflow": "unload", "status": unload["status"], "text": unload.get("text")}))
        if unload["status"] != "success":
            failures.append(f"unload: {unload.get('error')}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(hook, ignore_errors=True)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
