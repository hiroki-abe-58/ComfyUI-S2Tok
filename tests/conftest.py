"""Test setup. References come from environment variables, never from this repository:

S2TOK_COMFYUI_DIR     ComfyUI checkout (v0.38.0 or later)
S2TOK_UPSTREAM_DIR    Applied-Intuition-Open-Source/S2Tok checkout at the commit in s2tok_comfy/vendor/UPSTREAM.md
S2TOK_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No trained weights are used: the model tests build a tiny random S2Tok in the official checkpoint format and run it on the CPU.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("S2TOK_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("S2TOK_COMFYUI_DIR")
UPSTREAM = _path("S2TOK_UPSTREAM_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (S2TOK_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (S2TOK_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "S2Tok checkout (S2TOK_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """<tmp>/s2tok/tiny_s2tok.pt; tests that load it run inside tiny_model.tiny_backbone()."""
    from tiny_model import make_tiny_checkpoint

    return make_tiny_checkpoint(tmp_path_factory.mktemp("models") / "s2tok" / "tiny_s2tok.pt")


@pytest.fixture(scope="session")
def tiny_backbone_active():
    from tiny_model import tiny_backbone

    with tiny_backbone():
        yield
