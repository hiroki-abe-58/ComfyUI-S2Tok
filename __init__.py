"""S2Tok for ComfyUI (unofficial integration of https://github.com/Applied-Intuition-Open-Source/S2Tok; non-commercial)."""

import logging

try:
    from .s2tok_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[S2Tok] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
