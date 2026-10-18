"""Model location: `<ComfyUI models>/s2tok/*.pt` (plus any path added for `s2tok` in extra_model_paths.yaml). Workflows choose a file
from this list only; nothing is downloaded at run time."""

from __future__ import annotations

import os

FOLDER = "s2tok"


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def checkpoints() -> list[str]:
    import folder_paths

    return sorted(name for name in folder_paths.get_filename_list(FOLDER) if name.lower().endswith(".pt"))


def checkpoint_path(name: str) -> str:
    """The file of a listed name, resolved by ComfyUI inside the registered folders."""
    import folder_paths

    if name not in checkpoints():
        raise FileNotFoundError(f"checkpoint {name!r} was not found in models/{FOLDER} (see the README for the download command)")
    return folder_paths.get_full_path_or_raise(FOLDER, name)
