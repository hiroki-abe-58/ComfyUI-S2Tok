"""Compare two binary 3DGS .ply files (e.g. the official demo.py scene.ply and one saved through ComfyUI): byte equality, then the
vertex properties field by field.

    python scripts/compare_ply.py a.ply b.ply
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


def read_ply(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: not a binary little-endian ply")
    names = [line.split()[2] for line in header if line.startswith("property float ")]
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex "))
    return np.frombuffer(data, dtype=np.dtype([(n, "<f4") for n in names]), count=count, offset=end)


def main() -> int:
    a_path, b_path = sys.argv[1], sys.argv[2]
    same_bytes = hashlib.sha256(open(a_path, "rb").read()).digest() == hashlib.sha256(open(b_path, "rb").read()).digest()
    a, b = read_ply(a_path), read_ply(b_path)
    result = {"identical_bytes": same_bytes, "vertices": [len(a), len(b)], "same_properties": a.dtype.names == b.dtype.names}
    if len(a) == len(b) and a.dtype.names == b.dtype.names:
        diffs = {n: float(np.abs(a[n].astype(np.float64) - b[n].astype(np.float64)).max()) for n in a.dtype.names}
        result["max_abs_difference"] = max(diffs.values())
        result["fields_that_differ"] = {n: d for n, d in diffs.items() if d > 0}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
