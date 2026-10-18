"""A small synthetic video (a textured plane seen by a sliding camera) for workflow validation and the CPU queue check."""

from pathlib import Path

import av
import numpy as np


def make_sample_video(path: Path, frames: int = 12, size: tuple[int, int] = (160, 120)) -> Path:
    w, h = size
    rng = np.random.default_rng(0)
    texture = (rng.random((h, w + 4 * frames, 3)) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(texture[:, 4 * i: 4 * i + w]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
