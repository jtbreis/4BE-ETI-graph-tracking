"""Load stereomatched particle positions from rays_out_cpp.h5."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class FramePositions:
    """Particles in one time step."""

    frame_index: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def n(self) -> int:
        return int(self.x.size)

    def xyz(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.z])


@dataclass
class BlockFrames:
    """Four consecutive frames (one independent 4-frame tracking block)."""

    frame_indices: list[int]
    frames: list[FramePositions]

    def __len__(self) -> int:
        return len(self.frames)


def get_nframes(h5path: str) -> int:
    with h5py.File(h5path, "r") as f:
        keys = [k for k in f.keys() if k.startswith("frame")]
    return len(keys)


def load_frame(h5path: str, frame_index: int) -> FramePositions:
    key = f"frame{frame_index:05d}"
    with h5py.File(h5path, "r") as f:
        xyze = np.array(f[key]["xyze"])
    return FramePositions(
        frame_index=frame_index,
        x=np.asarray(xyze[0], dtype=np.float64).ravel(),
        y=np.asarray(xyze[1], dtype=np.float64).ravel(),
        z=np.asarray(xyze[2], dtype=np.float64).ravel(),
    )


def load_block(h5path: str, start_frame: int) -> BlockFrames:
    """Load frames [start, start+1, start+2, start+3]."""
    indices = [start_frame + i for i in range(4)]
    frames = [load_frame(h5path, fi) for fi in indices]
    return BlockFrames(frame_indices=indices, frames=frames)


def iter_block_starts(num_frames: int, track_length: int = 4) -> list[int]:
    """Non-overlapping block starts: 0, 4, 8, ..."""
    return list(range(0, num_frames, track_length))


def subsample_indices(n: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick up to max_count indices from range(n)."""
    if max_count <= 0 or max_count >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, max_count, replace=False))
