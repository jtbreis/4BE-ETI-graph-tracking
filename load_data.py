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
    error: np.ndarray | None = None

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


def load_frame(
    h5path: str,
    frame_index: int,
    *,
    with_error: bool = False,
    mask: np.ndarray | None = None,
) -> FramePositions:
    key = f"frame{frame_index:05d}"
    with h5py.File(h5path, "r") as f:
        xyze = np.array(f[key]["xyze"])
    err = None
    if with_error and xyze.shape[0] >= 4:
        err = np.asarray(xyze[3], dtype=np.float64).ravel()
    x = np.asarray(xyze[0], dtype=np.float64).ravel()
    y = np.asarray(xyze[1], dtype=np.float64).ravel()
    z = np.asarray(xyze[2], dtype=np.float64).ravel()
    if mask is not None:
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        x, y, z = x[idx], y[idx], z[idx]
        if err is not None:
            err = err[idx]
    return FramePositions(
        frame_index=frame_index,
        x=x,
        y=y,
        z=z,
        error=err,
    )


def load_all_frames(h5path: str, *, with_error: bool = False) -> list[FramePositions]:
    nframes = get_nframes(h5path)
    return [load_frame(h5path, fi, with_error=with_error) for fi in range(nframes)]


def load_all_errors(h5path: str) -> list[np.ndarray]:
    """Load stereomatching error row per frame (no positions)."""
    nframes = get_nframes(h5path)
    errors: list[np.ndarray] = []
    with h5py.File(h5path, "r") as f:
        for fi in range(nframes):
            key = f"frame{fi:05d}"
            xyze = f[key]["xyze"]
            if xyze.shape[0] < 4:
                raise ValueError(f"STM xyze dataset must include error row (4 x N) in {key}")
            errors.append(np.asarray(xyze[3], dtype=np.float64).ravel())
    return errors


def particle_counts_from_masks(masks: list[np.ndarray]) -> dict[int, int]:
    return {fi: int(np.sum(mask)) for fi, mask in enumerate(masks)}


def load_block(
    h5path: str,
    start_frame: int,
    *,
    masks: list[np.ndarray] | None = None,
) -> BlockFrames:
    """Load frames [start, start+1, start+2, start+3]."""
    indices = [start_frame + i for i in range(4)]
    if masks is not None:
        frames = [
            load_frame(h5path, fi, mask=masks[fi])
            for fi in indices
        ]
    else:
        frames = [load_frame(h5path, fi) for fi in indices]
    return BlockFrames(frame_indices=indices, frames=frames)


def iter_block_starts(num_frames: int, track_length: int = 4) -> list[int]:
    """Non-overlapping block starts: 0, 4, 8, ..."""
    return list(range(0, num_frames, track_length))


def block_starts_for_nframes(nframes: int, *, max_blocks: int = 0) -> list[int]:
    """4-frame block start indices; optionally cap count for smoke tests."""
    starts = iter_block_starts(nframes)
    if max_blocks > 0:
        starts = starts[:max_blocks]
    return starts


def subsample_indices(n: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick up to max_count indices from range(n)."""
    if max_count <= 0 or max_count >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, max_count, replace=False))
