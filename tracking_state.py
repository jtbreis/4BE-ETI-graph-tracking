"""Shared state and helpers for iterative graph relaxation trackers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assign import PathCandidate
from .load_data import BlockFrames, FramePositions

VELOCITY_VARIABLES = ("vx", "vy", "vz", "speed")


@dataclass
class UsedDetections:
    """Detections already assigned to an accepted track — excluded from later linking."""

    by_frame: dict[int, set[int]] = field(default_factory=dict)

    def is_used(self, frame_idx: int, particle_idx: int) -> bool:
        return particle_idx in self.by_frame.get(frame_idx, set())

    def mark(self, frame_idx: int, particle_idx: int) -> None:
        self.by_frame.setdefault(frame_idx, set()).add(int(particle_idx))

    def mark_path(
        self,
        block: BlockFrames,
        global_maps: list[list[int]],
        path: PathCandidate,
    ) -> None:
        for j, local_idx in enumerate(path.indices):
            layer = path.start_layer + j
            frame_idx = block.frame_indices[layer]
            global_idx = global_maps[layer][local_idx]
            self.mark(frame_idx, global_idx)

    def available_mask(self, frame_idx: int, fr: FramePositions, error_threshold: float) -> np.ndarray:
        if fr.error is None:
            raise ValueError("Frame error array required")
        mask = fr.error <= error_threshold
        used = self.by_frame.get(frame_idx)
        if used:
            idx = np.fromiter(used, dtype=np.int64, count=len(used))
            valid = idx[(idx >= 0) & (idx < fr.n)]
            mask[valid] = False
        return mask

    def matched_masks(self, frames: list[FramePositions]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for fi, fr in enumerate(frames):
            mask = np.zeros(fr.n, dtype=bool)
            for pi in self.by_frame.get(fi, set()):
                if 0 <= pi < fr.n:
                    mask[pi] = True
            out.append(mask)
        return out

    @property
    def total(self) -> int:
        return sum(len(s) for s in self.by_frame.values())


@dataclass
class VelocityPdf:
    variable: str
    values: np.ndarray
    lo: float
    hi: float
    mean: float
    std: float

    @classmethod
    def from_values(
        cls,
        values: np.ndarray,
        variable: str,
        percentile_lo: float,
        percentile_hi: float,
    ) -> VelocityPdf:
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return cls(variable=variable, values=arr, lo=-np.inf, hi=np.inf, mean=0.0, std=0.0)
        lo, hi = np.percentile(arr, [percentile_lo, percentile_hi])
        return cls(
            variable=variable,
            values=arr,
            lo=float(lo),
            hi=float(hi),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
        )

    def contains(self, velocity: np.ndarray) -> bool:
        comp = velocity_component(velocity, self.variable)
        return bool(self.lo <= comp <= self.hi)

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "n_samples": int(self.values.size),
            "lo": self.lo,
            "hi": self.hi,
            "mean": self.mean,
            "std": self.std,
        }


def velocity_component(velocity: np.ndarray, variable: str) -> float:
    v = np.asarray(velocity, dtype=np.float64).ravel()
    if variable == "vx":
        return float(v[0])
    if variable == "vy":
        return float(v[1])
    if variable == "vz":
        return float(v[2])
    if variable == "speed":
        return float(np.linalg.norm(v))
    raise ValueError(f"Unknown pdf_variable {variable!r}; expected one of {VELOCITY_VARIABLES}")


def segment_velocities_from_paths(
    block_frames: list[FramePositions],
    paths: list[PathCandidate],
    dt: float,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    out: list[np.ndarray] = []
    for path in paths:
        for k in range(len(path.indices) - 1):
            layer_a = path.start_layer + k
            layer_b = path.start_layer + k + 1
            pa = block_frames[layer_a].xyz()[path.indices[k]]
            pb = block_frames[layer_b].xyz()[path.indices[k + 1]]
            out.append((pb - pa) / dt)
    if not out:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(out)


def build_available_block(
    frames: list[FramePositions],
    start_frame: int,
    error_threshold: float,
    used: UsedDetections,
) -> tuple[BlockFrames, list[list[int]]]:
    """Build a 4-frame block from unmatched detections with error <= threshold."""
    frame_indices = [start_frame + i for i in range(4)]
    block_frames: list[FramePositions] = []
    global_maps: list[list[int]] = []
    for fi in frame_indices:
        fr = frames[fi]
        avail = used.available_mask(fi, fr, error_threshold)
        gidx = np.flatnonzero(avail)
        block_frames.append(
            FramePositions(
                frame_index=fi,
                x=fr.x[gidx],
                y=fr.y[gidx],
                z=fr.z[gidx],
                error=fr.error[gidx] if fr.error is not None else None,
            )
        )
        global_maps.append(gidx.astype(np.int64).tolist())
    return BlockFrames(frame_indices=frame_indices, frames=block_frames), global_maps


def frame_ranges_from_block_starts(starts: list[int]) -> list[list[int]]:
    return [list(range(s, s + 4)) for s in starts]


def block_starts_from_frame_ranges(frame_ranges: list[list[int]]) -> list[int]:
    return [min(fr) for fr in frame_ranges]


def serialize_frames(frames: list[FramePositions]) -> list[dict]:
    out: list[dict] = []
    for fr in frames:
        item: dict = {
            "frame_index": fr.frame_index,
            "x": fr.x,
            "y": fr.y,
            "z": fr.z,
        }
        if fr.error is not None:
            item["error"] = fr.error
        out.append(item)
    return out


def deserialize_frames(data: list[dict]) -> list[FramePositions]:
    return [
        FramePositions(
            frame_index=int(item["frame_index"]),
            x=item["x"],
            y=item["y"],
            z=item["z"],
            error=item.get("error"),
        )
        for item in data
    ]


def used_snapshot(used: UsedDetections) -> dict[int, list[int]]:
    return {fi: list(ids) for fi, ids in used.by_frame.items()}


def used_from_snapshot(data: dict[int, list[int]]) -> UsedDetections:
    used = UsedDetections()
    used.by_frame = {int(fi): set(ids) for fi, ids in data.items()}
    return used


def apply_marks(used: UsedDetections, marks: list[tuple[int, int]]) -> None:
    for frame_idx, particle_idx in marks:
        used.mark(frame_idx, particle_idx)


def block_costs_from_frames(
    frames: list[FramePositions],
    frame_ranges: list[list[int]],
) -> list[float]:
    counts = {fi: fr.n for fi, fr in enumerate(frames)}
    costs: list[float] = []
    for frame_range in frame_ranges:
        if not frame_range:
            costs.append(0.0)
            continue
        lo, hi = min(frame_range), max(frame_range)
        costs.append(float(sum(counts.get(j, 0) for j in range(lo, hi))))
    return costs
