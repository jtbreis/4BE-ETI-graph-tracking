"""
Enumerate candidate paths for one 4-frame block.

Supports:
  - 4-frame paths starting at frame 0  (f0 → f1 → f2 → f3)
  - 3-frame paths starting at frame 0  (f0 → f1 → f2)
  - 3-frame paths starting at frame 1  (f1 → f2 → f3)

Uses the same box gates as 4BE-ETI ``no_previous_tracks_3d``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import time

from .assign import PathCandidate, filter_bidirectional_four_frame
from .kinematics import (
    distance_xyz,
    predict_first_frame_position_xyz,
    predict_prev_constant_velocity_xyz,
    predict_third_frame_position_xyz,
)
from .load_data import BlockFrames, subsample_indices
from .progress import log, progress
from .spatial_grid import SpatialGrid


@dataclass
class LinkParams:
    """Search windows (mm) — mirrors FourFrameTracking / 05_4frame_tracking.py."""

    box_size_track: float = 0.2
    initial_x_lo: float = 0.0
    initial_x_hi: float = 3.5
    initial_y_lo: float = -1.5
    initial_y_hi: float = 1.5
    initial_z_lo: float = -1.5
    initial_z_hi: float = 1.5
    max_branch: int = 30
    max_seeds: int = 0  # 0 = all seeds on frame 0 / frame 1
    seed_rng: int = 0
    allow_3_frame: bool = True
    require_bidirectional: bool = False  # 4-frame paths must validate forward and backward
    progress_interval: int = 5000  # print every N seeds; 0 = quiet except phase lines


@dataclass
class BlockResult:
    """Output of linking one 4-frame block."""

    frame_indices: list[int]
    candidates: list[PathCandidate]
    n_seeds_tried: int = 0
    stats: dict = field(default_factory=dict)


def _cap(indices: np.ndarray, max_n: int) -> np.ndarray:
    if max_n <= 0 or indices.size <= max_n:
        return indices
    return indices[:max_n]


def _initial_box_offsets(params: LinkParams) -> tuple[float, ...]:
    return (
        min(params.initial_x_lo, params.initial_x_hi),
        max(params.initial_x_lo, params.initial_x_hi),
        min(params.initial_y_lo, params.initial_y_hi),
        max(params.initial_y_lo, params.initial_y_hi),
        min(params.initial_z_lo, params.initial_z_hi),
        max(params.initial_z_lo, params.initial_z_hi),
    )


def _query_initial_box(
    grid: SpatialGrid,
    p0: np.ndarray,
    offsets: tuple[float, ...],
    max_branch: int,
) -> np.ndarray:
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = offsets
    return _cap(
        grid.query_aabb(
            p0[0] + x_lo,
            p0[0] + x_hi,
            p0[1] + y_lo,
            p0[1] + y_hi,
            p0[2] + z_lo,
            p0[2] + z_hi,
        ),
        max_branch,
    )


def _paths_from_frame0(
    block: BlockFrames,
    params: LinkParams,
    seed_ids: np.ndarray,
    grid1: SpatialGrid,
    grid2: SpatialGrid,
    grid3: SpatialGrid,
    offsets: tuple[float, ...],
    *,
    progress_interval: int = 0,
    progress_label: str = "graph f0 seeds",
) -> tuple[list[PathCandidate], list[int]]:
    """4-frame and 3-frame (f0→f1→f2) paths seeded on frame 0."""
    f0, f1, f2, f3 = block.frames
    b = params.box_size_track
    mb = params.max_branch
    candidates: list[PathCandidate] = []
    branch_counts: list[int] = []
    n_seeds = int(seed_ids.size)
    t0 = time.perf_counter() if progress_interval >= 0 else None

    for si, i0 in enumerate(seed_ids, start=1):
        progress(progress_label, si, n_seeds, interval=progress_interval, t0=t0)
        p0 = np.array([f0.x[i0], f0.y[i0], f0.z[i0]])
        ind1 = _query_initial_box(grid1, p0, offsets, mb)
        if ind1.size == 0:
            continue

        n_branches = 0
        for i1 in ind1:
            p1 = np.array([f1.x[i1], f1.y[i1], f1.z[i1]])
            pred2 = 2.0 * p1 - p0

            ind2 = _cap(grid2.query_center_box(pred2[0], pred2[1], pred2[2], b), mb)
            if ind2.size == 0:
                continue

            for i2 in ind2:
                p2 = np.array([f2.x[i2], f2.y[i2], f2.z[i2]])

                if params.allow_3_frame:
                    cost2 = distance_xyz(pred2, p2)
                    candidates.append(
                        PathCandidate(
                            start_layer=0,
                            indices=(int(i0), int(i1), int(i2)),
                            cost=cost2,
                        )
                    )
                    n_branches += 1

                pred3 = predict_third_frame_position_xyz(p0, p1, p2)
                ind3 = _cap(grid3.query_center_box(pred3[0], pred3[1], pred3[2], b), mb)
                if ind3.size == 0:
                    continue

                for i3 in ind3:
                    p3 = np.array([f3.x[i3], f3.y[i3], f3.z[i3]])
                    cost3 = distance_xyz(pred3, p3)
                    candidates.append(
                        PathCandidate(
                            start_layer=0,
                            indices=(int(i0), int(i1), int(i2), int(i3)),
                            cost=cost3,
                        )
                    )
                    n_branches += 1

        if n_branches:
            branch_counts.append(n_branches)

    return candidates, branch_counts


def _paths_from_frame1(
    block: BlockFrames,
    params: LinkParams,
    seed_ids: np.ndarray,
    grid2: SpatialGrid,
    grid3: SpatialGrid,
    offsets: tuple[float, ...],
    *,
    progress_interval: int = 0,
    progress_label: str = "graph f1 seeds",
) -> tuple[list[PathCandidate], list[int]]:
    """3-frame paths f1 → f2 → f3 seeded on frame 1."""
    _, f1, f2, f3 = block.frames
    b = params.box_size_track
    mb = params.max_branch
    candidates: list[PathCandidate] = []
    branch_counts: list[int] = []
    n_seeds = int(seed_ids.size)
    t0 = time.perf_counter() if progress_interval >= 0 else None

    for si, i1 in enumerate(seed_ids, start=1):
        progress(progress_label, si, n_seeds, interval=progress_interval, t0=t0)
        p1 = np.array([f1.x[i1], f1.y[i1], f1.z[i1]])
        ind2 = _query_initial_box(grid2, p1, offsets, mb)
        if ind2.size == 0:
            continue

        n_branches = 0
        for i2 in ind2:
            p2 = np.array([f2.x[i2], f2.y[i2], f2.z[i2]])
            pred3 = 2.0 * p2 - p1

            ind3 = _cap(grid3.query_center_box(pred3[0], pred3[1], pred3[2], b), mb)
            if ind3.size == 0:
                continue

            for i3 in ind3:
                p3 = np.array([f3.x[i3], f3.y[i3], f3.z[i3]])
                cost = distance_xyz(pred3, p3)
                candidates.append(
                    PathCandidate(
                        start_layer=1,
                        indices=(int(i1), int(i2), int(i3)),
                        cost=cost,
                    )
                )
                n_branches += 1

        if n_branches:
            branch_counts.append(n_branches)

    return candidates, branch_counts


def _backward_box_offsets(offsets: tuple[float, ...]) -> tuple[float, ...]:
    """Flip initial-box offset signs (search opposite direction in time)."""
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = offsets
    return (
        min(-x_hi, -x_lo),
        max(-x_hi, -x_lo),
        min(-y_hi, -y_lo),
        max(-y_hi, -y_lo),
        min(-z_hi, -z_lo),
        max(-z_hi, -z_lo),
    )


def _paths_from_frame3(
    block: BlockFrames,
    params: LinkParams,
    seed_ids: np.ndarray,
    grid2: SpatialGrid,
    grid1: SpatialGrid,
    grid0: SpatialGrid,
    offsets: tuple[float, ...],
    *,
    progress_interval: int = 0,
    progress_label: str = "graph f3 seeds (backward)",
) -> tuple[list[PathCandidate], list[int]]:
    """4-frame paths f3 → f2 → f1 → f0 seeded on frame 3 (backward in time)."""
    f0, f1, f2, f3 = block.frames
    b = params.box_size_track
    mb = params.max_branch
    candidates: list[PathCandidate] = []
    branch_counts: list[int] = []
    n_seeds = int(seed_ids.size)
    t0 = time.perf_counter() if progress_interval >= 0 else None

    for si, i3 in enumerate(seed_ids, start=1):
        progress(progress_label, si, n_seeds, interval=progress_interval, t0=t0)
        p3 = np.array([f3.x[i3], f3.y[i3], f3.z[i3]])
        ind2 = _query_initial_box(grid2, p3, offsets, mb)
        if ind2.size == 0:
            continue

        n_branches = 0
        for i2 in ind2:
            p2 = np.array([f2.x[i2], f2.y[i2], f2.z[i2]])
            pred1 = predict_prev_constant_velocity_xyz(p2, p3)

            ind1 = _cap(grid1.query_center_box(pred1[0], pred1[1], pred1[2], b), mb)
            if ind1.size == 0:
                continue

            for i1 in ind1:
                p1 = np.array([f1.x[i1], f1.y[i1], f1.z[i1]])
                pred0 = predict_first_frame_position_xyz(p1, p2, p3)

                ind0 = _cap(grid0.query_center_box(pred0[0], pred0[1], pred0[2], b), mb)
                if ind0.size == 0:
                    continue

                for i0 in ind0:
                    p0 = np.array([f0.x[i0], f0.y[i0], f0.z[i0]])
                    cost0 = distance_xyz(pred0, p0)
                    candidates.append(
                        PathCandidate(
                            start_layer=0,
                            indices=(int(i0), int(i1), int(i2), int(i3)),
                            cost=cost0,
                        )
                    )
                    n_branches += 1

        if n_branches:
            branch_counts.append(n_branches)

    return candidates, branch_counts


def _enumerate_forward(
    block: BlockFrames,
    params: LinkParams,
    rng: np.random.Generator,
    offsets: tuple[float, ...],
    ppi: int,
) -> tuple[list[PathCandidate], list[int], dict]:
    """Forward path enumeration (frame 0 / frame 1 seeds)."""
    f0, f1, _, _ = block.frames
    b = params.box_size_track

    grid1 = SpatialGrid(f1.x, f1.y, f1.z, cell_size=max(b, 0.5))
    grid2 = SpatialGrid(block.frames[2].x, block.frames[2].y, block.frames[2].z, cell_size=max(b, 0.5))
    grid3 = SpatialGrid(block.frames[3].x, block.frames[3].y, block.frames[3].z, cell_size=max(b, 0.5))

    seeds0 = np.arange(f0.n, dtype=np.int64)
    if params.max_seeds > 0 and params.max_seeds < f0.n:
        seeds0 = subsample_indices(f0.n, params.max_seeds, rng)
    log(f"[graph] Enumerating forward paths from frame 0 ({len(seeds0)} seeds)...")

    c0, branches0 = _paths_from_frame0(
        block, params, seeds0, grid1, grid2, grid3, offsets,
        progress_interval=ppi,
        progress_label="graph f0 seeds",
    )

    candidates = list(c0)
    branches = list(branches0)
    n_seeds1 = 0

    if params.allow_3_frame and not params.require_bidirectional:
        seeds1 = np.arange(f1.n, dtype=np.int64)
        if params.max_seeds > 0 and params.max_seeds < f1.n:
            seeds1 = subsample_indices(f1.n, params.max_seeds, rng)
        log(f"[graph] Enumerating 3-frame paths from frame 1 ({len(seeds1)} seeds)...")
        c1, branches1 = _paths_from_frame1(
            block, params, seeds1, grid2, grid3, offsets,
            progress_interval=ppi,
            progress_label="graph f1 seeds",
        )
        candidates.extend(c1)
        branches.extend(branches1)
        n_seeds1 = int(seeds1.size)

    meta = {
        "n_seeds_frame0": int(seeds0.size),
        "n_seeds_frame1": n_seeds1,
    }
    return candidates, branches, meta


def _enumerate_backward(
    block: BlockFrames,
    params: LinkParams,
    rng: np.random.Generator,
    offsets: tuple[float, ...],
    ppi: int,
) -> tuple[list[PathCandidate], list[int], dict]:
    """Backward 4-frame enumeration seeded on frame 3."""
    f0, f1, f2, f3 = block.frames
    b = params.box_size_track
    bwd_offsets = _backward_box_offsets(offsets)

    grid2 = SpatialGrid(f2.x, f2.y, f2.z, cell_size=max(b, 0.5))
    grid1 = SpatialGrid(f1.x, f1.y, f1.z, cell_size=max(b, 0.5))
    grid0 = SpatialGrid(f0.x, f0.y, f0.z, cell_size=max(b, 0.5))

    seeds3 = np.arange(f3.n, dtype=np.int64)
    if params.max_seeds > 0 and params.max_seeds < f3.n:
        seeds3 = subsample_indices(f3.n, params.max_seeds, rng)
    log(f"[graph] Enumerating backward paths from frame 3 ({len(seeds3)} seeds)...")

    candidates, branches = _paths_from_frame3(
        block, params, seeds3, grid2, grid1, grid0, bwd_offsets,
        progress_interval=ppi,
        progress_label="graph f3 seeds (backward)",
    )
    meta = {"n_seeds_frame3": int(seeds3.size)}
    return candidates, branches, meta


def link_block(block: BlockFrames, params: LinkParams) -> BlockResult:
    """
    Build valid path candidates in one block.

    When ``require_bidirectional`` is True, only 4-frame paths that validate
    both forward (f0 seed) and backward (f3 seed, flipped initial box) are kept.

    Returns path candidates (not yet assigned). Run ``greedy_assign_paths`` next.
    """
    f0, f1, _, f3 = block.frames
    rng = np.random.default_rng(params.seed_rng)
    offsets = _initial_box_offsets(params)
    ppi = params.progress_interval

    log(
        f"[graph] Block {block.frame_indices}: "
        f"{f0.n}, {f1.n}, {block.frames[2].n}, {f3.n} particles/frame"
    )
    log("[graph] Building spatial grids...")

    candidates, branches, fwd_meta = _enumerate_forward(
        block, params, rng, offsets, ppi,
    )
    n_seeds_tried = fwd_meta["n_seeds_frame0"] + fwd_meta["n_seeds_frame1"]

    bwd_candidates: list[PathCandidate] = []
    n_bidir = 0
    n_fwd_only_4 = 0
    n_bwd_only_4 = 0

    if params.require_bidirectional:
        bwd_candidates, bwd_branches, bwd_meta = _enumerate_backward(
            block, params, rng, offsets, ppi,
        )
        branches.extend(bwd_branches)
        n_seeds_tried += bwd_meta["n_seeds_frame3"]

        fwd_4_keys = {
            c.indices
            for c in candidates
            if c.length == 4 and c.start_layer == 0
        }
        bwd_4_keys = {
            c.indices
            for c in bwd_candidates
            if c.length == 4 and c.start_layer == 0
        }
        n_bidir = len(fwd_4_keys & bwd_4_keys)
        n_fwd_only_4 = len(fwd_4_keys - bwd_4_keys)
        n_bwd_only_4 = len(bwd_4_keys - fwd_4_keys)

        candidates, _both = filter_bidirectional_four_frame(candidates, bwd_candidates)
        log(
            f"[graph] Bidirectional filter: {n_bidir} agree, "
            f"{n_fwd_only_4} forward-only, {n_bwd_only_4} backward-only (4-frame)"
        )

    log(f"[graph] Done: {len(candidates)} path candidates")

    n4 = sum(1 for c in candidates if c.length == 4)
    n3_f0 = sum(1 for c in candidates if c.length == 3 and c.start_layer == 0)
    n3_f1 = sum(1 for c in candidates if c.length == 3 and c.start_layer == 1)

    stats = {
        "n_candidates": len(candidates),
        "n_candidates_4frame": n4,
        "n_candidates_3frame_f0": n3_f0,
        "n_candidates_3frame_f1": n3_f1,
        "n_seeds_frame0": fwd_meta["n_seeds_frame0"],
        "n_seeds_frame1": fwd_meta["n_seeds_frame1"],
        "n_seeds_tried": n_seeds_tried,
        "mean_branches_per_seed": float(np.mean(branches)) if branches else 0.0,
        "require_bidirectional": params.require_bidirectional,
        "n_bidirectional_4frame": n_bidir,
        "n_forward_only_4frame": n_fwd_only_4,
        "n_backward_only_4frame": n_bwd_only_4,
        "n_candidates_backward": len(bwd_candidates),
    }
    if params.require_bidirectional:
        stats["n_seeds_frame3"] = bwd_meta["n_seeds_frame3"]

    return BlockResult(
        frame_indices=block.frame_indices,
        candidates=candidates,
        n_seeds_tried=n_seeds_tried,
        stats=stats,
    )
