"""
Compare graph tracking vs legacy 4BE-ETI on the same 4-frame block.

Extracts tracks from both methods, matches them by position, and reports
counts, differences, and runtime.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
from python_4be_eti.particle_tracking_code import set_mesh_debug_match_prints
from python_4be_eti.particle_tracking_numba import (
    no_previous_tracks_3d,
    previous_tracks_3d,
)
from python_4be_eti.utils.basic_utils import load_data_h5

from .assign import PathCandidate, greedy_assign_paths
from .link_block import LinkParams, link_block
from .load_data import BlockFrames, load_block
from .progress import log, progress
from .track_data import block_to_tracking_data


@dataclass
class Track:
    """One track as frame index → (x, y, z)."""

    track_id: int
    points: dict[int, tuple[float, float, float]]  # frame -> xyz

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def frames(self) -> list[int]:
        return sorted(self.points.keys())


@dataclass
class MethodResult:
    name: str
    time_s: float
    tracks: list[Track]
    extra: dict = field(default_factory=dict)


@dataclass
class ComparisonReport:
    frame_range: list[int]
    graph: MethodResult
    legacy: MethodResult
    matched_pairs: list[tuple[int, int]]  # (graph_track_id, legacy_track_id)
    graph_only_ids: list[int]
    legacy_only_ids: list[int]
    max_position_error_mm: float  # worst matched-pair error

    @property
    def speedup_legacy_over_graph(self) -> float:
        if self.graph.time_s <= 0:
            return float("inf")
        return self.legacy.time_s / self.graph.time_s


def run_legacy_block(
    h5path: str,
    start: int,
    params: LinkParams,
    *,
    max_candidates_mesh: int | None = None,
    max_targets_mesh: int | None = None,
    progress_interval: int | None = None,
) -> tuple[object, float]:
    """Run legacy 4BE-ETI on one block; return (data, seconds)."""
    frame_range = [start + i for i in range(4)]
    folder = os.path.dirname(h5path)
    filename = os.path.basename(h5path)
    mc = max_candidates_mesh if max_candidates_mesh is not None else params.max_branch
    mt = max_targets_mesh if max_targets_mesh is not None else max(2000, params.max_branch * 2)
    ppi = (
        params.progress_interval
        if progress_interval is None
        else progress_interval
    )

    set_mesh_debug_match_prints(0)
    log(f"[legacy] Loading frames {frame_range} from HDF5...")
    t_load = time.perf_counter()
    data_range = load_data_h5(os.path.join(folder, filename), frame_range)
    log(
        f"[legacy] Loaded {len(data_range.x)} particles in "
        f"{time.perf_counter() - t_load:.1f}s"
    )

    t0 = time.perf_counter()
    frame_steps = list(range(min(frame_range), max(frame_range)))
    for step_idx, jj in enumerate(frame_steps, start=1):
        im_init = np.where(data_range.Slice == jj)[0]
        n_particles = len(im_init)
        log(
            f"[legacy] Frame {jj} ({step_idx}/{len(frame_steps)}): "
            f"{n_particles} particles"
        )
        idx_0 = np.where(data_range.Slice == jj)[0]
        idx_1 = np.where(data_range.Slice == jj + 1)[0]
        idx_2 = np.where(data_range.Slice == jj + 2)[0]
        idx_3 = np.where(data_range.Slice == jj + 3)[0]
        idx_m1 = (
            np.where(data_range.Slice == jj - 1)[0]
            if jj > min(frame_range)
            else None
        )
        t_frame = time.perf_counter()
        for ii in range(n_particles):
            progress(
                f"legacy f{jj}",
                ii + 1,
                n_particles,
                interval=ppi,
                t0=t_frame,
            )
            if data_range.Count[im_init[ii]] == 0:
                data_range = no_previous_tracks_3d(
                    data_range,
                    jj,
                    ii,
                    params.box_size_track,
                    params.initial_x_lo,
                    params.initial_x_hi,
                    params.initial_y_lo,
                    params.initial_y_hi,
                    params.initial_z_lo,
                    params.initial_z_hi,
                    max_candidates_mesh=mc,
                    max_targets_mesh=mt,
                    _im0=idx_0,
                    _im1=idx_1,
                    _im2=idx_2,
                    _im3=idx_3,
                )
            else:
                data_range = previous_tracks_3d(
                    data_range,
                    jj,
                    ii,
                    params.box_size_track,
                    max_candidates_mesh=mc,
                    max_targets_mesh=mt,
                    _im0=idx_m1,
                    _im1=idx_0,
                    _im2=idx_1,
                    _im3=idx_2,
                )
        if len(data_range.Count[data_range.Count == -1]) > 0:
            data_range.Count[data_range.Count == -1] = 0
        log(
            f"[legacy] Frame {jj} done in {time.perf_counter() - t_frame:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    n_tracks = len(np.unique(data_range.Count[data_range.Count > 0]))
    log(f"[legacy] Finished in {elapsed:.1f}s ({n_tracks} track ids assigned)")
    return data_range, elapsed


def run_graph_block(
    h5path: str,
    start: int,
    params: LinkParams,
) -> tuple[BlockFrames, list[PathCandidate], float, object]:
    """Run graph linker on one block; return (block, assigned, seconds, link_result)."""
    t0 = time.perf_counter()
    log(f"[graph] Loading block starting at frame {start}...")
    t_load = time.perf_counter()
    block = load_block(h5path, start)
    log(f"[graph] Loaded in {time.perf_counter() - t_load:.1f}s")
    result = link_block(block, params)
    log(f"[graph] Greedy assignment on {len(result.candidates)} candidates...")
    t_assign = time.perf_counter()
    assigned = greedy_assign_paths(result.candidates)
    log(
        f"[graph] Assigned {len(assigned)} tracks in "
        f"{time.perf_counter() - t_assign:.1f}s"
    )
    elapsed = time.perf_counter() - t0
    log(f"[graph] Finished in {elapsed:.1f}s")
    return block, assigned, elapsed, result


def extract_tracks_from_data(
    data,
    frame_range: list[int],
    *,
    min_length: int = 1,
) -> list[Track]:
    """Build Track list from a 4BE-ETI ``data`` object."""
    frame_set = set(frame_range)
    tracks: dict[int, dict[int, tuple[float, float, float]]] = {}
    x = np.asarray(data.x).ravel()
    y = np.asarray(data.y).ravel()
    z = np.asarray(data.z).ravel()
    slices = np.asarray(data.Slice).ravel()
    counts = np.asarray(data.Count).ravel()

    for i in range(counts.size):
        tid = int(counts[i])
        fr = int(slices[i])
        if tid == 0 or fr not in frame_set:
            continue
        tracks.setdefault(tid, {})[fr] = (float(x[i]), float(y[i]), float(z[i]))

    out = [
        Track(track_id=tid, points=pts)
        for tid, pts in sorted(tracks.items())
        if len(pts) >= min_length
    ]
    return out


def extract_tracks_from_graph(
    block: BlockFrames,
    assigned: list[PathCandidate],
) -> list[Track]:
    """Build Track list from graph path assignments."""
    frames_list = block.frames
    tracks: list[Track] = []
    for tid, path in enumerate(assigned, start=1):
        pts: dict[int, tuple[float, float, float]] = {}
        for j, local_idx in enumerate(path.indices):
            layer = path.start_layer + j
            fr = frames_list[layer]
            pts[fr.frame_index] = (
                float(fr.x[local_idx]),
                float(fr.y[local_idx]),
                float(fr.z[local_idx]),
            )
        tracks.append(Track(track_id=tid, points=pts))
    return tracks


def _position_error(
    ta: Track,
    tb: Track,
    *,
    pos_tol: float,
) -> float | None:
    """Max |Δr| on shared frames, or None if no overlap."""
    common = set(ta.frames) & set(tb.frames)
    if not common:
        return None
    errs = []
    for fr in common:
        a = np.array(ta.points[fr])
        b = np.array(tb.points[fr])
        errs.append(float(np.linalg.norm(a - b)))
    return max(errs) if errs else None


def tracks_are_same(
    ta: Track,
    tb: Track,
    *,
    pos_tol: float = 0.05,
    require_same_length: bool = False,
) -> bool:
    common = set(ta.frames) & set(tb.frames)
    if require_same_length and ta.length != tb.length:
        return False
    if len(common) < min(ta.length, tb.length):
        return False
    for fr in common:
        a = np.array(ta.points[fr])
        b = np.array(tb.points[fr])
        if float(np.linalg.norm(a - b)) > pos_tol:
            return False
    return True


def compare_block(
    h5path: str,
    start: int,
    params: LinkParams,
    *,
    pos_tol: float = 0.05,
    legacy_min_length: int = 4,
    graph_min_length: int = 3,
    run_legacy: bool = True,
) -> ComparisonReport:
    """
    Run both methods on the same block and match tracks by position.

    Legacy only produces meaningful 4-frame tracks; set ``graph_min_length=4``
    for a strict apples-to-apples comparison.
    """
    log(f"[compare] One 4-frame block starting at frame {start}")
    block, assigned, t_graph, link_result = run_graph_block(h5path, start, params)
    graph_tracks_all = extract_tracks_from_graph(block, assigned)
    graph_tracks = [t for t in graph_tracks_all if t.length >= graph_min_length]

    if run_legacy:
        log("[compare] Starting legacy tracker...")
        legacy_data, t_legacy = run_legacy_block(h5path, start, params)
        legacy_tracks = extract_tracks_from_data(
            legacy_data,
            block.frame_indices,
            min_length=legacy_min_length,
        )
    else:
        t_legacy = 0.0
        legacy_tracks = []

    log("[compare] Matching tracks between methods...")
    matched: list[tuple[int, int]] = []
    used_legacy: set[int] = set()
    max_err = 0.0

    for gt in graph_tracks:
        best_lt = None
        best_err = None
        for lt in legacy_tracks:
            if lt.track_id in used_legacy:
                continue
            err = _position_error(gt, lt, pos_tol=pos_tol)
            if err is None or err > pos_tol:
                continue
            if len(set(gt.frames) & set(lt.frames)) < min(gt.length, lt.length):
                continue
            if best_err is None or err < best_err:
                best_err = err
                best_lt = lt
        if best_lt is not None:
            matched.append((gt.track_id, best_lt.track_id))
            used_legacy.add(best_lt.track_id)
            max_err = max(max_err, best_err or 0.0)

    graph_matched = {g for g, _ in matched}
    legacy_matched = {l for _, l in matched}

    return ComparisonReport(
        frame_range=block.frame_indices,
        graph=MethodResult(
            name="graph",
            time_s=t_graph,
            tracks=graph_tracks,
            extra={
                "n_candidates": link_result.stats.get("n_candidates", len(assigned)),
                "n_all_paths": len(assigned),
            },
        ),
        legacy=MethodResult(
            name="legacy",
            time_s=t_legacy,
            tracks=legacy_tracks,
        ),
        matched_pairs=matched,
        graph_only_ids=[t.track_id for t in graph_tracks if t.track_id not in graph_matched],
        legacy_only_ids=[
            t.track_id for t in legacy_tracks if t.track_id not in legacy_matched
        ],
        max_position_error_mm=max_err,
    )


def format_report(report: ComparisonReport) -> str:
    """Human-readable summary."""
    lines = [
        f"Frame range: {report.frame_range}",
        "",
        "Runtime:",
        f"  Graph:  {report.graph.time_s:.3f} s",
        f"  Legacy: {report.legacy.time_s:.3f} s",
        f"  Ratio (legacy/graph): {report.speedup_legacy_over_graph:.2f}x",
        "",
        "Track counts:",
        f"  Graph:  {len(report.graph.tracks)} (min length filter applied)",
        f"  Legacy: {len(report.legacy.tracks)}",
        f"  Matched pairs: {len(report.matched_pairs)}",
        f"  Graph-only: {len(report.graph_only_ids)}",
        f"  Legacy-only: {len(report.legacy_only_ids)}",
        f"  Max position error on matched (mm): {report.max_position_error_mm:.4f}",
    ]
    if report.graph_only_ids:
        lines.append(f"  Graph-only ids: {report.graph_only_ids[:20]}")
    if report.legacy_only_ids:
        lines.append(f"  Legacy-only ids: {report.legacy_only_ids[:20]}")
    return "\n".join(lines)
