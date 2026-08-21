"""
Build-once candidate graphs and filter edges per relaxation pass.

Enumerate path candidates once at maximum permissiveness, annotate each path with
kinematic metadata, then progressively drop connections that fail tighter criteria
and re-run greedy disjoint assignment — without rebuilding the graph.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import numpy as np

from .assign import PathCandidate, greedy_assign_paths
from .link_block import LinkParams, link_block
from .load_data import BlockFrames, FramePositions, block_starts_for_nframes
from .tracking_state import (
    UsedDetections,
    VelocityPdf,
    apply_marks,
    block_costs_from_frames,
    block_starts_from_frame_ranges,
    build_available_block,
    deserialize_frames,
    frame_ranges_from_block_starts,
    segment_velocities_from_paths,
    serialize_frames,
    used_from_snapshot,
    used_snapshot,
    velocity_component,
)


@dataclass
class AcceptedPathStore:
    """Cumulative accepted paths per 4-frame block (local block indices)."""

    by_block: dict[int, list[PathCandidate]] = field(default_factory=dict)

    def extend_block(self, block_start: int, paths: list[PathCandidate]) -> None:
        if paths:
            self.by_block.setdefault(int(block_start), []).extend(paths)

    def extend_from_updates(self, updates: dict[int, list[dict]]) -> None:
        for start, items in updates.items():
            self.extend_block(int(start), paths_from_updates(items))

    def total_tracks(self) -> int:
        return sum(len(paths) for paths in self.by_block.values())


def paths_from_updates(updates: list[dict]) -> list[PathCandidate]:
    return [
        PathCandidate(
            start_layer=int(item["start_layer"]),
            indices=tuple(int(i) for i in item["indices"]),
            cost=float(item.get("cost", 0.0)),
            hop_distances=tuple(float(x) for x in item.get("hop_distances", ())),
        )
        for item in updates
    ]


@dataclass
class AnnotatedPath:
    path: PathCandidate
    max_error: float
    max_hop_distance: float
    segment_velocities: np.ndarray
    disp_f0f1: np.ndarray | None
    disp_f1f2: np.ndarray | None


@dataclass
class BlockCandidateGraph:
    block_start: int
    block: BlockFrames
    global_maps: list[list[int]]
    paths: list[AnnotatedPath]
    n_candidates_raw: int


@dataclass
class PassCriteria:
    max_error: float | None = None
    max_box_size: float | None = None
    velocity_variable: str = "vx"
    velocity_bands: list[tuple[float, float]] | None = None
    global_pdf: VelocityPdf | None = None
    initial_offsets: tuple[float, float, float, float, float, float] | None = None
    apply_velocity_gate: bool = True
    block_refs: dict[int, dict] | None = None  # block_start -> BlockReference.to_dict()


def _path_max_error(block_frames: list[FramePositions], path: PathCandidate) -> float:
    errors: list[float] = []
    for j, local_idx in enumerate(path.indices):
        layer = path.start_layer + j
        fr = block_frames[layer]
        if fr.error is not None:
            errors.append(float(fr.error[local_idx]))
    return max(errors) if errors else 0.0


def _path_max_hop_distance(path: PathCandidate) -> float:
    if path.hop_distances:
        return float(max(path.hop_distances))
    return 0.0


def _path_displacements(
    block_frames: list[FramePositions],
    path: PathCandidate,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    disp_f0f1: np.ndarray | None = None
    disp_f1f2: np.ndarray | None = None
    idx = path.indices
    layers = [path.start_layer + j for j in range(len(idx))]
    if path.start_layer == 0 and len(idx) >= 2:
        p0 = block_frames[layers[0]].xyz()[idx[0]]
        p1 = block_frames[layers[1]].xyz()[idx[1]]
        disp_f0f1 = p1 - p0
    if path.start_layer <= 1 and len(idx) >= 2:
        local_a = max(0, 1 - path.start_layer)
        local_b = local_a + 1
        if local_b < len(idx):
            pa = block_frames[layers[local_a]].xyz()[idx[local_a]]
            pb = block_frames[layers[local_b]].xyz()[idx[local_b]]
            disp_f1f2 = pb - pa
    return disp_f0f1, disp_f1f2


def annotate_path(
    block_frames: list[FramePositions],
    path: PathCandidate,
    dt: float,
) -> AnnotatedPath:
    vels = segment_velocities_from_paths(block_frames, [path], dt)
    disp_f0f1, disp_f1f2 = _path_displacements(block_frames, path)
    return AnnotatedPath(
        path=path,
        max_error=_path_max_error(block_frames, path),
        max_hop_distance=_path_max_hop_distance(path),
        segment_velocities=vels,
        disp_f0f1=disp_f0f1,
        disp_f1f2=disp_f1f2,
    )


def _displacement_in_offsets(
    disp: np.ndarray,
    offsets: tuple[float, float, float, float, float, float],
) -> bool:
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = offsets
    return bool(
        x_lo <= disp[0] <= x_hi
        and y_lo <= disp[1] <= y_hi
        and z_lo <= disp[2] <= z_hi
    )


def _velocities_match_bands(
    velocities: np.ndarray,
    bands: list[tuple[float, float]],
    variable: str,
) -> bool:
    if velocities.size == 0:
        return False
    for v in velocities:
        comp = velocity_component(v, variable)
        if not any(lo <= comp <= hi for lo, hi in bands):
            return False
    return True


def path_matches_criteria(
    ap: AnnotatedPath,
    criteria: PassCriteria,
    *,
    block_start: int | None = None,
) -> bool:
    velocity_bands = criteria.velocity_bands
    initial_offsets = criteria.initial_offsets
    if criteria.block_refs and block_start is not None:
        ref = criteria.block_refs.get(int(block_start))
        if ref:
            if ref.get("velocity_bands"):
                velocity_bands = [tuple(b) for b in ref["velocity_bands"]]
            if ref.get("initial_offsets") is not None:
                initial_offsets = tuple(ref["initial_offsets"])

    if criteria.max_error is not None and ap.max_error > criteria.max_error + 1e-12:
        return False
    if criteria.max_box_size is not None and ap.max_hop_distance > criteria.max_box_size + 1e-12:
        return False
    if initial_offsets is not None:
        ok = True
        if ap.disp_f0f1 is not None:
            ok = ok and _displacement_in_offsets(ap.disp_f0f1, initial_offsets)
        if ap.disp_f1f2 is not None:
            ok = ok and _displacement_in_offsets(ap.disp_f1f2, initial_offsets)
        if not ok:
            return False
    if not criteria.apply_velocity_gate:
        return True
    if criteria.global_pdf is not None:
        if ap.segment_velocities.size == 0:
            return False
        return all(criteria.global_pdf.contains(v) for v in ap.segment_velocities)
    if velocity_bands:
        return _velocities_match_bands(
            ap.segment_velocities, velocity_bands, criteria.velocity_variable
        )
    return True


def path_is_available(
    path: PathCandidate,
    block: BlockFrames,
    global_maps: list[list[int]],
    used: UsedDetections,
) -> bool:
    for j, local_idx in enumerate(path.indices):
        layer = path.start_layer + j
        frame_idx = block.frame_indices[layer]
        global_idx = global_maps[layer][local_idx]
        if used.is_used(frame_idx, global_idx):
            return False
    return True


def assign_pass_on_block(
    graph: BlockCandidateGraph,
    used: UsedDetections,
    criteria: PassCriteria,
) -> tuple[list[PathCandidate], int, int]:
    """Filter pre-built graph edges, assign disjoint paths, reserve detections."""
    kept: list[PathCandidate] = []
    rejected = 0
    for ap in graph.paths:
        if not path_is_available(ap.path, graph.block, graph.global_maps, used):
            continue
        if not path_matches_criteria(ap, criteria, block_start=graph.block_start):
            rejected += 1
            continue
        kept.append(ap.path)
    assigned = greedy_assign_paths(kept)
    for path in assigned:
        used.mark_path(graph.block, graph.global_maps, path)
    return assigned, graph.n_candidates_raw, rejected


def build_block_candidate_graph(
    block: BlockFrames,
    global_maps: list[list[int]],
    link_params: LinkParams,
    dt: float,
    *,
    block_start: int,
) -> BlockCandidateGraph:
    result = link_block(block, link_params)
    paths = [annotate_path(block.frames, cand, dt) for cand in result.candidates]
    return BlockCandidateGraph(
        block_start=block_start,
        block=block,
        global_maps=global_maps,
        paths=paths,
        n_candidates_raw=len(result.candidates),
    )


def _graph_to_dict(graph: BlockCandidateGraph) -> dict:
    paths_out: list[dict] = []
    for ap in graph.paths:
        p = ap.path
        paths_out.append(
            {
                "start_layer": p.start_layer,
                "indices": list(p.indices),
                "cost": p.cost,
                "hop_distances": list(p.hop_distances),
                "max_error": ap.max_error,
                "max_hop_distance": ap.max_hop_distance,
                "segment_velocities": ap.segment_velocities.tolist(),
                "disp_f0f1": ap.disp_f0f1.tolist() if ap.disp_f0f1 is not None else None,
                "disp_f1f2": ap.disp_f1f2.tolist() if ap.disp_f1f2 is not None else None,
            }
        )
    return {
        "block_start": graph.block_start,
        "frame_indices": list(graph.block.frame_indices),
        "global_maps": graph.global_maps,
        "frames_data": serialize_frames(graph.block.frames),
        "paths": paths_out,
        "n_candidates_raw": graph.n_candidates_raw,
    }


def _graph_from_dict(data: dict) -> BlockCandidateGraph:
    frames = deserialize_frames(data["frames_data"])
    block = BlockFrames(frame_indices=data["frame_indices"], frames=frames)
    paths: list[AnnotatedPath] = []
    for item in data["paths"]:
        path = PathCandidate(
            start_layer=int(item["start_layer"]),
            indices=tuple(int(i) for i in item["indices"]),
            cost=float(item["cost"]),
            hop_distances=tuple(float(x) for x in item.get("hop_distances", ())),
        )
        vels = np.asarray(item.get("segment_velocities", []), dtype=np.float64)
        if vels.size and vels.ndim == 1:
            vels = vels.reshape(1, 3)
        disp0 = item.get("disp_f0f1")
        disp1 = item.get("disp_f1f2")
        paths.append(
            AnnotatedPath(
                path=path,
                max_error=float(item["max_error"]),
                max_hop_distance=float(item["max_hop_distance"]),
                segment_velocities=vels,
                disp_f0f1=np.asarray(disp0, dtype=np.float64) if disp0 is not None else None,
                disp_f1f2=np.asarray(disp1, dtype=np.float64) if disp1 is not None else None,
            )
        )
    return BlockCandidateGraph(
        block_start=int(data["block_start"]),
        block=block,
        global_maps=data["global_maps"],
        paths=paths,
        n_candidates_raw=int(data["n_candidates_raw"]),
    )


def _criteria_to_dict(criteria: PassCriteria) -> dict:
    pdf = None
    if criteria.global_pdf is not None:
        pdf = {
            "variable": criteria.global_pdf.variable,
            "lo": criteria.global_pdf.lo,
            "hi": criteria.global_pdf.hi,
        }
    return {
        "max_error": criteria.max_error,
        "max_box_size": criteria.max_box_size,
        "velocity_variable": criteria.velocity_variable,
        "velocity_bands": criteria.velocity_bands,
        "global_pdf": pdf,
        "initial_offsets": criteria.initial_offsets,
        "apply_velocity_gate": criteria.apply_velocity_gate,
        "block_refs": criteria.block_refs,
    }


def _criteria_from_dict(data: dict) -> PassCriteria:
    pdf = None
    if data.get("global_pdf"):
        item = data["global_pdf"]
        pdf = VelocityPdf(
            variable=item["variable"],
            values=np.empty(0, dtype=np.float64),
            lo=float(item["lo"]),
            hi=float(item["hi"]),
            mean=0.0,
            std=0.0,
        )
    bands = data.get("velocity_bands")
    return PassCriteria(
        max_error=data.get("max_error"),
        max_box_size=data.get("max_box_size"),
        velocity_variable=data.get("velocity_variable", "vx"),
        velocity_bands=[tuple(b) for b in bands] if bands else None,
        global_pdf=pdf,
        initial_offsets=tuple(data["initial_offsets"]) if data.get("initial_offsets") else None,
        apply_velocity_gate=bool(data.get("apply_velocity_gate", True)),
        block_refs=data.get("block_refs"),
    )


def _build_graphs_for_starts(
    block_starts: list[int],
    frames_data: list[dict],
    link_params_dict: dict,
    max_error: float,
    dt: float,
) -> dict[int, dict]:
    frames = deserialize_frames(frames_data)
    params = LinkParams(**link_params_dict)
    used = UsedDetections()
    out: dict[int, dict] = {}
    for start in sorted(block_starts):
        block, global_maps = build_available_block(frames, start, max_error, used)
        if all(fr.n == 0 for fr in block.frames):
            continue
        graph = build_block_candidate_graph(
            block, global_maps, params, dt, block_start=start
        )
        out[int(start)] = _graph_to_dict(graph)
    return out


def build_all_block_graphs(
    frames: list[FramePositions],
    starts: list[int],
    link_params: LinkParams,
    max_error: float,
    dt: float,
    *,
    workers: int = 1,
    verbose: bool = True,
    log_prefix: str = "graph",
) -> dict[int, BlockCandidateGraph]:
    if not starts:
        return {}

    workers = max(1, int(workers))
    params_dict = asdict(link_params)
    params_dict["quiet_linking"] = True
    frames_data = serialize_frames(frames)

    if verbose:
        print(
            f"[{log_prefix}] building candidate graphs for {len(starts)} block(s) "
            f"(error<={max_error:.4f}, box<={link_params.box_size_track:.4f})",
            flush=True,
        )

    graphs: dict[int, BlockCandidateGraph] = {}

    if workers <= 1 or len(starts) <= 1:
        built = _build_graphs_for_starts(starts, frames_data, params_dict, max_error, dt)
        graphs = {k: _graph_from_dict(v) for k, v in built.items()}
    else:
        from python_4be_eti.utils.basic_utils import balance_frame_ranges

        frame_ranges = frame_ranges_from_block_starts(starts)
        costs = block_costs_from_frames(frames, frame_ranges)
        batches, _ = balance_frame_ranges(frame_ranges, workers, costs=costs)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for batch in batches:
                batch_starts = block_starts_from_frame_ranges(batch)
                fut = executor.submit(
                    _build_graphs_for_starts,
                    batch_starts,
                    frames_data,
                    params_dict,
                    max_error,
                    dt,
                )
                futures.append(fut)
            for fut in as_completed(futures):
                for k, v in fut.result().items():
                    graphs[int(k)] = _graph_from_dict(v)

    if verbose:
        total_edges = sum(g.n_candidates_raw for g in graphs.values())
        print(
            f"[{log_prefix}] candidate graphs ready: {len(graphs)} block(s), "
            f"{total_edges} path(s)",
            flush=True,
        )
    return graphs


def _relaxation_pass_batch(
    batch_idx: int,
    block_starts: list[int],
    graphs_data: dict[int, dict],
    used_snapshot_data: dict[int, list[int]],
    criteria_data: dict,
    dt: float,
    *,
    pass_label: str = "pass",
    log_prefix: str = "relax",
    verbose: bool = False,
) -> dict:
    used = used_from_snapshot(used_snapshot_data)
    initial_used = used_from_snapshot(used_snapshot_data)
    criteria = _criteria_from_dict(criteria_data)

    total_assigned = 0
    total_candidates = 0
    total_rejected = 0
    block_logs: list[str] = []
    block_path_updates: dict[int, list[dict]] = {}
    all_velocities: list[np.ndarray] = []

    for start in sorted(block_starts):
        graph_data = graphs_data.get(int(start))
        if graph_data is None:
            continue
        graph = _graph_from_dict(graph_data)
        assigned, n_cand, n_rej = assign_pass_on_block(graph, used, criteria)
        total_assigned += len(assigned)
        total_candidates += n_cand
        total_rejected += n_rej
        vels = segment_velocities_from_paths(graph.block.frames, assigned, dt)
        if vels.size:
            all_velocities.append(vels)
        if assigned:
            block_path_updates[int(start)] = [
                {
                    "start_layer": p.start_layer,
                    "indices": list(p.indices),
                    "cost": p.cost,
                    "hop_distances": list(p.hop_distances),
                }
                for p in assigned
            ]
        if verbose:
            block_logs.append(
                f"[{log_prefix}] {pass_label} block {start}: "
                f"{len(assigned)} tracks assigned "
                f"({n_cand} graph edges, {n_rej} filtered out)"
            )

    if all_velocities:
        velocities = np.vstack(all_velocities)
    else:
        velocities = np.empty((0, 3), dtype=np.float64)

    marks: list[tuple[int, int]] = []
    for fi, ids in used.by_frame.items():
        prev = initial_used.by_frame.get(fi, set())
        for pi in ids:
            if pi not in prev:
                marks.append((int(fi), int(pi)))

    return {
        "batch_idx": batch_idx,
        "marks": marks,
        "velocities": velocities,
        "block_path_updates": block_path_updates,
        "tracks_assigned": total_assigned,
        "candidates_seen": total_candidates,
        "filtered_rejected": total_rejected,
        "block_logs": block_logs,
        "n_blocks": len(block_starts),
    }


def run_relaxation_pass(
    block_graphs: dict[int, BlockCandidateGraph],
    frames: list[FramePositions],
    used: UsedDetections,
    criteria: PassCriteria,
    dt: float,
    *,
    workers: int = 1,
    verbose: bool = True,
    pass_label: str = "pass",
    log_prefix: str = "relax",
    max_blocks: int = 0,
    accepted_store: AcceptedPathStore | None = None,
) -> tuple[np.ndarray, dict]:
    from python_4be_eti.utils.basic_utils import (
        balance_frame_ranges,
        format_tracking_batch_plan,
    )

    nframes = len(frames)
    starts = block_starts_for_nframes(nframes, max_blocks=max_blocks)
    starts = [s for s in starts if s in block_graphs]
    if not starts:
        return np.empty((0, 3), dtype=np.float64), {
            "tracks_assigned": 0,
            "candidates_seen": 0,
            "filtered_rejected": 0,
            "matched_total": used.total,
            "velocity_segments": 0,
        }

    workers = max(1, int(workers))
    graphs_data = {int(k): _graph_to_dict(v) for k, v in block_graphs.items()}
    snapshot = used_snapshot(used)
    criteria_data = _criteria_to_dict(criteria)

    frame_ranges = frame_ranges_from_block_starts(starts)
    costs = block_costs_from_frames(frames, frame_ranges)
    batches, batch_costs = balance_frame_ranges(frame_ranges, workers, costs=costs)
    n_batches = len(batches)
    n_blocks = len(starts)

    if verbose:
        print(
            f"[{log_prefix}] {pass_label}: filter+assign on {n_blocks} block(s) "
            f"in {n_batches} batch(es), {min(workers, n_batches)} worker(s)",
            flush=True,
        )
        for line in format_tracking_batch_plan(batches, batch_costs):
            print(f"[{log_prefix}] {pass_label}: {line}", flush=True)

    all_velocities: list[np.ndarray] = []
    total_assigned = 0
    total_candidates = 0
    total_rejected = 0
    merged_path_updates: dict[int, list[dict]] = {}

    def _consume(result: dict) -> None:
        nonlocal total_assigned, total_candidates, total_rejected
        apply_marks(used, result["marks"])
        total_assigned += int(result["tracks_assigned"])
        total_candidates += int(result["candidates_seen"])
        total_rejected += int(result["filtered_rejected"])
        vels = result["velocities"]
        if isinstance(vels, np.ndarray) and vels.size:
            all_velocities.append(vels)
        for start, updates in result.get("block_path_updates", {}).items():
            merged_path_updates.setdefault(int(start), []).extend(updates)
        if accepted_store is not None:
            accepted_store.extend_from_updates(result.get("block_path_updates", {}))
        if verbose:
            for line in result.get("block_logs", []):
                print(line, flush=True)

    if workers <= 1 or n_batches <= 1:
        _consume(
            _relaxation_pass_batch(
                0,
                starts,
                graphs_data,
                snapshot,
                criteria_data,
                dt,
                pass_label=pass_label,
                log_prefix=log_prefix,
                verbose=verbose,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for batch_idx, batch in enumerate(batches):
                batch_starts = block_starts_from_frame_ranges(batch)
                fut = executor.submit(
                    _relaxation_pass_batch,
                    batch_idx,
                    batch_starts,
                    graphs_data,
                    snapshot,
                    criteria_data,
                    dt,
                    pass_label=pass_label,
                    log_prefix=log_prefix,
                    verbose=False,
                )
                futures.append((batch_idx, batch, fut))
            fut_to_batch = {fut: (idx, batch) for idx, batch, fut in futures}
            done = 0
            for fut in as_completed(fut_to_batch):
                batch_idx, batch = fut_to_batch[fut]
                result = fut.result()
                _consume(result)
                done += 1
                if verbose:
                    ordered = sorted(batch, key=lambda fr: min(fr) if fr else 0)
                    if ordered:
                        span_lo = min(min(fr) for fr in ordered)
                        span_hi = max(max(fr) for fr in ordered)
                        span = f"frames {span_lo}-{span_hi}"
                    else:
                        span = "frames (none)"
                    print(
                        f"[{log_prefix}] {pass_label} batch {done}/{n_batches} complete "
                        f"({result['n_blocks']} block(s), {span}, "
                        f"{result['tracks_assigned']} tracks assigned)",
                        flush=True,
                    )

    if all_velocities:
        velocities = np.vstack(all_velocities)
    else:
        velocities = np.empty((0, 3), dtype=np.float64)

    stats = {
        "tracks_assigned": total_assigned,
        "candidates_seen": total_candidates,
        "filtered_rejected": total_rejected,
        "matched_total": used.total,
        "velocity_segments": int(velocities.shape[0]) if velocities.size else 0,
        "n_blocks": n_blocks,
        "n_batches": n_batches,
        "workers": workers,
        "block_path_updates": merged_path_updates,
    }
    return velocities, stats


def write_accepted_paths_h5part(
    block_graphs: dict[int, BlockCandidateGraph],
    accepted_store: AcceptedPathStore,
    output_dir: str,
    output_path: str,
    *,
    min_track_length: int = 3,
    dt: float = 1e-3,
    rep_rate: float = 10.0,
    position_unit: str = "mm",
    verbose: bool = True,
) -> dict:
    """Write h5part directly from accumulated graph assignments (no re-linking)."""
    import os

    from python_4be_eti.utils.basic_utils import write_data_h5

    from .track_data import block_to_tracking_data, merge_block_data

    output_dir = os.path.abspath(output_dir)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(output_path):
        os.remove(output_path)

    merged_data = None
    all_frames: list[int] = []
    total_tracks = 0
    n_blocks_written = 0
    next_track_id = 1
    for start in sorted(accepted_store.by_block.keys()):
        paths = accepted_store.by_block.get(start) or []
        if not paths:
            continue
        graph = block_graphs.get(int(start))
        if graph is None:
            continue
        block_data = block_to_tracking_data(graph.block, paths)
        if next_track_id > 1:
            mask = block_data.Count > 0
            block_data.Count[mask] += next_track_id - 1
        next_track_id += len(paths)
        total_tracks += len(paths)
        n_blocks_written += 1
        merged_data = merge_block_data(merged_data, block_data)
        all_frames.extend(graph.block.frame_indices)

    if merged_data is not None and all_frames:
        write_data_h5(
            merged_data,
            output_dir,
            sorted(set(all_frames)),
            output_h5part=output_path,
            dt=dt,
            rep_rate_hz=rep_rate,
            track_length=4,
            position_scale_to_m=1e-3,
            position_units_str=position_unit,
            use_bspline=True,
            min_track_length=min_track_length,
            verbose=False,
            show_progress=False,
        )

    if verbose:
        print(
            f"[relax] wrote {total_tracks} track(s) from accepted paths → {output_path}",
            flush=True,
        )
    return {
        "total_tracks": total_tracks,
        "n_blocks": n_blocks_written,
        "output_h5part": output_path,
    }
