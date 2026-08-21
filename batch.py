"""Parallel 4-frame block batches for graph tracking (merge part h5part files)."""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from python_4be_eti.utils.basic_utils import (
    balance_frame_ranges,
    estimate_frame_range_tracking_cost,
    format_tracking_batch_plan,
    write_data_h5,
)

from .assign import greedy_assign_paths
from .link_block import LinkParams, link_block
from .load_data import (
    get_nframes,
    iter_block_starts,
    load_block,
    block_starts_for_nframes,
    particle_counts_from_masks,
)
from .progress import (
    _pct,
    report_graph_batch_done,
    report_graph_block,
    report_graph_merged,
)
from .track_data import block_to_tracking_data

GRAPH_PART_PREFIX = "tracks_graph_part_"


@dataclass
class GraphBatchResult:
    batch_idx: int
    n_blocks: int
    total_tracks: int
    elapsed_s: float
    frame_lo: int
    frame_hi: int
    block_rows: list[dict]


def link_params_to_dict(params: LinkParams) -> dict:
    return asdict(params)


def link_params_from_dict(data: dict) -> LinkParams:
    return LinkParams(**data)


def frame_ranges_from_block_starts(starts: list[int]) -> list[list[int]]:
    return [list(range(s, s + 4)) for s in starts]


def block_starts_from_frame_ranges(frame_ranges: list[list[int]]) -> list[int]:
    return [min(fr) for fr in frame_ranges]


def _batch_frame_span(frame_ranges: list[list[int]]) -> tuple[int, int]:
    ordered = sorted(frame_ranges, key=lambda fr: min(fr) if fr else 0)
    if not ordered:
        return 0, 0
    return min(min(fr) for fr in ordered), max(max(fr) for fr in ordered)


def remove_graph_tracks_h5part_parts(folder: str) -> None:
    pattern = os.path.join(folder, f"{GRAPH_PART_PREFIX}*.h5part")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass


def _graph_part_index(path: str) -> int:
    m = re.search(rf"{GRAPH_PART_PREFIX}(\d+)\.h5part$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def merge_graph_tracks_h5part_parts(folder: str, out_path: str) -> str | None:
    """Merge tracks_graph_part_*.h5part into one h5part (disjoint Step# groups)."""
    import h5py

    pattern = os.path.join(folder, f"{GRAPH_PART_PREFIX}*.h5part")
    part_files = sorted(glob.glob(pattern), key=_graph_part_index)
    if not part_files:
        return None

    tmp_path = out_path + ".tmp"
    root_attrs: dict = {}
    n_copied = 0
    with h5py.File(tmp_path, "w") as out:
        for part_path in part_files:
            with h5py.File(part_path, "r") as inc:
                for ak, av in inc.attrs.items():
                    if ak not in root_attrs:
                        root_attrs[ak] = av
                for name in inc.keys():
                    if name in out:
                        logging.warning(
                            "merge_graph_tracks_h5part_parts: skip duplicate %s from %s",
                            name,
                            part_path,
                        )
                        continue
                    inc.copy(name, out)
                    n_copied += 1
        for ak, av in root_attrs.items():
            out.attrs[ak] = av
    os.replace(tmp_path, out_path)
    for part_path in part_files:
        try:
            os.remove(part_path)
        except OSError:
            pass
    logging.info(
        "Merged %d graph part file(s) (%d Step groups) into %s",
        len(part_files),
        n_copied,
        out_path,
    )
    return out_path


def process_graph_tracking_batch(
    batch_idx: int,
    block_starts: list[int],
    h5in: str,
    out_dir: str,
    link_params_dict: dict,
    *,
    min_track_length: int = 3,
    dt: float = 1e-3,
    rep_rate: float = 10.0,
    position_unit: str = "mm",
    show_write_progress: bool = False,
    n_blocks_total: int = 0,
    masks: list | None = None,
) -> dict:
    """Process one load-balanced batch; write tracks_graph_part_<batch_idx>.h5part."""
    t_batch = time.perf_counter()
    params = link_params_from_dict(link_params_dict)
    h5in = os.path.abspath(h5in)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{GRAPH_PART_PREFIX}{batch_idx}.h5part")
    if os.path.isfile(out_path):
        os.remove(out_path)

    total_tracks = 0
    next_track_id = 1
    ordered = sorted(block_starts)
    block_rows: list[dict] = []
    frame_lo = frame_hi = 0

    for bi, start in enumerate(ordered, start=1):
        t_block = time.perf_counter()
        block = load_block(h5in, start, masks=masks)
        if bi == 1:
            frame_lo = block.frame_indices[0]
        if bi == len(ordered):
            frame_hi = block.frame_indices[-1]

        result = link_block(block, params)
        assigned = greedy_assign_paths(result.candidates)
        block_data = block_to_tracking_data(block, assigned)
        if next_track_id > 1:
            mask = block_data.Count > 0
            block_data.Count[mask] += next_track_id - 1
        next_track_id += len(assigned)
        total_tracks += len(assigned)

        write_data_h5(
            block_data,
            out_dir,
            block.frame_indices,
            output_h5part=out_path,
            dt=dt,
            rep_rate_hz=rep_rate,
            track_length=4,
            position_scale_to_m=1e-3,
            position_units_str=position_unit,
            use_bspline=True,
            min_track_length=min_track_length,
            verbose=False,
            show_progress=show_write_progress,
        )

        block_elapsed = time.perf_counter() - t_block
        block_rows.append(
            {
                "block_idx": bi,
                "n_blocks_total": n_blocks_total or len(ordered),
                "frame_indices": list(block.frame_indices),
                "n_tracks": len(assigned),
                "elapsed_s": block_elapsed,
                "n_candidates": result.stats.get("n_candidates", 0),
                "n_seeds": result.stats.get("n_seeds_tried", 0),
            }
        )

    return asdict(
        GraphBatchResult(
            batch_idx=batch_idx,
            n_blocks=len(ordered),
            total_tracks=total_tracks,
            elapsed_s=time.perf_counter() - t_batch,
            frame_lo=int(frame_lo),
            frame_hi=int(frame_hi),
            block_rows=block_rows,
        )
    )


def run_graph_blocks_parallel(
    h5in: str,
    out_dir: str,
    out_path: str,
    block_starts: list[int],
    params: LinkParams,
    *,
    workers: int = 1,
    min_track_length: int = 3,
    dt: float = 1e-3,
    rep_rate: float = 10.0,
    position_unit: str = "mm",
    show_write_progress: bool = True,
    verbose: bool = True,
    masks: list | None = None,
) -> tuple[int, int]:
    """
    Link and write all blocks, optionally in parallel worker batches.

    When ``verbose`` is False (weblauncher / quiet), workers stay silent and the
    parent prints one line per batch (parallel) or per block (single worker).

    Returns (total_tracks, n_blocks).
    """
    if not block_starts:
        return 0, 0

    workers = max(1, int(workers))
    params_dict = link_params_to_dict(params)
    if workers > 1:
        params_dict["quiet_linking"] = True
    remove_graph_tracks_h5part_parts(out_dir)
    if os.path.isfile(out_path):
        os.remove(out_path)

    frame_ranges = frame_ranges_from_block_starts(block_starts)
    if masks is not None:
        counts = particle_counts_from_masks(masks)
        costs = [estimate_frame_range_tracking_cost(counts, fr) for fr in frame_ranges]
        batches, batch_costs = balance_frame_ranges(frame_ranges, workers, costs=costs)
    else:
        batches, batch_costs = balance_frame_ranges(
            frame_ranges, workers, filename=h5in,
        )
    n_batches = len(batches)
    n_blocks = len(block_starts)
    structured = not verbose

    for line in format_tracking_batch_plan(batches, batch_costs):
        print(line, flush=True)
    if structured:
        from .progress import report_graph_started

        report_graph_started(n_blocks, n_batches, workers)
    elif verbose:
        print(
            f"Graph tracking: {n_blocks} block(s) in {n_batches} batch(es), "
            f"{min(workers, n_batches)} worker(s)",
            flush=True,
        )

    total_tracks = 0

    if workers <= 1 or n_batches <= 1:
        result = process_graph_tracking_batch(
            0,
            block_starts,
            h5in,
            out_dir,
            params_dict,
            min_track_length=min_track_length,
            dt=dt,
            rep_rate=rep_rate,
            position_unit=position_unit,
            show_write_progress=show_write_progress and not structured,
            n_blocks_total=n_blocks,
            masks=masks,
        )
        batch = GraphBatchResult(**result)
        total_tracks = batch.total_tracks
        if structured:
            for row in batch.block_rows:
                report_graph_block(
                    row["block_idx"],
                    row["n_blocks_total"],
                    row["frame_indices"],
                    row["n_tracks"],
                    row["elapsed_s"],
                    n_candidates=row.get("n_candidates", 0),
                    n_seeds=row.get("n_seeds", 0),
                )
        if out_path != os.path.join(out_dir, f"{GRAPH_PART_PREFIX}0.h5part"):
            part_path = os.path.join(out_dir, f"{GRAPH_PART_PREFIX}0.h5part")
            os.replace(part_path, out_path)
        return total_tracks, n_blocks

    show_progress = show_write_progress and not structured
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for batch_idx, batch in enumerate(batches):
            starts = block_starts_from_frame_ranges(batch)
            fut = executor.submit(
                process_graph_tracking_batch,
                batch_idx,
                starts,
                h5in,
                out_dir,
                params_dict,
                min_track_length=min_track_length,
                dt=dt,
                rep_rate=rep_rate,
                position_unit=position_unit,
                show_write_progress=show_progress,
                n_blocks_total=n_blocks,
                masks=masks,
            )
            futures.append((batch_idx, batch, fut))

        fut_to_batch = {fut: (idx, batch) for idx, batch, fut in futures}
        done = 0
        blocks_done = 0
        for fut in as_completed(fut_to_batch):
            batch_idx, batch = fut_to_batch[fut]
            result = GraphBatchResult(**fut.result())
            total_tracks += result.total_tracks
            done += 1
            blocks_done += result.n_blocks
            if structured:
                frame_lo, frame_hi = _batch_frame_span(batch)
                report_graph_batch_done(
                    done,
                    n_batches,
                    batch_idx,
                    result.n_blocks,
                    frame_lo,
                    frame_hi,
                    result.total_tracks,
                    result.elapsed_s,
                    blocks_done=blocks_done,
                    n_blocks_total=n_blocks,
                )
            elif verbose:
                frame_lo, frame_hi = _batch_frame_span(batch)
                print(
                    f"Graph batch {done}/{n_batches} ({_pct(done, n_batches)}) complete, "
                    f"job {_pct(blocks_done, n_blocks)} "
                    f"({result.n_blocks} block(s), frames {frame_lo}-{frame_hi}, "
                    f"{result.total_tracks} tracks, {result.elapsed_s:.1f}s)",
                    flush=True,
                )

    merged = merge_graph_tracks_h5part_parts(out_dir, out_path)
    if structured and merged:
        report_graph_merged(len(batches), out_path)
    return total_tracks, n_blocks
