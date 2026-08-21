"""Simple stdout progress helpers for long graph / legacy runs."""

from __future__ import annotations

import sys
import time


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(
    label: str,
    current: int,
    total: int,
    *,
    interval: int = 5000,
    t0: float | None = None,
) -> None:
    """
    Print ``label`` progress every ``interval`` steps (and at start/end).

    ``interval=0`` disables periodic updates (still prints at end).
    ``interval<0`` disables all progress output.
    """
    if total <= 0 or interval < 0:
        return
    at_end = current >= total
    at_start = current == 1
    periodic = interval > 0 and current % interval == 0
    if not (at_start or periodic or at_end):
        return
    pct = 100.0 * current / total
    elapsed = ""
    if t0 is not None and current > 0:
        dt = time.perf_counter() - t0
        rate = current / dt if dt > 0 else 0.0
        elapsed = f", {dt:.1f}s, {rate:.0f}/s"
    log(f"  [{label}] {current}/{total} ({pct:.1f}%){elapsed}")


def _frame_span(frame_indices: list[int]) -> tuple[int, int]:
    if not frame_indices:
        return 0, 0
    return int(frame_indices[0]), int(frame_indices[-1])


def _pct(current: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * current / total:.1f}%"


def report_graph_started(n_blocks: int, n_batches: int, workers: int) -> None:
    log(
        f"[graph] Started: {n_blocks} block(s), {n_batches} batch(es), "
        f"{min(workers, n_batches)} worker(s)"
    )


def report_graph_block(
    block_idx: int,
    n_blocks: int,
    frame_indices: list[int],
    n_tracks: int,
    elapsed_s: float,
    *,
    n_candidates: int = 0,
    n_seeds: int = 0,
) -> None:
    lo, hi = _frame_span(frame_indices)
    extra = ""
    if n_candidates or n_seeds:
        extra = f", {n_seeds} seeds, {n_candidates} cand"
    log(
        f"[graph] block {block_idx}/{n_blocks} ({_pct(block_idx, n_blocks)}) "
        f"frames {lo}-{hi}: {n_tracks} tracks, {elapsed_s:.1f}s{extra}"
    )


def report_graph_batch_done(
    done: int,
    n_batches: int,
    batch_idx: int,
    n_blocks: int,
    frame_lo: int,
    frame_hi: int,
    n_tracks: int,
    elapsed_s: float,
    *,
    blocks_done: int,
    n_blocks_total: int,
) -> None:
    log(
        f"[graph] batch {done}/{n_batches} ({_pct(done, n_batches)}) complete, "
        f"job {_pct(blocks_done, n_blocks_total)} "
        f"(worker batch {batch_idx}, {n_blocks} block(s), "
        f"frames {frame_lo}-{frame_hi}, {n_tracks} tracks, {elapsed_s:.1f}s)"
    )


def report_graph_merged(n_parts: int, out_path: str) -> None:
    log(f"[graph] Merged {n_parts} part file(s) → {out_path}")
