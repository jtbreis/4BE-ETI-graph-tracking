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
    """
    if total <= 0:
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
