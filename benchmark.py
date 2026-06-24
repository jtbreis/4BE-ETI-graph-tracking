#!/usr/bin/env python3
"""
Benchmark graph tracking vs legacy 4BE-ETI on the same 4-frame block(s).

Example:
  python graph_tracking/benchmark.py \\
    --input data/.../Run1/rays_out_cpp.h5 \\
    --max-blocks 1 \\
    --max-seeds 5000
"""

from __future__ import annotations

import argparse
import os
import time

from .assign import greedy_assign_paths
from .link_block import LinkParams, link_block
from .load_data import get_nframes, iter_block_starts, load_block


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark graph vs legacy 4-frame tracking")
    p.add_argument("--input", required=True)
    p.add_argument("--max-blocks", type=int, default=1)
    p.add_argument("--box-size-track", type=float, default=0.2)
    p.add_argument("--initial-x-lo", type=float, default=0.0)
    p.add_argument("--initial-x-hi", type=float, default=3.5)
    p.add_argument("--initial-y-lo", type=float, default=-1.5)
    p.add_argument("--initial-y-hi", type=float, default=1.5)
    p.add_argument("--initial-z-lo", type=float, default=-1.5)
    p.add_argument("--initial-z-hi", type=float, default=1.5)
    p.add_argument("--max-branch", type=int, default=30)
    p.add_argument("--max-seeds", type=int, default=5000, help="Limit seeds for faster benchmark")
    p.add_argument("--skip-legacy", action="store_true", help="Skip legacy 4BE-ETI timing")
    return p.parse_args()


def run_graph(h5path: str, start: int, params: LinkParams) -> dict:
    t0 = time.time()
    block = load_block(h5path, start)
    result = link_block(block, params)
    assigned = greedy_assign_paths(result.candidates)
    elapsed = time.time() - t0
    return {
        "time_s": elapsed,
        "candidates": len(result.candidates),
        "tracks": len(assigned),
        "particles": [f.n for f in block.frames],
        "stats": result.stats,
    }


def run_legacy(h5path: str, start: int, params: LinkParams) -> dict:
    from python_4be_eti.particle_tracking import process_frame_range

    frame_range = [start + i for i in range(4)]
    folder = os.path.dirname(h5path)
    filename = os.path.basename(h5path)
    t0 = time.time()
    process_frame_range(
        frame_range,
        folder,
        filename,
        dimension="3d",
        box_size=params.box_size_track,
        box_size_initial_x_lo=params.initial_x_lo,
        box_size_initial_x_hi=params.initial_x_hi,
        box_size_initial_y_lo=params.initial_y_lo,
        box_size_initial_y_hi=params.initial_y_hi,
        box_size_initial_z_lo=params.initial_z_lo,
        box_size_initial_z_hi=params.initial_z_hi,
        start_time=t0,
        show_progress=False,
        output_h5part=None,
        verbose=False,
        particle_progress_interval=0,
        max_candidates_mesh=params.max_branch,
        max_targets_mesh=max(2000, params.max_branch * 2),
    )
    elapsed = time.time() - t0
    return {"time_s": elapsed, "frame_range": frame_range}


def main() -> int:
    args = parse_args()
    h5path = os.path.abspath(args.input)
    params = LinkParams(
        box_size_track=args.box_size_track,
        initial_x_lo=args.initial_x_lo,
        initial_x_hi=args.initial_x_hi,
        initial_y_lo=args.initial_y_lo,
        initial_y_hi=args.initial_y_hi,
        initial_z_lo=args.initial_z_lo,
        initial_z_hi=args.initial_z_hi,
        max_branch=args.max_branch,
        max_seeds=args.max_seeds,
    )

    nframes = get_nframes(h5path)
    starts = iter_block_starts(nframes)[: args.max_blocks]
    print(f"Benchmark: {h5path}")
    print(f"Blocks: {starts}, max_seeds={params.max_seeds}, max_branch={params.max_branch}\n")

    for start in starts:
        print(f"=== Block starting frame {start} ===")
        g = run_graph(h5path, start, params)
        print(
            f"Graph:  {g['time_s']:.2f}s | candidates={g['candidates']} | "
            f"tracks={g['tracks']} | particles/frame={g['particles']}"
        )

        if not args.skip_legacy:
            print(
                "Running legacy 4BE-ETI on ALL particles in the block "
                "(graph may use --max-seeds to subsample frame-0 seeds)..."
            )
            leg = run_legacy(h5path, start, params)
            print(f"Legacy: {leg['time_s']:.2f}s")
            if g["time_s"] > 0:
                print(f"Speedup (legacy/graph): {leg['time_s'] / g['time_s']:.2f}x")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
