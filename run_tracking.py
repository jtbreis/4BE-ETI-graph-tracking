#!/usr/bin/env python3
"""
Run graph-based 4-frame tracking on rays_out_cpp.h5.

Example:
  python graph_tracking/run_tracking.py \\
    --input data/.../Run1/rays_out_cpp.h5 \\
    --output-dir data/.../Run1 \\
    --workers 8
"""

from __future__ import annotations

import argparse
import os
import time

from .batch import block_starts_for_nframes, run_graph_blocks_parallel
from .link_block import LinkParams
from .load_data import get_nframes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-based 4-frame tracking")
    p.add_argument("--input", required=True, help="Path to rays_out_cpp.h5")
    p.add_argument("--output-dir", required=True, help="Directory for tracks_graph.h5part")
    p.add_argument("--output", default=None, help="Output h5part path (default: output-dir/tracks_graph.h5part)")
    p.add_argument("--max-blocks", type=int, default=0, help="Max 4-frame blocks (0 = all)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers for frame-range batches")
    p.add_argument("--box-size-track", type=float, default=0.2)
    p.add_argument("--initial-x-lo", type=float, default=0.0)
    p.add_argument("--initial-x-hi", type=float, default=3.5)
    p.add_argument("--initial-y-lo", type=float, default=-1.5)
    p.add_argument("--initial-y-hi", type=float, default=1.5)
    p.add_argument("--initial-z-lo", type=float, default=-1.5)
    p.add_argument("--initial-z-hi", type=float, default=1.5)
    p.add_argument("--max-branch", type=int, default=30, help="Max candidates per path stage")
    p.add_argument("--max-seeds", type=int, default=0, help="Limit frame-0 seeds (0 = all)")
    p.add_argument("--no-3-frame", action="store_true", help="Only allow 4-frame tracks")
    p.add_argument(
        "--use-bidirectional-graph",
        "--bidirectional",
        action="store_true",
        dest="use_bidirectional_graph",
        help="Merge forward and backward 4-frame graphs (consensus paths only), "
        "then assign on the combined graph (--bidirectional is an alias)",
    )
    p.add_argument("--min-track-length", type=int, default=3, help="Min frames per track in h5part output")
    p.add_argument("--progress-interval", type=int, default=5000,
                   help="Print progress every N seeds (0 = phase lines only)")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--rep-rate", type=float, default=10.0)
    p.add_argument("--position-unit", default="mm")
    return p.parse_args()


def run_graph_tracking(
    h5in: str,
    out_dir: str,
    *,
    output: str | None = None,
    max_blocks: int = 0,
    workers: int = 1,
    link_params: LinkParams | None = None,
    min_track_length: int = 3,
    dt: float = 1e-3,
    rep_rate: float = 10.0,
    position_unit: str = "mm",
    show_write_progress: bool = True,
    verbose: bool = True,
    masks: list | None = None,
) -> dict:
    """
    Run graph-based 4-frame tracking on one rays_out_cpp.h5 file.

    ``max_blocks``: number of 4-frame blocks to process; 0 = all blocks in file.
    ``workers``: parallel batches (load-balanced by particle count per block).
    """
    h5in = os.path.abspath(h5in)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    h5out = output or os.path.join(out_dir, "tracks_graph.h5part")

    params = link_params or LinkParams()
    nframes = get_nframes(h5in)
    starts = block_starts_for_nframes(nframes, max_blocks=max_blocks)

    if verbose:
        print(f"Input: {h5in} ({nframes} frames, processing {len(starts)} block(s))")
        print(
            f"Params: box_track={params.box_size_track}, max_branch={params.max_branch}, "
            f"allow_3_frame={params.allow_3_frame}, min_track_length={min_track_length}, "
            f"use_numba={params.use_numba}, workers={workers}"
        )
        if params.use_bidirectional_graph:
            print("Bidirectional graph mode: ON (consensus 4-frame paths only)")

    t0 = time.time()
    total_tracks, n_blocks = run_graph_blocks_parallel(
        h5in,
        out_dir,
        h5out,
        starts,
        params,
        workers=workers,
        min_track_length=min_track_length,
        dt=dt,
        rep_rate=rep_rate,
        position_unit=position_unit,
        show_write_progress=show_write_progress,
        verbose=verbose,
        masks=masks,
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"Done: {total_tracks} tracks across {n_blocks} block(s) in {elapsed:.2f}s")
        print(f"Output: {h5out}")
    else:
        print(
            f"[graph] Done: {total_tracks} tracks across {n_blocks} block(s) "
            f"in {elapsed:.1f}s → {h5out}",
            flush=True,
        )

    return {
        "total_tracks": total_tracks,
        "n_blocks": n_blocks,
        "n_frames": nframes,
        "workers": workers,
        "output_h5part": h5out,
        "elapsed_s": elapsed,
    }


def main() -> int:
    args = parse_args()
    h5in = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output_dir)
    h5out = args.output or os.path.join(out_dir, "tracks_graph.h5part")

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
        allow_3_frame=not args.no_3_frame,
        use_bidirectional_graph=args.use_bidirectional_graph,
        progress_interval=args.progress_interval,
    )

    run_graph_tracking(
        h5in,
        out_dir,
        output=h5out,
        max_blocks=args.max_blocks,
        workers=args.workers,
        link_params=params,
        min_track_length=args.min_track_length,
        dt=args.dt,
        rep_rate=args.rep_rate,
        position_unit=args.position_unit,
        show_write_progress=True,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
