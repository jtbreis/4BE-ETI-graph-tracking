#!/usr/bin/env python3
"""
Run graph-based 4-frame tracking on rays_out_cpp.h5.

Example:
  python graph_tracking/run_tracking.py \\
    --input data/.../Run1/rays_out_cpp.h5 \\
    --output-dir data/.../Run1 \\
    --max-blocks 1
"""

from __future__ import annotations

import argparse
import os
import time

from python_4be_eti.utils.basic_utils import write_data_h5

from .assign import greedy_assign_paths
from .link_block import LinkParams, link_block
from .load_data import get_nframes, iter_block_starts, load_block
from .track_data import block_to_tracking_data, merge_block_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-based 4-frame tracking")
    p.add_argument("--input", required=True, help="Path to rays_out_cpp.h5")
    p.add_argument("--output-dir", required=True, help="Directory for tracks_graph.h5part")
    p.add_argument("--output", default=None, help="Output h5part path (default: output-dir/tracks_graph.h5part)")
    p.add_argument("--max-blocks", type=int, default=1, help="Max 4-frame blocks to process (default: 1)")
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
        "--bidirectional",
        action="store_true",
        help="Require 4-frame paths to validate forward and backward (flipped initial box)",
    )
    p.add_argument("--min-track-length", type=int, default=3, help="Min frames per track in h5part output")
    p.add_argument("--progress-interval", type=int, default=5000,
                   help="Print progress every N seeds (0 = phase lines only)")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--rep-rate", type=float, default=10.0)
    p.add_argument("--position-unit", default="mm")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h5in = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    h5out = args.output or os.path.join(out_dir, "tracks_graph.h5part")
    if os.path.isfile(h5out):
        os.remove(h5out)

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
        require_bidirectional=args.bidirectional,
        progress_interval=args.progress_interval,
    )

    nframes = get_nframes(h5in)
    starts = iter_block_starts(nframes)[: args.max_blocks]
    print(f"Input: {h5in} ({nframes} frames, processing {len(starts)} block(s))")
    print(
        f"Params: box_track={params.box_size_track}, max_branch={params.max_branch}, "
        f"allow_3_frame={params.allow_3_frame}, min_track_length={args.min_track_length}"
    )
    if params.require_bidirectional:
        print("Bidirectional 4-frame filter: ON (forward + backward enumeration)")

    t0 = time.time()
    combined = None
    total_tracks = 0
    next_track_id = 1

    for bi, start in enumerate(starts):
        tb = time.time()
        block = load_block(h5in, start)
        print(
            f"Block {bi + 1}/{len(starts)} frames {block.frame_indices} "
            f"({', '.join(str(f.n) for f in block.frames)} particles)"
        )

        result = link_block(block, params)
        assigned = greedy_assign_paths(result.candidates)
        print(
            f"  candidates={result.stats['n_candidates']} "
            f"(4f={result.stats.get('n_candidates_4frame', 0)}, "
            f"3f0={result.stats.get('n_candidates_3frame_f0', 0)}, "
            f"3f1={result.stats.get('n_candidates_3frame_f1', 0)}), "
            f"assigned={len(assigned)}, "
            f"seeds={result.stats['n_seeds_tried']}, "
            f"link {time.time() - tb:.2f}s"
        )

        block_data = block_to_tracking_data(block, assigned)
        # Offset track IDs so blocks do not collide when appended to one h5part
        if next_track_id > 1:
            mask = block_data.Count > 0
            block_data.Count[mask] += next_track_id - 1
        next_track_id += len(assigned)
        combined = merge_block_data(combined, block_data)
        total_tracks += len(assigned)

        frame_range = block.frame_indices
        write_data_h5(
            block_data,
            out_dir,
            frame_range,
            output_h5part=h5out,
            dt=args.dt,
            rep_rate_hz=args.rep_rate,
            track_length=4,
            position_scale_to_m=1e-3,
            position_units_str=args.position_unit,
            use_bspline=True,
            min_track_length=args.min_track_length,
            verbose=False,
            show_progress=True,
        )

    elapsed = time.time() - t0
    print(f"Done: {total_tracks} tracks across {len(starts)} block(s) in {elapsed:.2f}s")
    print(f"Output: {h5out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
