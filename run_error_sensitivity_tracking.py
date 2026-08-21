#!/usr/bin/env python3
"""CLI for stereomatching-error sensitivity sweeps via graph tracking."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_BE_ROOT = os.path.join(_REPO_ROOT, "4BE-ETI", "src")
if _BE_ROOT not in sys.path:
    sys.path.insert(0, _BE_ROOT)

from graph_tracking.error_sensitivity_tracking import (
    ErrorSensitivityConfig,
    run_error_sensitivity_tracking,
)
from graph_tracking.link_block import LinkParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep stereomatching error thresholds: for each ceiling, filter the STM "
            "in-memory (error <= threshold), run graph tracking, and write tracks files."
        )
    )
    p.add_argument("--input", required=True, help="Path to rays_out_cpp.h5 (high maxdistance STM)")
    p.add_argument("--output-dir", required=True, help="Directory for per-threshold outputs")
    p.add_argument("--error-min", type=float, default=0.05, help="First error threshold")
    p.add_argument("--error-max", type=float, default=0.0,
                   help="Last error threshold (0 = use maxdistance attr from STM file)")
    p.add_argument("--error-step", type=float, default=0.01, help="Error sweep step size")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--box-size-track", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=0, help="Limit 4-frame blocks (0 = all)")
    p.add_argument("--min-track-length", type=int, default=3)
    p.add_argument("--rep-rate", type=float, default=10.0)
    p.add_argument("--position-unit", default="mm")
    p.add_argument("--tracks-template", default="tracks_error_{error:.4f}.h5part")
    p.add_argument("--initial-x-lo", type=float, default=0.0)
    p.add_argument("--initial-x-hi", type=float, default=3.5)
    p.add_argument("--initial-y-lo", type=float, default=-1.5)
    p.add_argument("--initial-y-hi", type=float, default=1.5)
    p.add_argument("--initial-z-lo", type=float, default=-1.5)
    p.add_argument("--initial-z-hi", type=float, default=1.5)
    p.add_argument("--max-branch", type=int, default=30)
    p.add_argument("--max-seeds", type=int, default=0)
    p.add_argument("--no-3-frame", action="store_true")
    p.add_argument(
        "--use-bidirectional-graph",
        "--bidirectional",
        action="store_true",
        dest="use_bidirectional_graph",
        help="Merge forward and backward 4-frame graphs (consensus paths only)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    error_max = args.error_max if args.error_max > 0 else 0.2
    link = LinkParams(
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
        quiet_linking=False,
    )
    cfg = ErrorSensitivityConfig(
        error_min=args.error_min,
        error_max=error_max,
        error_step=args.error_step,
        dt=args.dt,
        box_size_track=args.box_size_track,
        workers=args.workers,
        max_blocks=args.max_blocks,
        min_track_length=args.min_track_length,
        rep_rate=args.rep_rate,
        position_unit=args.position_unit,
        link_params=link,
        tracks_template=args.tracks_template,
    )
    run_error_sensitivity_tracking(args.input, args.output_dir, cfg, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
