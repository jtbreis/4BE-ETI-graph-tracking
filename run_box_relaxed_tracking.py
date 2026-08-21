#!/usr/bin/env python3
"""CLI for progressive box_size relaxation with per-block dominant-flow gates."""

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

from graph_tracking.box_relaxed_tracking import BoxRelaxedTrackingConfig, run_box_relaxed_tracking
from graph_tracking.link_block import LinkParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Set initial tracking parameters, then progressively relax box_size_track. "
            "At the tightest box_size, each 4-frame block picks the busiest velocity "
            "histogram bin among seed-valid candidates; relaxed passes only accept "
            "tracks whose velocities stay in that bin."
        )
    )
    p.add_argument("--input", required=True, help="Path to rays_out_cpp.h5")
    p.add_argument("--output-dir", required=True, help="Directory for outputs")
    p.add_argument("--max-stereo-error", type=float, default=0.1,
                   help="STM particle pool: stereomatching error ceiling")
    p.add_argument("--box-size-min", type=float, default=0.15,
                   help="Initial (tightest) hop limit in mm")
    p.add_argument("--box-size-max", type=float, default=0.5,
                   help="Maximum relaxed hop limit in mm (also graph-build ceiling)")
    p.add_argument("--box-size-step", type=float, default=0.05, help="Hop limit step per pass")
    p.add_argument("--velocity-variable", default="vx", choices=["vx", "vy", "vz", "speed"],
                   help="Velocity component for the gate")
    p.add_argument("--velocity-target", type=float, default=None,
                   help="Optional fixed peak override. Default: detect from histogram at tight box.")
    p.add_argument("--velocity-sigma", type=float, default=1.0,
                   help="Fixed ±σ around detected peak (seed pass uses 1×σ)")
    p.add_argument("--velocity-bin-count", type=int, default=0,
                   help="Histogram bins for peak detection at tight box (0 = auto)")
    p.add_argument("--velocity-band-margin", type=float, default=2.0,
                   help="Max σ multiplier at final relax pass (seed=1, ramps linearly)")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--min-track-length", type=int, default=3)
    p.add_argument("--rep-rate", type=float, default=10.0)
    p.add_argument("--position-unit", default="mm")
    p.add_argument("--filtered-stm-name", default="rays_out_box_filtered.h5")
    p.add_argument("--tracks-name", default="tracks_box_relaxed.h5part")
    p.add_argument("--no-write-tracks-each-pass", action="store_true")
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
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    link = LinkParams(
        box_size_track=args.box_size_min,
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
    cfg = BoxRelaxedTrackingConfig(
        max_stereo_error=args.max_stereo_error,
        box_size_min=args.box_size_min,
        box_size_max=args.box_size_max,
        box_size_step=args.box_size_step,
        velocity_variable=args.velocity_variable,
        velocity_target=args.velocity_target,
        velocity_sigma=args.velocity_sigma,
        velocity_bin_count=args.velocity_bin_count,
        velocity_band_margin=args.velocity_band_margin,
        dt=args.dt,
        workers=args.workers,
        max_blocks=args.max_blocks,
        min_track_length=args.min_track_length,
        rep_rate=args.rep_rate,
        position_unit=args.position_unit,
        link_params=link,
        filtered_stm_name=args.filtered_stm_name,
        tracks_name=args.tracks_name,
        write_tracks_each_pass=not args.no_write_tracks_each_pass,
    )
    run_box_relaxed_tracking(args.input, args.output_dir, cfg, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
