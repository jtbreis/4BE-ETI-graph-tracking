#!/usr/bin/env python3
# %% [markdown]
# # Compare graph vs legacy 4-frame tracking
#
# Runs both methods on **one** 4-frame block (default: frames 0–3) of
# `rays_out_cpp.h5`, compares track counts / matched paths, and reports runtime.
#
# Default input: TNTI Run1 stereomatching output. Use `--synthetic` for the
# fast two-particle sanity check.

# %%
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from graph_tracking.compare import compare_block, format_report
from graph_tracking.link_block import LinkParams
from graph_tracking.synthetic.two_particles import (
    TwoParticleScenario,
    expected_link_params,
    write_rays_h5,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
DEFAULT_RUN1 = os.path.join(
    PROJECT_ROOT,
    "data/Windtunnel_PTV_TNTI/X300cm_PTV_center_TNTI/TNTI_aligned_with_gravity/Run1",
    "rays_out_cpp.h5",
)


def run1_link_params(**overrides) -> dict:
    """Defaults from Processing Steps/05_4frame_tracking.py (TNTI Run1)."""
    base = {
        "box_size_track": 0.2,
        "initial_x_lo": 0.0,
        "initial_x_hi": 3.5,
        "initial_y_lo": -1.5,
        "initial_y_hi": 1.5,
        "initial_z_lo": -1.5,
        "initial_z_hi": 1.5,
        "max_branch": 30,
        "max_seeds": 0,
        "allow_3_frame": True,
        "progress_interval": 5000,
    }
    base.update(overrides)
    return base


def resolve_input_path(args: argparse.Namespace) -> tuple[str, bool, str | None]:
    """
    Return (h5path, is_synthetic, temp_dir).

    Default: actual Run1 ``rays_out_cpp.h5``. ``--synthetic`` builds two-particle data.
    """
    if args.synthetic:
        tmp_dir = tempfile.mkdtemp(prefix="graph_compare_")
        h5path = os.path.join(tmp_dir, "rays_out_cpp.h5")
        write_rays_h5(h5path, TwoParticleScenario(seed=42))
        return h5path, True, tmp_dir

    h5path = os.path.abspath(args.input)
    if not os.path.isfile(h5path):
        raise FileNotFoundError(f"rays file not found: {h5path}")
    return h5path, False, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare graph vs legacy on one 4-frame block",
    )
    p.add_argument(
        "--input",
        default=DEFAULT_RUN1,
        help=f"Path to rays_out_cpp.h5 (default: {DEFAULT_RUN1})",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated two-particle data instead of --input",
    )
    p.add_argument(
        "--block-start",
        type=int,
        default=0,
        help="First frame of the single 4-frame block (default: 0 → frames 0–3)",
    )
    p.add_argument("--max-seeds", type=int, default=None, help="Graph frame-0 seed cap (0=all)")
    p.add_argument("--max-branch", type=int, default=None)
    p.add_argument("--box-size-track", type=float, default=None)
    p.add_argument("--initial-x-lo", type=float, default=None)
    p.add_argument("--initial-x-hi", type=float, default=None)
    p.add_argument("--initial-y-lo", type=float, default=None)
    p.add_argument("--initial-y-hi", type=float, default=None)
    p.add_argument("--initial-z-lo", type=float, default=None)
    p.add_argument("--initial-z-hi", type=float, default=None)
    p.add_argument("--no-3-frame", action="store_true", help="Graph: 4-frame paths only")
    p.add_argument("--graph-min-length", type=int, default=4)
    p.add_argument("--legacy-min-length", type=int, default=4)
    p.add_argument("--pos-tol", type=float, default=0.05, help="Match tolerance (mm)")
    p.add_argument("--skip-legacy", action="store_true", help="Only time graph (no comparison)")
    p.add_argument(
        "--progress-interval",
        type=int,
        default=5000,
        help="Print progress every N seeds/particles (0 = phase lines only)",
    )
    p.add_argument(
        "--require-match",
        action="store_true",
        help="Exit 1 if graph and legacy tracks disagree",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        h5path, synthetic, tmp_dir = resolve_input_path(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    block_frames = [args.block_start + i for i in range(4)]
    if synthetic:
        print(f"Using synthetic data: {h5path}")
        kw = expected_link_params()
    else:
        print(f"Using real data: {h5path}")
        kw = run1_link_params()

    if args.max_branch is not None:
        kw["max_branch"] = args.max_branch
    if args.max_seeds is not None:
        kw["max_seeds"] = args.max_seeds
    if args.box_size_track is not None:
        kw["box_size_track"] = args.box_size_track
    if args.initial_x_lo is not None:
        kw["initial_x_lo"] = args.initial_x_lo
    if args.initial_x_hi is not None:
        kw["initial_x_hi"] = args.initial_x_hi
    if args.initial_y_lo is not None:
        kw["initial_y_lo"] = args.initial_y_lo
    if args.initial_y_hi is not None:
        kw["initial_y_hi"] = args.initial_y_hi
    if args.initial_z_lo is not None:
        kw["initial_z_lo"] = args.initial_z_lo
    if args.initial_z_hi is not None:
        kw["initial_z_hi"] = args.initial_z_hi
    kw["allow_3_frame"] = not args.no_3_frame
    kw["progress_interval"] = args.progress_interval
    params = LinkParams(**kw)

    print(f"Single 4-frame block: {block_frames}")
    print(f"Link params: {kw}")
    print("Running comparison...")

    report = compare_block(
        h5path,
        args.block_start,
        params,
        pos_tol=args.pos_tol,
        legacy_min_length=args.legacy_min_length,
        graph_min_length=args.graph_min_length,
        run_legacy=not args.skip_legacy,
    )

    print(format_report(report))
    print()

    if not args.skip_legacy:
        n_g = len(report.graph.tracks)
        n_l = len(report.legacy.tracks)
        n_m = len(report.matched_pairs)
        print(
            f"Summary: graph={n_g} legacy={n_l} matched={n_m} "
            f"graph_only={len(report.graph_only_ids)} legacy_only={len(report.legacy_only_ids)}"
        )

        if synthetic:
            assert n_g == 2, f"expected 2 graph tracks on synthetic data, got {n_g}"
            assert n_l == 2, f"expected 2 legacy tracks on synthetic data, got {n_l}"
            assert n_m == 2, f"expected 2 matched pairs, got {n_m}"
            assert not report.graph_only_ids and not report.legacy_only_ids
            print("Synthetic test PASSED: identical 2 tracks, both methods.")
        elif args.require_match and (report.graph_only_ids or report.legacy_only_ids):
            return 1

    if tmp_dir and os.environ.get("KEEP_COMPARE_TMP"):
        print(f"Temp data kept at {tmp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
