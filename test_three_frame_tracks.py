# %% [markdown]
# # Test 3-frame tracking on real data
#
# Runs graph linking with ``allow_3_frame=True`` on **one** 4-frame block of
# Run1 ``rays_out_cpp.h5``. Reports how many 3-frame vs 4-frame tracks are found
# (starting at frame 0 or frame 1).
#
# Use ``--synthetic`` for the tiny hand-built 3-path sanity check.

# %%
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
from graph_tracking.assign import greedy_assign_paths
from graph_tracking.link_block import LinkParams, link_block
from graph_tracking.load_data import load_block
from graph_tracking.synthetic.two_particles import expected_link_params
from graph_tracking.track_data import block_to_tracking_data
from python_4be_eti.utils.basic_utils import write_data_h5

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


def _write_synthetic_three_frame_h5(path: str) -> None:
    """Tiny dataset: one 4f, one 3f@f1, one 3f@f0 path (for --synthetic)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames_xyz = {
        0: [(0.0, 0.0, 0.0), None, (3.0, 0.0, 0.0)],
        1: [(0.1, 0.0, 0.0), (-2.0, 0.1, 0.0), (3.08, 0.0, 0.0)],
        2: [(0.21, 0.0, 0.0), (-1.9, 0.1, 0.0), (3.17, 0.0, 0.0)],
        3: [(0.33, 0.0, 0.0), (-1.78, 0.1, 0.0), (10.0, 0.0, 0.0)],
    }
    with h5py.File(path, "w") as f:
        for fi in range(4):
            pts = [p for p in frames_xyz[fi] if p is not None]
            n = len(pts)
            xyze = np.zeros((4, n))
            for j, (x, y, z) in enumerate(pts):
                xyze[0, j], xyze[1, j], xyze[2, j], xyze[3, j] = x, y, z, 1.0
            grp = f.create_group(f"frame{fi:05d}")
            grp.create_dataset("xyze", data=xyze)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test 3-frame graph tracking")
    p.add_argument("--input", default=DEFAULT_RUN1, help="rays_out_cpp.h5")
    p.add_argument("--synthetic", action="store_true", help="Use tiny synthetic 3-path data")
    p.add_argument("--block-start", type=int, default=0, help="First frame of 4-frame block")
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--max-branch", type=int, default=None)
    p.add_argument("--progress-interval", type=int, default=5000)
    p.add_argument(
        "--output",
        default=None,
        help="h5part path (default: next to rays file, tracks_graph_3frame.h5part)",
    )
    p.add_argument("--no-write", action="store_true", help="Skip writing h5part")
    return p.parse_args()


def summarize_assigned(assigned, block) -> None:
    """Print counts by path type and example tracks."""
    by_type: dict[tuple[int, int], int] = {}
    for p in assigned:
        key = (p.start_layer, p.length)
        by_type[key] = by_type.get(key, 0) + 1

    print("\nAssigned tracks by type (start_layer, length) → frames:")
    labels = {
        (0, 4): "4-frame f0→f1→f2→f3",
        (0, 3): "3-frame f0→f1→f2",
        (1, 3): "3-frame f1→f2→f3",
    }
    for key in sorted(by_type):
        fr = block.frame_indices
        if key == (0, 4):
            frame_str = f"{fr[0]}–{fr[3]}"
        elif key == (0, 3):
            frame_str = f"{fr[0]}–{fr[2]}"
        elif key == (1, 3):
            frame_str = f"{fr[1]}–{fr[3]}"
        else:
            frame_str = str(key)
        print(f"  {labels.get(key, key)}: {by_type[key]}  (frames {frame_str})")

    print(f"\nTotal assigned: {len(assigned)}")
    return by_type


# %%
args = parse_args() if __name__ == "__main__" else argparse.Namespace(
    input=DEFAULT_RUN1,
    synthetic=False,
    block_start=0,
    max_seeds=None,
    max_branch=None,
    progress_interval=5000,
    output=None,
    no_write=False,
)

tmp_dir = None
if args.synthetic:
    tmp_dir = tempfile.mkdtemp(prefix="graph_3frame_")
    rays_h5 = os.path.join(tmp_dir, "rays_out_cpp.h5")
    _write_synthetic_three_frame_h5(rays_h5)
    print(f"Using synthetic data: {rays_h5}")
    kw = expected_link_params()
    kw["allow_3_frame"] = True
else:
    rays_h5 = os.path.abspath(args.input)
    if not os.path.isfile(rays_h5):
        raise FileNotFoundError(f"rays file not found: {rays_h5}")
    print(f"Using real data: {rays_h5}")
    kw = run1_link_params()

if args.max_branch is not None:
    kw["max_branch"] = args.max_branch
if args.max_seeds is not None:
    kw["max_seeds"] = args.max_seeds
kw["progress_interval"] = args.progress_interval
params = LinkParams(**kw)

block_frames = [args.block_start + i for i in range(4)]
print(f"Single 4-frame block: {block_frames}")
print(f"Link params: {kw}")

# %%
t0 = time.perf_counter()
block = load_block(rays_h5, args.block_start)
print(
    f"Particles/frame: {[f.n for f in block.frames]} "
    f"(load {time.perf_counter() - t0:.1f}s)"
)

result = link_block(block, params)
assigned = greedy_assign_paths(result.candidates)

print("\nCandidates:", {k: result.stats[k] for k in result.stats if k.startswith("n_candidates")})
print(f"Link + assign: {time.perf_counter() - t0:.1f}s")

by_type = summarize_assigned(assigned, block)

# Show a few examples of each 3-frame type
for want in [(0, 3), (1, 3), (0, 4)]:
    examples = [p for p in assigned if (p.start_layer, p.length) == want][:3]
    if examples:
        label = {(0, 4): "4-frame", (0, 3): "3-frame@f0", (1, 3): "3-frame@f1"}[want]
        print(f"\nExample {label} paths:")
        for p in examples:
            print(f"  start={p.start_layer} indices={p.indices} cost={p.cost:.4f}")

# %%
if args.synthetic:
    assert (0, 4) in by_type, "missing 4-frame track"
    assert (1, 3) in by_type, "missing 3-frame track starting at frame 1"
    assert (0, 3) in by_type, "missing 3-frame track starting at frame 0"
    print("\nSynthetic sanity check PASSED (all three path types found).")

block_data = block_to_tracking_data(block, assigned)
length_hist: dict[int, int] = {}
for tid in sorted(set(block_data.Count) - {0}):
    frames = sorted(int(f) for f in np.unique(block_data.Slice[block_data.Count == tid]))
    length_hist[len(frames)] = length_hist.get(len(frames), 0) + 1

print("\nTrack length histogram (frames per track id):")
for length in sorted(length_hist):
    print(f"  {length} frames: {length_hist[length]} tracks")

# %%
if not args.no_write:
    out_dir = os.path.dirname(rays_h5)
    out_path = args.output or os.path.join(out_dir, "tracks_graph_3frame.h5part")
    if os.path.isfile(out_path):
        os.remove(out_path)
    write_data_h5(
        block_data,
        out_dir,
        block.frame_indices,
        output_h5part=out_path,
        dt=1e-3,
        rep_rate_hz=10.0,
        track_length=4,
        min_track_length=3,
        position_scale_to_m=1e-3,
        position_units_str="mm",
        use_bspline=True,
        verbose=False,
    )
    print(f"\nWrote: {out_path}")

# %%
