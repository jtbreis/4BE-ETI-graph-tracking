# Graph-based 4-frame particle tracking

Readable prototype that links stereomatched particle positions across four
consecutive frames using **path enumeration + global greedy assignment**.

**Detailed algorithm write-up:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md)

## How it works

1. **Load** four frames of `(x, y, z)` from `rays_out_cpp.h5`.
2. **Enumerate** valid paths per block:
   - 4-frame: f0 → f1 → f2 → f3
   - 3-frame from frame 0: f0 → f1 → f2
   - 3-frame from frame 1: f1 → f2 → f3
3. **Assign** paths globally (prefer longer paths, then lower cost).
4. **Write** `tracks_graph.h5part` via the existing `write_data_h5` helper.

## Benchmark notes (Run1, frames 0–3)

At ~42k–57k detections/frame:

| Config | Graph time | Tracks | Legacy time (all particles) |
|--------|------------|--------|------------------------------|
| 2000 seeds, max_branch=20 | ~5 s | 265 | ~427 s |
| 10000 seeds, max_branch=30 | ~13 s | 1405 | (not re-run) |

Legacy always iterates every particle in frames 0–2; graph can subsample frame-0 seeds via `--max-seeds`. Compare fairly by either using `--max-seeds 0` (all seeds) or subsampling legacy (not implemented here).

## Install

From the repo root (once per environment):

```bash
pip install -e 4BE-ETI -e Post-Processing -e graph_tracking
# or: pip install -r requirements.txt
```

This registers `graph_tracking`, `python_4be_eti`, and `check_tracking` as importable packages.
No `sys.path` bootstrap is required in scripts or notebooks.

## Run

```bash
# Interactive / Jupyter-style test (not part of the installed package)
python scripts/test_graph_tracking.py

# Quick test: first 4-frame block only
python graph_tracking/run_tracking.py \
  --input /path/to/rays_out_cpp.h5 \
  --output-dir /path/to/Run1 \
  --max-blocks 1

# Compare with legacy 4BE-ETI on the same block
python graph_tracking/benchmark.py \
  --input /path/to/rays_out_cpp.h5 \
  --max-blocks 1
```

## Synthetic two-particle test

```bash
python graph_tracking/test_two_particles.py
```

Jupyter-style cells; writes `synthetic_data/two_particles/rays_out_cpp.h5`. Two particles
in the x–y plane (~4 mm apart) move toward each other with v₀ = 0.08 mm/frame and
Δv = 0.01 mm/frame. Tuned for initial box **±1** and `box_size_track=0.2`.

## Graph vs legacy comparison

```bash
# One 4-frame block (frames 0–3) on Run1 — default input path
python graph_tracking/test_compare_graph_legacy.py --no-3-frame

# Fast synthetic sanity check
python graph_tracking/test_compare_graph_legacy.py --synthetic

# Custom rays file / block
python graph_tracking/test_compare_graph_legacy.py \
  --input data/.../rays_out_cpp.h5 \
  --block-start 0 \
  --no-3-frame
```

Reports runtime, track counts, matched pairs, and graph-only / legacy-only tracks.
Synthetic run asserts both methods find the same 2 four-frame tracks.

## Parameters

Same meaning as `FourFrameTracking` / `05_4frame_tracking.py`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--box-size-track` | 0.2 | Search box after initialization (mm) |
| `--initial-x-lo/hi` | 0 / 3.5 | Asymmetric frame-0 → frame-1 x gate |
| `--initial-y-lo/hi` | -1.5 / 1.5 | y gate |
| `--initial-z-lo/hi` | -1.5 / 1.5 | z gate |
| `--max-branch` | 30 | Max candidates kept per path stage |
| `--max-seeds` | 0 | Limit frame-0 seeds (0 = all; use for quick tests) |
| `--no-3-frame` | off | Disable 3-frame paths (4-frame only) |
| `--min-track-length` | 3 | Minimum frames per track written to h5part |

## Files

- `load_data.py` — read frames from HDF5
- `spatial_grid.py` — uniform grid for box queries
- `kinematics.py` — prediction helpers (matches 4BE-ETI)
- `link_block.py` — enumerate candidate 4-paths for one block
- `assign.py` — global greedy disjoint assignment
- `track_data.py` — build `basic_utils.a` for h5part export
- `run_tracking.py` — CLI entry point
- `benchmark.py` — timing + stats vs legacy tracker
