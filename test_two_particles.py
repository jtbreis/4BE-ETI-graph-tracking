# %% [markdown]
# # Synthetic test: two particles approaching in a 2D plane
#
# Generates a tiny ``rays_out_cpp.h5``, runs graph tracking with
# initial box ±1 and ``box_size_track=0.2``, and checks that both
# particles are linked correctly across 4 frames.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from graph_tracking.assign import greedy_assign_paths
from graph_tracking.link_block import LinkParams, link_block
from graph_tracking.load_data import load_block
from graph_tracking.synthetic.two_particles import (
    TwoParticleScenario,
    expected_link_params,
    write_rays_h5,
)
from graph_tracking.track_data import block_to_tracking_data
from python_4be_eti.utils.basic_utils import write_data_h5

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
# --- Output paths ---
SYNTH_DIR = os.path.join(PROJECT_ROOT, "graph_tracking", "synthetic_data", "two_particles")
rays_h5 = os.path.join(SYNTH_DIR, "rays_out_cpp.h5")
tracks_h5 = os.path.join(SYNTH_DIR, "tracks_graph.h5part")

scenario = TwoParticleScenario(
    x0_a=-2.0,
    y0_a=0.05,
    x0_b=2.0,
    y0_b=-0.05,
    v0=0.08,
    dv=0.01,
    noise_std=0.015,
    n_frames=4,
    seed=42,
)
truth = write_rays_h5(rays_h5, scenario)
print(f"Wrote synthetic rays: {rays_h5}")
print("Frame-0 positions (particle A, B):")
for p in range(2):
    print(
        f"  p{p}: x={truth['x'][p, 0]:.3f}, y={truth['y'][p, 0]:.3f}, "
        f"z={truth['z'][p, 0]:.3f}"
    )

# %%
# --- Tracking (initial box ±1, box_size_track 0.2) ---
kw = expected_link_params()
params = LinkParams(**kw)

block = load_block(rays_h5, 0)
result = link_block(block, params)
assigned = greedy_assign_paths(result.candidates)

print(f"Candidates: {len(result.candidates)}")
print(f"Tracks:     {len(assigned)}")
assert len(assigned) == 2, f"Expected 2 tracks, got {len(assigned)}"

# %%
# --- Bidirectional mode: same tracks must validate forward and backward ---
params_bidir = LinkParams(**{**kw, "require_bidirectional": True})
result_bidir = link_block(block, params_bidir)
assigned_bidir = greedy_assign_paths(result_bidir.candidates)
print(f"Bidirectional candidates: {result_bidir.stats['n_candidates']}")
print(f"Bidirectional 4-frame agree: {result_bidir.stats['n_bidirectional_4frame']}")
assert len(assigned_bidir) == 2, f"Expected 2 bidirectional tracks, got {len(assigned_bidir)}"

# %%
# --- Verify each synthetic particle is matched to itself (index 0↔0, 1↔1) ---
def path_matches_particle(path, particle_row: int) -> bool:
    # In synthetic h5, particle A is index 0, B is index 1 in every frame
    return (
        path.start_layer == 0
        and path.length == 4
        and path.indices == (particle_row, particle_row, particle_row, particle_row)
    )


matched = {0: False, 1: False}
for path in assigned:
    for p in (0, 1):
        if path_matches_particle(path, p):
            matched[p] = True
            print(f"Particle {p} OK — cost={path.cost:.5f}, indices={path.indices}")

assert all(matched.values()), f"Not all particles matched: {matched}"
print("Assertion passed: both particles tracked with correct indices.")

# %%
# --- Write h5part ---
if os.path.isfile(tracks_h5):
    os.remove(tracks_h5)

block_data = block_to_tracking_data(block, assigned)
write_data_h5(
    block_data,
    SYNTH_DIR,
    block.frame_indices,
    output_h5part=tracks_h5,
    dt=1e-3,
    rep_rate_hz=10.0,
    track_length=4,
    position_scale_to_m=1e-3,
    position_units_str="mm",
    use_bspline=True,
    min_track_length=3,
    verbose=False,
)
print(f"Wrote: {tracks_h5}")

# %%
# --- Plot trajectories in the x–y plane ---
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["tab:blue", "tab:orange"]
labels = ["Particle A (+x)", "Particle B (−x)"]

for p in range(2):
    ax.plot(truth["x"][p], truth["y"][p], "o-", color=colors[p], label=labels[p], lw=2)
    ax.annotate("t=0", (truth["x"][p, 0], truth["y"][p, 0]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.annotate("t=3", (truth["x"][p, -1], truth["y"][p, -1]), textcoords="offset points", xytext=(4, 4), fontsize=8)

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title("Two-particle synthetic test (approaching, accelerating slightly)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
plt.show()

# %%
# --- Print separation vs time (should decrease) ---
for fi in range(scenario.n_frames):
    dx = truth["x"][1, fi] - truth["x"][0, fi]
    dy = truth["y"][1, fi] - truth["y"][0, fi]
    print(f"Frame {fi}: separation = {np.hypot(dx, dy):.3f} mm")

# %%
