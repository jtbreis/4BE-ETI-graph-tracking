"""
Synthetic two-particle dataset for graph-tracking tests.

Two particles in the x–y plane (z = 0), separated along x, moving toward
each other with a small linear speed increase each frame.

Designed for:
  initial box offsets ±1 on each axis
  box_size_track = 0.2
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class TwoParticleScenario:
    """Kinematics tuned for initial box ±1 and box_size_track 0.2."""

    # Frame-0 positions (mm)
    x0_a: float = -2.0
    y0_a: float = 0.05
    x0_b: float = 2.0
    y0_b: float = -0.05
    z: float = 0.0

    v0: float = 0.08       # speed at frame 0→1 (mm / frame)
    dv: float = 0.01       # speed increase per frame
    noise_std: float = 0.015  # measurement noise (mm)

    n_frames: int = 4
    seed: int = 42


def _positions_along_x(
    x_start: float,
    y_start: float,
    vx_sign: float,
    scenario: TwoParticleScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate position with slowly increasing speed."""
    xs = np.zeros(scenario.n_frames)
    ys = np.zeros(scenario.n_frames)
    xs[0] = x_start
    ys[0] = y_start
    v = scenario.v0
    for t in range(scenario.n_frames - 1):
        xs[t + 1] = xs[t] + vx_sign * v
        ys[t + 1] = ys[t]  # stay in plane; optional tiny y drift could be added
        v += scenario.dv
    xs += rng.normal(0, scenario.noise_std, scenario.n_frames)
    ys += rng.normal(0, scenario.noise_std, scenario.n_frames)
    return xs, ys


def build_trajectories(
    scenario: TwoParticleScenario | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Returns (x, y, z) arrays shaped (2, n_frames) and frame indices.

    Row 0 = particle A (moves +x), row 1 = particle B (moves −x).
    """
    sc = scenario or TwoParticleScenario()
    rng = np.random.default_rng(sc.seed)
    xa, ya = _positions_along_x(sc.x0_a, sc.y0_a, +1.0, sc, rng)
    xb, yb = _positions_along_x(sc.x0_b, sc.y0_b, -1.0, sc, rng)
    x = np.vstack([xa, xb])
    y = np.vstack([ya, yb])
    z = np.full_like(x, sc.z)
    frames = list(range(sc.n_frames))
    return x, y, z, frames


def write_rays_h5(
    outpath: str,
    scenario: TwoParticleScenario | None = None,
) -> dict:
    """
    Write ``rays_out_cpp.h5`` with two particles per frame.

    Returns a truth dict for tests (indices and positions).
    """
    sc = scenario or TwoParticleScenario()
    x, y, z, frames = build_trajectories(sc)
    os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)

    with h5py.File(outpath, "w") as f:
        for fi, frame_idx in enumerate(frames):
            grp = f.create_group(f"frame{frame_idx:05d}")
            # shape (4, n_pts): rows are x, y, z, e
            n = 2
            xyze = np.zeros((4, n), dtype=np.float64)
            xyze[0, :] = x[:, fi]
            xyze[1, :] = y[:, fi]
            xyze[2, :] = z[:, fi]
            xyze[3, :] = 1.0
            grp.create_dataset("xyze", data=xyze)
            grp.create_dataset("diameter", data=np.full(n, 0.1))
            grp.create_dataset("intensity", data=np.full(n, 100.0))
            grp.create_dataset("mass", data=np.ones(n))

    truth = {
        "scenario": sc,
        "x": x,
        "y": y,
        "z": z,
        "frames": frames,
        "particle_ids": [0, 1],  # row index in synthetic data
    }
    return truth


def expected_link_params() -> dict:
    """LinkParams values that match this scenario."""
    return {
        "box_size_track": 0.2,
        "initial_x_lo": -1.0,
        "initial_x_hi": 1.0,
        "initial_y_lo": -1.0,
        "initial_y_hi": 1.0,
        "initial_z_lo": -1.0,
        "initial_z_hi": 1.0,
        "max_branch": 10,
        "max_seeds": 0,
    }
