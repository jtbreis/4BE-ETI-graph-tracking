"""Motion prediction helpers (same physics as 4BE-ETI particle_tracking_code)."""

from __future__ import annotations

import numpy as np


def predict_next_constant_velocity(p0: float, p1: float) -> float:
    """Position at n+2 from p0, p1 at n, n+1 (uniform dt): x_pred = 2*p1 - p0."""
    return 2.0 * p1 - p0


def predict_third_frame_position(p0: float, p1: float, p2: float) -> float:
    """
    Position at n+3 from p0, p1, p2 (quadratic through three samples).
    Matches ``predict_third_frame_position`` in 4BE-ETI.
    """
    return 3.0 * p2 - 3.0 * p1 + p0


def predict_third_frame_position_xyz(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    return 3.0 * p2 - 3.0 * p1 + p0


def predict_prev_constant_velocity_xyz(p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """One step backward in time: p1 from p2, p3 (mirror of 2*p1 - p0)."""
    return 2.0 * p2 - p3


def predict_first_frame_position_xyz(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    """Extrapolate p0 from p1, p2, p3 (mirror of 3*p2 - 3*p1 + p0)."""
    return 3.0 * p1 - 3.0 * p2 + p3


def predict_prev_constant_velocity(p2: float, p3: float) -> float:
    return 2.0 * p2 - p3


def predict_first_frame_position(p1: float, p2: float, p3: float) -> float:
    return 3.0 * p1 - 3.0 * p2 + p3


def distance_xyz(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))
