"""Numba-accelerated flat CSR spatial grid queries."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def numba_available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True)
    def _scatter_particles(flat_ids: np.ndarray, starts: np.ndarray, indices: np.ndarray) -> None:
        cursor = starts.copy()
        n = flat_ids.size
        for i in range(n):
            pos = cursor[flat_ids[i]]
            indices[pos] = i
            cursor[flat_ids[i]] = pos + 1

    @njit(cache=True)
    def _query_aabb_numba(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xmin: float,
        ymin: float,
        zmin: float,
        inv_cs: float,
        ix_min: int,
        iy_min: int,
        iz_min: int,
        nx: int,
        ny: int,
        nz: int,
        cell_starts: np.ndarray,
        cell_indices: np.ndarray,
        xlo: float,
        xhi: float,
        ylo: float,
        yhi: float,
        zlo: float,
        zhi: float,
        out: np.ndarray,
    ) -> int:
        ix0 = int(np.floor((xlo - xmin) * inv_cs))
        ix1 = int(np.floor((xhi - xmin) * inv_cs))
        iy0 = int(np.floor((ylo - ymin) * inv_cs))
        iy1 = int(np.floor((yhi - ymin) * inv_cs))
        iz0 = int(np.floor((zlo - zmin) * inv_cs))
        iz1 = int(np.floor((zhi - zmin) * inv_cs))

        n_flat = nx * ny * nz
        n_hit = 0
        out_size = out.size

        for ix in range(ix0, ix1 + 1):
            dx = ix - ix_min
            for iy in range(iy0, iy1 + 1):
                dy = iy - iy_min
                base_y = dx + nx * dy
                for iz in range(iz0, iz1 + 1):
                    flat = base_y + nx * ny * (iz - iz_min)
                    if flat < 0 or flat >= n_flat:
                        continue
                    s = cell_starts[flat]
                    e = cell_starts[flat + 1]
                    for k in range(s, e):
                        i = cell_indices[k]
                        if (
                            xlo <= x[i] <= xhi
                            and ylo <= y[i] <= yhi
                            and zlo <= z[i] <= zhi
                        ):
                            if n_hit < out_size:
                                out[n_hit] = i
                            n_hit += 1
        return n_hit

else:  # pragma: no cover
    _scatter_particles = None
    _query_aabb_numba = None


def build_flat_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmin: float,
    ymin: float,
    zmin: float,
    cell_size: float,
) -> dict:
    """Build CSR arrays for uniform-grid neighbor lookup."""
    n = int(x.size)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return {
            "ix_min": 0,
            "iy_min": 0,
            "iz_min": 0,
            "nx": 0,
            "ny": 0,
            "nz": 0,
            "cell_starts": empty,
            "cell_indices": empty,
        }

    inv = 1.0 / cell_size
    ix = np.floor((x - xmin) * inv).astype(np.int64)
    iy = np.floor((y - ymin) * inv).astype(np.int64)
    iz = np.floor((z - zmin) * inv).astype(np.int64)

    ix_min = int(ix.min())
    ix_max = int(ix.max())
    iy_min = int(iy.min())
    iy_max = int(iy.max())
    iz_min = int(iz.min())
    iz_max = int(iz.max())

    nx = ix_max - ix_min + 1
    ny = iy_max - iy_min + 1
    nz = iz_max - iz_min + 1
    flat = (ix - ix_min) + nx * ((iy - iy_min) + ny * (iz - iz_min))

    ncells = nx * ny * nz
    counts = np.bincount(flat, minlength=ncells)
    starts = np.zeros(ncells + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    indices = np.empty(n, dtype=np.int64)
    _scatter_particles(flat, starts, indices)

    return {
        "ix_min": ix_min,
        "iy_min": iy_min,
        "iz_min": iz_min,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "cell_starts": starts,
        "cell_indices": indices,
    }


def query_aabb_flat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmin: float,
    ymin: float,
    zmin: float,
    cell_size: float,
    flat: dict,
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
    zlo: float,
    zhi: float,
    out: np.ndarray,
) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=np.int64)

    inv = 1.0 / cell_size
    n_hit = _query_aabb_numba(
        x,
        y,
        z,
        xmin,
        ymin,
        zmin,
        inv,
        flat["ix_min"],
        flat["iy_min"],
        flat["iz_min"],
        flat["nx"],
        flat["ny"],
        flat["nz"],
        flat["cell_starts"],
        flat["cell_indices"],
        xlo,
        xhi,
        ylo,
        yhi,
        zlo,
        zhi,
        out,
    )
    return out[:n_hit].copy()
