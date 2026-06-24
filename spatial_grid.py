"""
Uniform 3D grid for fast "particles inside axis-aligned box" queries.

Cell size should be on the order of the search box width.
"""

from __future__ import annotations

import numpy as np


class SpatialGrid:
    """Hash particles into cubic cells; query by AABB."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cell_size: float,
    ):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.n = self.x.size
        self.cell_size = float(cell_size)
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")

        self.xmin = float(self.x.min()) if self.n else 0.0
        self.ymin = float(self.y.min()) if self.n else 0.0
        self.zmin = float(self.z.min()) if self.n else 0.0

        self._cells: dict[tuple[int, int, int], list[int]] = {}
        inv = 1.0 / self.cell_size
        for i in range(self.n):
            key = self._cell_key(self.x[i], self.y[i], self.z[i], inv)
            self._cells.setdefault(key, []).append(i)

    def _cell_key(self, x, y, z, inv=None) -> tuple[int, int, int]:
        if inv is None:
            inv = 1.0 / self.cell_size
        return (
            int(np.floor((x - self.xmin) * inv)),
            int(np.floor((y - self.ymin) * inv)),
            int(np.floor((z - self.zmin) * inv)),
        )

    def query_aabb(
        self,
        xlo: float,
        xhi: float,
        ylo: float,
        yhi: float,
        zlo: float,
        zhi: float,
    ) -> np.ndarray:
        """Return indices of particles inside the closed AABB."""
        if self.n == 0:
            return np.array([], dtype=np.int64)

        inv = 1.0 / self.cell_size
        ix0 = int(np.floor((xlo - self.xmin) * inv))
        ix1 = int(np.floor((xhi - self.xmin) * inv))
        iy0 = int(np.floor((ylo - self.ymin) * inv))
        iy1 = int(np.floor((yhi - self.ymin) * inv))
        iz0 = int(np.floor((zlo - self.zmin) * inv))
        iz1 = int(np.floor((zhi - self.zmin) * inv))

        hits: list[int] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    for i in self._cells.get((ix, iy, iz), ()):
                        if (
                            xlo <= self.x[i] <= xhi
                            and ylo <= self.y[i] <= yhi
                            and zlo <= self.z[i] <= zhi
                        ):
                            hits.append(i)
        return np.asarray(hits, dtype=np.int64)

    def query_center_box(
        self,
        cx: float,
        cy: float,
        cz: float,
        half: float,
    ) -> np.ndarray:
        h = float(half)
        return self.query_aabb(cx - h, cx + h, cy - h, cy + h, cz - h, cz + h)

    def _aabb_cell_index_ranges(
        self,
        xlo: float,
        xhi: float,
        ylo: float,
        yhi: float,
        zlo: float,
        zhi: float,
    ) -> tuple[range, range, range]:
        inv = 1.0 / self.cell_size
        ix0 = int(np.floor((xlo - self.xmin) * inv))
        ix1 = int(np.floor((xhi - self.xmin) * inv))
        iy0 = int(np.floor((ylo - self.ymin) * inv))
        iy1 = int(np.floor((yhi - self.ymin) * inv))
        iz0 = int(np.floor((zlo - self.zmin) * inv))
        iz1 = int(np.floor((zhi - self.zmin) * inv))
        return range(ix0, ix1 + 1), range(iy0, iy1 + 1), range(iz0, iz1 + 1)

    def iter_cells_overlapping_aabb(
        self,
        xlo: float,
        xhi: float,
        ylo: float,
        yhi: float,
        zlo: float,
        zhi: float,
    ):
        """Yield ``(ix, iy, iz)`` keys for cells overlapping the closed AABB."""
        ix_rng, iy_rng, iz_rng = self._aabb_cell_index_ranges(
            xlo, xhi, ylo, yhi, zlo, zhi)
        for ix in ix_rng:
            for iy in iy_rng:
                for iz in iz_rng:
                    yield ix, iy, iz

    def cell_bounds(
        self, ix: int, iy: int, iz: int
    ) -> tuple[float, float, float, float, float, float]:
        """World-space bounds ``(xlo, xhi, ylo, yhi, zlo, zhi)`` for one cell."""
        cs = self.cell_size
        return (
            self.xmin + ix * cs,
            self.xmin + (ix + 1) * cs,
            self.ymin + iy * cs,
            self.ymin + (iy + 1) * cs,
            self.zmin + iz * cs,
            self.zmin + (iz + 1) * cs,
        )
