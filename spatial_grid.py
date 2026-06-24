"""
Uniform 3D grid for fast "particles inside axis-aligned box" queries.

Cell size should be on the order of the search box width.
"""

from __future__ import annotations

import numpy as np

from . import spatial_grid_numba as _sgn


def track_grid_cell_size(box_size_track: float) -> float:
    """Cell size for 0.2 mm (typical) tracking-box queries."""
    return max(float(box_size_track), 0.5)


def initial_grid_cell_size(
    offsets: tuple[float, float, float, float, float, float],
    box_size_track: float,
) -> float:
    """
    Coarse cell size for the large first-step initial search window.

    Derived from initial-box offset spans (often several mm). Wider cells
    mean fewer cell visits per seed; per-particle AABB checks stay exact.
    """
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = offsets
    max_span = max(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo)
    return max(max_span / 2.0, track_grid_cell_size(box_size_track))


class SpatialGrid:
    """Hash particles into cubic cells; query by AABB."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cell_size: float,
        *,
        use_numba: bool | None = None,
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

        if use_numba is None:
            use_numba = _sgn.numba_available()
        self._use_numba = bool(use_numba and _sgn.numba_available())
        self._cells_cache: dict[tuple[int, int, int], list[int]] | None = None
        self._query_out: np.ndarray | None = None

        if self._use_numba:
            self._flat = _sgn.build_flat_grid(
                self.x, self.y, self.z,
                self.xmin, self.ymin, self.zmin, self.cell_size,
            )
        else:
            self._flat = None
            inv = 1.0 / self.cell_size
            cells: dict[tuple[int, int, int], list[int]] = {}
            for i in range(self.n):
                key = self._cell_key(self.x[i], self.y[i], self.z[i], inv)
                cells.setdefault(key, []).append(i)
            self._cells_cache = cells

    def _ensure_query_buffer(self) -> np.ndarray:
        if self._query_out is None or self._query_out.size < self.n:
            self._query_out = np.empty(max(self.n, 1), dtype=np.int64)
        return self._query_out

    def _ensure_cells_cache(self) -> dict[tuple[int, int, int], list[int]]:
        if self._cells_cache is not None:
            return self._cells_cache
        if not self._use_numba or self._flat is None:
            self._cells_cache = {}
            return self._cells_cache

        inv = 1.0 / self.cell_size
        cells: dict[tuple[int, int, int], list[int]] = {}
        for i in range(self.n):
            key = self._cell_key(self.x[i], self.y[i], self.z[i], inv)
            cells.setdefault(key, []).append(i)
        self._cells_cache = cells
        return self._cells_cache

    @property
    def _cells(self) -> dict[tuple[int, int, int], list[int]]:
        """Backward-compatible cell map (lazy when using Numba queries)."""
        return self._ensure_cells_cache()

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

        if self._use_numba and self._flat is not None:
            return _sgn.query_aabb_flat(
                self.x,
                self.y,
                self.z,
                self.xmin,
                self.ymin,
                self.zmin,
                self.cell_size,
                self._flat,
                xlo,
                xhi,
                ylo,
                yhi,
                zlo,
                zhi,
                self._ensure_query_buffer(),
            )

        inv = 1.0 / self.cell_size
        ix0 = int(np.floor((xlo - self.xmin) * inv))
        ix1 = int(np.floor((xhi - self.xmin) * inv))
        iy0 = int(np.floor((ylo - self.ymin) * inv))
        iy1 = int(np.floor((yhi - self.ymin) * inv))
        iz0 = int(np.floor((zlo - self.zmin) * inv))
        iz1 = int(np.floor((zhi - self.zmin) * inv))

        cells = self._ensure_cells_cache()
        hits: list[int] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    for i in cells.get((ix, iy, iz), ()):
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

    @property
    def cells_dict(self) -> dict[tuple[int, int, int], list[int]]:
        """Cell → particle indices (for visualization)."""
        return self._ensure_cells_cache()