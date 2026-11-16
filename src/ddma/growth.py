"""Guaranteed model of one scalar function with bounded derivatives (growth cones).

For an unknown f : Z -> R with

    df/dz_j in [g_lo_j, g_hi_j]    and data    y_k = f(z_k) + d_k,  d_k in [d_lo, d_hi],

every data point bounds f + d everywhere by a cone with apex (z_k, y_k). On a rectangular
partition of Z, :class:`GrowthMap` keeps, for every cell, an interval [lower, upper] that
contains f(z) + d for all z in the cell and all d in [d_lo, d_hi]: the intersection of the
cones of all data seen so far, evaluated at the grid vertices (thesis, Chapter 3,
approaches 4 and 5) and, with ``local="slab"`` (default), on the cells themselves.

This is the growth model of :class:`ddma.BoundedDerivativeModel` (``method="growth"``) for
one output, usable on its own: e.g. to learn a vehicle's velocity as a function of
(heading, speed) or a turn rate as a function of (speed, steering), as in the compatible
models approach to safe learning-based MPC.

Adding data only shrinks the intervals, so every model built along the way stays sound.
:meth:`update` is exact (equals a refit on all data); single points take the per-point
path (about 0.1 ms on a 40 x 40 grid). :meth:`update_local` only updates a window of cells
around each new point (cheaper on large grids, never tighter).

Example::

    grid = RectangularPartition.uniform([-3.14, 0.1], [3.14, 2.0], [40, 40])
    vx = GrowthMap.from_data(Z, y, grid, g_lo=[-2, -1], g_hi=[2, 1], d_lo=-0.05, d_hi=0.05)
    lo, hi = vx.bounds(z_query)          # bounds on f(z) + d at the cells of z_query
    vx.update(z_new[None], [y_new])      # online
"""

from __future__ import annotations

import numpy as np

from ._bd_components import GrowthComponent
from .partition import RectangularPartition

__all__ = ["GrowthMap"]


class GrowthMap(GrowthComponent):
    """Interval model of f + D for one scalar output, on a rectangular partition.

    Parameters
    ----------
    partition : RectangularPartition of the input box Z.
    g_lo, g_hi : bounds on the partial derivatives of f (length n).
    d_lo, d_hi : bounds on the measurement disturbance.
    local : ``"slab"`` (default, tightest), ``"cell"`` or ``None``; see
        :class:`ddma.BoundedDerivativeModel`.
    final : ``"box"`` (default, approach 4) or ``"lp"`` (approach 5).
    backend : ``"auto"`` (default: compiled kernel for batch fits if installed), ``"cpp"``
        or ``"numpy"``; same model either way.

    Attributes
    ----------
    lower, upper : arrays of shape ``partition.shape``; bounds on f + d per cell
        (-inf / +inf where the data say nothing yet).
    n_data : number of data points added.
    """

    def __init__(self, partition: RectangularPartition, g_lo, g_hi, d_lo: float, d_hi: float,
                 local: str = "slab", final: str = "box", backend: str = "auto"):
        n = partition.ndim
        g_lo = np.broadcast_to(np.asarray(g_lo, float), (n,))
        g_hi = np.broadcast_to(np.asarray(g_hi, float), (n,))
        if d_lo > d_hi:
            raise ValueError("disturbance bounds must satisfy d_lo <= d_hi")
        if final not in ("box", "lp"):
            raise ValueError("final must be 'box' or 'lp'")
        super().__init__(partition, g_lo, g_hi, d_lo, d_hi, final=final, local=local,
                         backend=backend)
        self.n_data = 0

    @classmethod
    def from_data(cls, Z, y, partition, g_lo, g_hi, d_lo, d_hi, **kwargs) -> GrowthMap:
        """Model from data Z (K, n), y (K,)."""
        return cls(partition, g_lo, g_hi, d_lo, d_hi, **kwargs).update(Z, y)

    # ------------------------------------------------------------------ updates
    def update(self, Z, y) -> GrowthMap:
        """Add data (exact; in place; returns self)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        y = np.asarray(y, float).ravel()
        if Z.shape != (y.size, self.partition.ndim):
            raise ValueError("Z must have shape (K, n) and y shape (K,)")
        super().update(Z, y)
        self.n_data += y.size
        return self

    def update_local(self, Z, y, window) -> GrowthMap:
        """Add data, updating only ``window`` cells around each point (in place)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        y = np.asarray(y, float).ravel()
        super().update_local(Z, y, window)
        self.n_data += y.size
        return self

    def copy(self) -> GrowthMap:
        """An independent copy of the model."""
        out = GrowthMap(self.partition, self.g_lo, self.g_hi, self.d_lo, self.d_hi,
                        local=self.local, final=self.final, backend=self.backend)
        for a in ("v_upper", "v_lower", "lower", "upper"):
            setattr(out, a, getattr(self, a).copy())
        if self.local is not None:
            out.loc_upper, out.loc_lower = self.loc_upper.copy(), self.loc_lower.copy()
        out.n_data = self.n_data
        return out

    # ------------------------------------------------------------------ queries
    def cell_index(self, Z) -> np.ndarray:
        """Cell index (K, n) of each point (points outside the box: nearest edge cell)."""
        return self._cells(np.atleast_2d(np.asarray(Z, float)))

    def bounds(self, Z):
        """Bounds on f(z) + d at the cells of the points Z: (lower (K,), upper (K,))."""
        idx = tuple(self.cell_index(Z).T)
        return self.lower[idx], self.upper[idx]

    def f_bounds(self):
        """Bounds on f itself per cell: (lower - d_lo, upper - d_hi)."""
        return self.lower - self.d_lo, self.upper - self.d_hi

    def widths(self) -> np.ndarray:
        """upper - lower per cell (inf where unbounded)."""
        return self.upper - self.lower
