"""Tightest interval-valued simulating map on a fixed rectangular partition.

Given a partition (R_q) of the input box, the map

    sigma(q) = [ S_lo(lower corner of R_q), S_hi(upper corner of R_q) ]

composed with the quantizer z -> q (cell containing z) is a simulating map of the
data, and it is the smallest interval-valued map on this partition with that property
(TAC paper, simulating maps on fixed partitions). It is computed with the
paper's sweep: each data point is written into one cell, then a running min (upper
bound) or max (lower bound) is propagated along every axis. Cost:
O(|K| sum_i log K_i + |Q| n m) time, O(|Q| m) memory.
"""

from __future__ import annotations

import numpy as np

from ._signs import normalize_signs
from .msm import _as_bounds, _as_data
from .partition import RectangularPartition

try:  # optional compiled kernel (C++), identical results, ~10x faster on large data
    from . import _core
except ImportError:  # pragma: no cover
    _core = None


def compiled_backend_available() -> bool:
    """Whether the compiled C++ kernel (ddma._core) is installed."""
    return _core is not None

__all__ = ["PartitionedSimulatingMap"]


def _round32(a, down):
    """float32 copy of a, rounded toward -inf (down) or +inf (up)."""
    b = a.astype(np.float32)
    with np.errstate(invalid="ignore"):
        bad = (b > a) if down else (b < a)
    if bad.any():
        b[bad] = np.nextafter(b[bad], np.float32(-np.inf if down else np.inf))
    return b


def _sweep(out, reverse, ufunc):
    """Running min/max along every axis (in place where possible)."""
    for ax, rev in enumerate(reverse):
        if rev:
            view = np.flip(out, ax)
            ufunc.accumulate(view, axis=ax, out=view)
        else:
            ufunc.accumulate(out, axis=ax, out=out)
    return out


def _scatter(shape, flat, valid, values, init, ufunc):
    out = np.full(int(np.prod(shape)), init)
    if valid is None:
        ufunc.at(out, flat, values)
    elif valid.any():
        ufunc.at(out, flat[valid], values[valid])
    return out.reshape(shape)


class PartitionedSimulatingMap:
    """Interval-valued simulating map sigma on a fixed rectangular partition.

    Attributes
    ----------
    partition : RectangularPartition of the input box.
    lower, upper : arrays of shape (*partition.shape, m), the bounds of sigma(q).
        Entries are -inf/+inf where no data point constrains the cell.
    d_lo, d_hi, signs : as in :class:`ddma.MinimalSimulatingMap`.
    """

    def __init__(self, partition, lower, upper, d_lo, d_hi, signs):
        self.partition = partition
        self.lower = lower
        self.upper = upper
        self.d_lo = d_lo
        self.d_hi = d_hi
        self.signs = signs

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_data(
        cls, Z, Y, partition: RectangularPartition, d_lo=0.0, d_hi=0.0, signs=None,
        backend: str = "auto",
        compact: bool = False,
    ):
        """Compute sigma from data (Z, Y) on ``partition``.

        Data points outside the partitioned box are allowed: they still bound the cells
        they dominate (or are dominated by).

        ``backend``: "auto" (compiled kernel if installed, else NumPy), "cpp" or "numpy".
        Both backends give identical results.

        ``compact=True`` stores the bounds as float32 (half the memory), rounded outward
        (lower bounds down, upper bounds up), so the map stays a sound over-approximation.
        """
        Z, Y = _as_data(Z, Y)
        if Z.shape[1] != partition.ndim:
            raise ValueError(f"data has dimension {Z.shape[1]}, partition has {partition.ndim}")
        m, n, shape = Y.shape[1], Z.shape[1], partition.shape
        d_lo, d_hi = _as_bounds(d_lo, d_hi, m)
        signs = normalize_signs(signs, m, n)
        theta = d_hi - d_lo
        if backend not in ("auto", "cpp", "numpy"):
            raise ValueError("backend must be 'auto', 'cpp' or 'numpy'")
        if backend == "cpp" and _core is None:
            raise RuntimeError("compiled kernel ddma._core is not installed")
        if backend in ("auto", "cpp") and _core is not None:
            lower, upper = _core.fit_sigma(
                Z, Y, partition.edges, theta, np.ascontiguousarray(signs, dtype=np.int8),
                bool(compact),
            )
            return cls(partition, lower, upper, d_lo, d_hi, signs)

        # For every coordinate j: "ub" = index of the last cell whose upper edge is <= z,
        # "lb" = index of the first cell whose lower edge is >= z. A data point bounds
        # the upper bound of every cell q with q <= ub (componentwise, in the order of the
        # output) and the lower bound of every cell with q >= lb; the scatter writes it at
        # that corner and the sweep propagates it.
        last = np.array(shape) - 1
        strides = np.array([int(np.prod(shape[j + 1 :])) for j in range(n)], dtype=np.intp)
        ub = np.empty(Z.shape, dtype=np.intp)
        lb = np.empty(Z.shape, dtype=np.intp)
        for j in range(n):
            ub[:, j] = partition.search(j, Z[:, j], "right") - 2
            lb[:, j] = partition.search(j, Z[:, j], "left")
        ub_ok = ub >= 0
        lb_ok = lb <= last
        np.minimum(ub, last, out=ub)
        np.maximum(lb, 0, out=lb)

        def corner(s, upper):
            """Flat index and validity of the corner each point writes to."""
            flat = np.zeros(Z.shape[0], dtype=np.intp)
            valid = None
            for j in range(n):
                use_ub = (s[j] > 0) if upper else (s[j] < 0)
                if s[j] == 0:
                    col, ok = last[j], None
                elif use_ub:
                    col, ok = ub[:, j], ub_ok[:, j]
                else:
                    col, ok = lb[:, j], lb_ok[:, j]
                flat += col * strides[j]
                if ok is not None:
                    valid = ok.copy() if valid is None else (valid & ok)
            return flat, valid

        cache = {}
        lower = np.empty(shape + (m,))
        upper = np.empty(shape + (m,))
        for i in range(m):
            s = signs[i]
            key = tuple(s)
            if key not in cache:
                cache[key] = (corner(s, True), corner(s, False))
            (fu, vu), (fl, vl) = cache[key]
            upper[..., i] = _sweep(
                _scatter(shape, fu, vu, Y[:, i] + theta[i], np.inf, np.minimum), s >= 0, np.minimum
            )
            lower[..., i] = _sweep(
                _scatter(shape, fl, vl, Y[:, i] - theta[i], -np.inf, np.maximum), s <= 0, np.maximum
            )
        out = cls(partition, lower, upper, d_lo, d_hi, signs)
        return out.compact() if compact else out

    # ------------------------------------------------------------------ queries
    @property
    def m(self) -> int:
        """Output dimension."""
        return self.lower.shape[-1]

    @property
    def n(self) -> int:
        """Input dimension."""
        return self.partition.ndim

    def bounds_at(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """sigma(q) for cell indices ``idx`` (N, n)."""
        idx = tuple(np.atleast_2d(np.asarray(idx, dtype=np.intp)).T)
        return self.lower[idx], self.upper[idx]

    def bounds(self, z) -> tuple[np.ndarray, np.ndarray]:
        """sigma(phi(z)) at points ``z`` (M, n) inside the partitioned box."""
        return self.bounds_at(self.partition.locate(z))

    __call__ = bounds

    def f_bounds(self, z) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on the unknown function f: [lower - d_lo, upper - d_hi]."""
        lo, hi = self.bounds(z)
        return lo - self.d_lo, hi - self.d_hi

    def contains(self, z, y, tol: float = 1e-12) -> np.ndarray:
        """Whether each pair (z, y) satisfies y in sigma(phi(z))."""
        lo, hi = self.bounds(z)
        y = np.asarray(y, dtype=float).reshape(lo.shape)
        return np.all((lo - tol <= y) & (y <= hi + tol), axis=1)

    # ------------------------------------------------------------------ data updates
    def _check_compatible(self, other: PartitionedSimulatingMap):
        if not (
            self.partition.same_as(other.partition)
            and np.array_equal(self.d_lo, other.d_lo)
            and np.array_equal(self.d_hi, other.d_hi)
            and np.array_equal(self.signs, other.signs)
        ):
            raise ValueError("maps must share partition, disturbance bounds and sign pattern")

    def compact(self) -> PartitionedSimulatingMap:
        """Store the bounds as float32, rounded outward (sound). Returns self."""
        self.lower = _round32(self.lower, down=True)
        self.upper = _round32(self.upper, down=False)
        return self

    def intersect_(self, other: PartitionedSimulatingMap) -> PartitionedSimulatingMap:
        """In-place intersection (no copy of the bound arrays). Returns self."""
        self._check_compatible(other)
        if self.lower.dtype == np.float32 and other.lower.dtype != np.float32:
            olo, ohi = _round32(other.lower, True), _round32(other.upper, False)  # outward
        else:
            olo, ohi = other.lower, other.upper
        np.maximum(self.lower, olo, out=self.lower)
        np.minimum(self.upper, ohi, out=self.upper)
        return self

    def intersect(self, other: PartitionedSimulatingMap) -> PartitionedSimulatingMap:
        """Map of the union of both data sets (intersection of the two maps)."""
        self._check_compatible(other)
        return PartitionedSimulatingMap(
            self.partition,
            np.maximum(self.lower, other.lower),
            np.minimum(self.upper, other.upper),
            self.d_lo, self.d_hi, self.signs,
        )

    def update(self, Z, Y) -> PartitionedSimulatingMap:
        """Refine the map with new data, without revisiting the old data."""
        new = PartitionedSimulatingMap.from_data(
            Z, Y, self.partition, self.d_lo, self.d_hi, self.signs
        )
        return self.intersect(new)

    # ------------------------------------------------------------------ diagnostics
    def widths(self) -> np.ndarray:
        """upper - lower per cell and output (inf where unbounded)."""
        return self.upper - self.lower

    def bounded_fraction(self) -> float:
        """Fraction of cells whose interval is bounded in every output."""
        return float(np.mean(np.all(np.isfinite(self.widths()), axis=-1)))

    def is_empty(self) -> np.ndarray:
        """Cells where lower > upper. Never happens for consistent data."""
        return np.any(self.lower > self.upper, axis=-1)
