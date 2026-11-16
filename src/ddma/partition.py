"""Rectangular partitions of boxes in R^n.

A partition is given, for every coordinate i, by strictly increasing edges

    lo_i = e_i[0] < e_i[1] < ... < e_i[K_i] = hi_i,

which define the K_i intervals [e_i[j], e_i[j+1]) (the last one closed). Cells are
products of such intervals and are indexed by integer tuples q = (q_1, ..., q_n).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ["RectangularPartition"]


class RectangularPartition:
    """Product partition of the box [lo, hi] into rectangular cells."""

    def __init__(self, edges: Sequence[Iterable[float]]):
        self.edges = [np.asarray(e, dtype=float).copy() for e in edges]
        if not self.edges:
            raise ValueError("a partition needs at least one coordinate")
        for i, e in enumerate(self.edges):
            if e.ndim != 1 or e.size < 2:
                raise ValueError(f"edges[{i}] must be 1-D with at least two values")
            if not np.all(np.isfinite(e)):
                raise ValueError(f"edges[{i}] must be finite")
            if np.any(np.diff(e) <= 0):
                raise ValueError(f"edges[{i}] must be strictly increasing")
        # uniform axes allow O(1) cell lookup instead of a binary search
        self._uniform = [
            bool(np.allclose(np.diff(e), (e[-1] - e[0]) / (e.size - 1), rtol=1e-12, atol=0))
            for e in self.edges
        ]

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, lo, hi, n_cells) -> RectangularPartition:
        """Uniform partition of [lo, hi] with ``n_cells[i]`` cells along coordinate i."""
        lo = np.atleast_1d(np.asarray(lo, dtype=float))
        hi = np.atleast_1d(np.asarray(hi, dtype=float))
        n_cells = np.broadcast_to(np.atleast_1d(n_cells), lo.shape)
        return cls([np.linspace(a, b, int(k) + 1) for a, b, k in zip(lo, hi, n_cells)])

    @classmethod
    def with_margins(cls, lo, hi, inner_lo, inner_hi, n_inner) -> RectangularPartition:
        """Uniform cells on [inner_lo, inner_hi] plus one margin cell on each side.

        Edges are [lo, linspace(inner_lo, inner_hi, n_inner + 1), hi] per coordinate; a
        margin cell is dropped where inner_lo == lo (or inner_hi == hi). This is the
        partition used in the experiments of the TAC and ADHS papers.
        """
        lo, hi = np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float))
        ilo = np.broadcast_to(np.asarray(inner_lo, float), lo.shape)
        ihi = np.broadcast_to(np.asarray(inner_hi, float), lo.shape)
        nin = np.broadcast_to(np.atleast_1d(n_inner), lo.shape)
        edges = []
        for a, b, c, d, k in zip(lo, hi, ilo, ihi, nin):
            inner = np.linspace(c, d, int(k) + 1)
            e = np.concatenate(([a] if a < c else [], inner, [b] if b > d else []))
            edges.append(e)
        return cls(edges)

    @classmethod
    def from_data(cls, Z, lo, hi) -> RectangularPartition:
        """Data-induced partition: one breakpoint per distinct data coordinate.

        On this partition the fixed-partition simulating map coincides with the
        minimal simulating map (almost everywhere). The number of cells is up to
        (|K| + 1)^n, so this is only practical for small data sets or low dimension.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        lo = np.atleast_1d(np.asarray(lo, dtype=float))
        hi = np.atleast_1d(np.asarray(hi, dtype=float))
        edges = []
        for i in range(Z.shape[1]):
            inner = np.unique(Z[:, i])
            inner = inner[(inner > lo[i]) & (inner < hi[i])]
            edges.append(np.concatenate(([lo[i]], inner, [hi[i]])))
        return cls(edges)

    @classmethod
    def product(cls, *parts: RectangularPartition) -> RectangularPartition:
        """Cartesian product of partitions (coordinates concatenated in order)."""
        return cls([e for p in parts for e in p.edges])

    # ------------------------------------------------------------------ basic info
    @property
    def ndim(self) -> int:
        """Number of coordinates."""
        return len(self.edges)

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of cells along each coordinate."""
        return tuple(e.size - 1 for e in self.edges)

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return int(np.prod(self.shape))

    @property
    def lo(self) -> np.ndarray:
        """Lower corner of the partitioned box."""
        return np.array([e[0] for e in self.edges])

    @property
    def hi(self) -> np.ndarray:
        """Upper corner of the partitioned box."""
        return np.array([e[-1] for e in self.edges])

    def project(self, dims: Sequence[int]) -> RectangularPartition:
        """Partition of the selected coordinates."""
        return RectangularPartition([self.edges[d] for d in dims])

    def same_as(self, other: RectangularPartition) -> bool:
        """Whether both partitions have the same edges."""
        return self.ndim == other.ndim and all(
            a.shape == b.shape and np.array_equal(a, b) for a, b in zip(self.edges, other.edges)
        )

    def __repr__(self) -> str:
        return f"RectangularPartition(shape={self.shape}, lo={self.lo}, hi={self.hi})"

    # ------------------------------------------------------------------ geometry
    def search(self, i: int, values, side: str = "right") -> np.ndarray:
        """``np.searchsorted(self.edges[i], values, side)``, O(1) per value on uniform axes.

        On a uniform axis the index is computed arithmetically and then corrected by
        comparing with the actual edges, so the result is identical to searchsorted.
        """
        e = self.edges[i]
        v = np.asarray(values, dtype=float)
        if not self._uniform[i] or v.size < 64:
            return np.searchsorted(e, v, side=side)
        k = e.size - 1
        idx = np.floor((v - e[0]) * (k / (e[-1] - e[0]))).astype(np.intp) + 1
        np.clip(idx, 0, k + 1, out=idx)
        # exact correction: want e[idx-1] (<= or <) v (< or <=) e[idx]
        right = side == "right"
        while True:  # floating-point rounding can put idx one off; fix exactly
            left_edge = e[np.maximum(idx - 1, 0)]
            lo_bad = (idx > 0) & ((left_edge > v) if right else (left_edge >= v))
            right_edge = e[np.minimum(idx, k)]
            hi_bad = (idx <= k) & ((right_edge <= v) if right else (right_edge < v))
            if not (lo_bad.any() or hi_bad.any()):
                return idx
            idx += hi_bad.astype(np.intp) - lo_bad.astype(np.intp)

    def contains(self, Z) -> np.ndarray:
        """Boolean mask of points inside the closed box [lo, hi]."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        return np.all((Z >= self.lo) & (Z <= self.hi), axis=1)

    def locate(self, Z) -> np.ndarray:
        """Cell index (N, n) of each point. Points outside [lo, hi] raise ValueError."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[1] != self.ndim:
            raise ValueError(f"points have dimension {Z.shape[1]}, partition has {self.ndim}")
        if not np.all(self.contains(Z)):
            raise ValueError("some points lie outside the partitioned box")
        idx = np.empty(Z.shape, dtype=np.intp)
        for i, e in enumerate(self.edges):
            idx[:, i] = np.clip(self.search(i, Z[:, i], "right") - 1, 0, e.size - 2)
        return idx

    def cell_bounds(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper corners of the cells with index array ``idx`` (N, n)."""
        idx = np.atleast_2d(np.asarray(idx, dtype=np.intp))
        lo = np.stack([e[idx[:, i]] for i, e in enumerate(self.edges)], axis=1)
        hi = np.stack([e[idx[:, i] + 1] for i, e in enumerate(self.edges)], axis=1)
        return lo, hi

    def cell_centers(self, idx) -> np.ndarray:
        """Centres of the cells with subscripts ``idx`` (K, n)."""
        lo, hi = self.cell_bounds(idx)
        return 0.5 * (lo + hi)

    def all_indices(self) -> np.ndarray:
        """All cell indices in C order, shape (n_cells, n)."""
        return np.stack(np.unravel_index(np.arange(self.n_cells), self.shape), axis=1)

    def grid_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper corners of all cells as arrays of shape (*shape, n)."""
        lo = np.stack(np.meshgrid(*[e[:-1] for e in self.edges], indexing="ij"), axis=-1)
        hi = np.stack(np.meshgrid(*[e[1:] for e in self.edges], indexing="ij"), axis=-1)
        return lo, hi

    def box_to_cells(self, a, b) -> tuple[np.ndarray, np.ndarray]:
        """Index range of the cells that intersect the closed boxes [a, b].

        ``a`` and ``b`` are (N, n) arrays inside [lo, hi]. Returns (first, last)
        index arrays; the cells meeting [a_k, b_k] are exactly first_k <= q <= last_k.
        """
        a = np.atleast_2d(np.asarray(a, dtype=float))
        b = np.atleast_2d(np.asarray(b, dtype=float))
        first = np.empty(a.shape, dtype=np.intp)
        last = np.empty(b.shape, dtype=np.intp)
        for i, e in enumerate(self.edges):
            first[:, i] = np.clip(self.search(i, a[:, i], "right") - 1, 0, e.size - 2)
            last[:, i] = np.clip(self.search(i, b[:, i], "right") - 1, 0, e.size - 2)
        return first, last
