"""Finite symbolic abstractions built from data.

System (L4DC 2021 / ADHS 2021 / thesis Ch. 4):

    x+ in f(x, u) + g(x, u) + D,

with f unknown and monotone, g known, D = [d_lo, d_hi] a known box. From transitions
(x_k, u_k, x_k') one forms y_k = x_k' - g(x_k, u_k), learns a simulating map sigma of
f + D on a partition, and builds the symbolic system on state cells q and input cells p:

    p enabled at q        <=>  sigma(q, p) + G(q, p)  is bounded and inside X,
    Delta(q, p)           =    { q' : X_q' meets sigma(q, p) + G(q, p) },

where G(q, p) is an interval containing g(X_q, U_p). This symbolic system is
alternatingly simulated by the true system, so controllers synthesized on it are
correct for the true system. Because sigma + G is a box, Delta(q, p) is a box of cells
and is stored by its first and last cell index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Optional

import numpy as np

from .partition import RectangularPartition
from .partitioned import PartitionedSimulatingMap

__all__ = ["SymbolicAbstraction", "affine_bounds", "transitions_to_data"]

BoundsFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple]


def affine_bounds(A, B=None, c=None) -> BoundsFn:
    """Exact interval image of g(x, u) = A x + B u + c over boxes.

    Returns a function (x_lo, x_hi, u_lo, u_hi) -> (g_lo, g_hi) usable as ``g_bounds``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = None if B is None else np.atleast_2d(np.asarray(B, dtype=float))
    c = np.zeros(A.shape[0]) if c is None else np.asarray(c, dtype=float)

    def bounds(x_lo, x_hi, u_lo, u_hi):
        mid = 0.5 * (x_lo + x_hi) @ A.T + c
        rad = 0.5 * (x_hi - x_lo) @ np.abs(A).T
        if B is not None:
            mid = mid + 0.5 * (u_lo + u_hi) @ B.T
            rad = rad + 0.5 * (u_hi - u_lo) @ np.abs(B).T
        return mid - rad, mid + rad

    return bounds


def transitions_to_data(X, U, Xnext, g: Optional[Callable] = None):
    """Data (z_k, y_k) = ((x_k, u_k), x_k' - g(x_k, u_k)) for learning f + D."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    Xnext = np.atleast_2d(np.asarray(Xnext, dtype=float))
    Y = Xnext if g is None else Xnext - g(X, U)
    return np.hstack([X, U]), Y


class SymbolicAbstraction:
    """Symbolic abstraction on state cells x input cells.

    Parameters
    ----------
    state_partition, input_partition : RectangularPartition of X and U.
    sim_map : PartitionedSimulatingMap of f + D. Its inputs are the coordinates
        ``features`` of the stacked vector (x, u) (default: all of them), and its
        partition must be the projection of state_partition x input_partition on
        these coordinates. Leaving out coordinates states that f does not depend on them.
    g_bounds : optional function (x_lo, x_hi, u_lo, u_hi) -> (g_lo, g_hi) giving an
        interval containing g over a cell pair (rows are cell pairs). Use
        :func:`affine_bounds` for affine g. None means g = 0.
        ``sim_map`` may instead be a model with a ``successor_bounds`` method (e.g.
        :class:`ddma.BoundedDerivativeModel`), which already includes g.
    assumption : optional (lo, hi) state box describing an environment assumption, for
        assume-guarantee contracts ("if the leader's speed stays in [a, b] then ...").
        Successor sets are restricted to the state cells inside this box (on the
        constrained coordinates), and clamped into it if they would miss it entirely,
        which is conservative: trajectories that break the assumption are not the
        controller's responsibility. Use -inf / inf for unconstrained coordinates; finite
        bounds should lie on cell edges.
    lazy : if True, transitions are not precomputed: :meth:`pair_boxes` evaluates them on
        demand. Memory is then O(|Q|) instead of O(|Q| |P|). Use with
        :func:`ddma.safety`, which evaluates only the pairs it needs; the other
        synthesis functions need the eager abstraction.
    """

    def __init__(
        self,
        state_partition: RectangularPartition,
        input_partition: RectangularPartition,
        sim_map: PartitionedSimulatingMap,
        g_bounds: Optional[BoundsFn] = None,
        features: Optional[Sequence[int]] = None,
        chunk: int = 1_000_000,
        assumption=None,
        lazy: bool = False,
    ):
        self.X = state_partition
        self.U = input_partition
        self.sim_map = sim_map
        n, nu = self.X.ndim, self.U.ndim
        self.features = list(range(n + nu)) if features is None else list(features)
        joint = RectangularPartition.product(self.X, self.U)
        if not joint.project(self.features).same_as(sim_map.partition):
            raise ValueError(
                "sim_map.partition must equal the state x input partition restricted to features"
            )
        is_model = hasattr(sim_map, "successor_bounds")
        if is_model and features is not None and list(features) != list(range(n + nu)):
            raise ValueError("models with successor_bounds must be defined on all of (x, u)")
        if assumption is not None:
            a_lo = np.broadcast_to(np.asarray(assumption[0], float), (n,))
            a_hi = np.broadcast_to(np.asarray(assumption[1], float), (n,))
            # index range of the cells inside the assumption box, per coordinate
            c_lo = np.array([np.searchsorted(e, a, "left") if np.isfinite(a) else 0
                             for e, a in zip(self.X.edges, a_lo)])
            c_hi = np.array([np.searchsorted(e, b, "right") - 2 if np.isfinite(b) else e.size - 2
                             for e, b in zip(self.X.edges, a_hi)])
            if np.any(c_lo > c_hi):
                raise ValueError("the assumption box contains no complete cell")
        if sim_map.m != n:
            raise ValueError(f"sim_map has {sim_map.m} outputs, state has dimension {n}")

        self._g_bounds = g_bounds
        self._is_model = is_model
        self._assumption = None
        if assumption is not None:
            self._assumption = (a_lo, a_hi, c_lo, c_hi)
        nq, npi = self.X.n_cells, self.U.n_cells
        # smallest integer type that holds a cell index (memory: 1 byte per index up to 256)
        kmax = max(self.X.shape)
        self.index_dtype = np.uint8 if kmax <= 256 else (np.uint16 if kmax <= 65536 else np.int32)
        self._q_all = self.X.all_indices()
        self._p_all = self.U.all_indices()
        self._xq_lo, self._xq_hi = self.X.cell_bounds(self._q_all)
        self._up_lo, self._up_hi = self.U.cell_bounds(self._p_all)
        self.lazy = bool(lazy)
        self.n_evaluated = 0  # state-input pairs evaluated by pair_boxes
        if self.lazy:
            # transitions are computed on demand (pair_boxes); no |Q| x |P| arrays
            self.first = self.last = self.enabled = None
            return

        self.first = np.zeros((nq, npi, n), dtype=self.index_dtype)
        self.last = np.zeros((nq, npi, n), dtype=self.index_dtype)
        self.enabled = np.zeros((nq, npi), dtype=bool)
        rows_per_q = max(1, chunk // npi)
        for s in range(0, nq, rows_per_q):
            qs = np.arange(s, min(nq, s + rows_per_q))
            Q = np.repeat(qs, npi)
            P = np.tile(np.arange(npi), qs.size)
            ok, first, last = self.pair_boxes(Q, P)
            self.enabled[qs] = ok.reshape(qs.size, npi)
            self.first[qs] = first.reshape(qs.size, npi, n)
            self.last[qs] = last.reshape(qs.size, npi, n)

    def pair_boxes(self, Q, P):
        """Transitions of the state-input pairs (Q[k], P[k]) (flat cell indices).

        Returns (enabled, first, last): enabled (k,) and the successor box of cells
        first..last (k, n) (zeros where not enabled). This is what the eager
        abstraction stores for every pair; a lazy abstraction calls it on demand.
        """
        Q = np.asarray(Q, dtype=np.intp).ravel()
        P = np.asarray(P, dtype=np.intp).ravel()
        n = self.X.ndim
        self.n_evaluated += Q.size
        joint_idx = np.hstack([self._q_all[Q], self._p_all[P]])
        x_lo, x_hi, u_lo, u_hi = self._xq_lo[Q], self._xq_hi[Q], self._up_lo[P], self._up_hi[P]
        if self._is_model:
            lo, hi = self.sim_map.successor_bounds(joint_idx, x_lo, x_hi, u_lo, u_hi)
        else:
            lo, hi = self.sim_map.bounds_at(joint_idx[:, self.features])
            if self._g_bounds is not None:
                g_lo, g_hi = self._g_bounds(x_lo, x_hi, u_lo, u_hi)
                lo, hi = lo + g_lo, hi + g_hi
        if self._assumption is not None:
            a_lo, a_hi, c_lo, c_hi = self._assumption
            with np.errstate(invalid="ignore"):
                lo = np.clip(lo, a_lo, a_hi)
                hi = np.clip(hi, a_lo, a_hi)
        ok = np.all(lo <= hi, axis=1)
        ok &= np.all(np.isfinite(lo) & np.isfinite(hi), axis=1)
        ok &= np.all((lo >= self.X.lo) & (hi <= self.X.hi), axis=1)
        first = np.zeros((Q.size, n), dtype=self.index_dtype)
        last = np.zeros((Q.size, n), dtype=self.index_dtype)
        if np.any(ok):
            f_, l_ = self.X.box_to_cells(lo[ok], hi[ok])
            if self._assumption is not None:
                f_ = np.clip(f_, c_lo, c_hi)
                l_ = np.clip(l_, c_lo, c_hi)
            first[ok], last[ok] = f_, l_
        return ok, first, last

    @property
    def n_states(self) -> int:
        return self.X.n_cells

    @property
    def n_inputs(self) -> int:
        return self.U.n_cells

    def successors(self, q: int, p: int) -> np.ndarray:
        """Flat indices of the successor cells of (q, p) (empty if p is not enabled)."""
        if self.lazy:
            ok, f, hi = self.pair_boxes([q], [p])
            ok, f, hi = ok[0], f[0], hi[0]
        else:
            ok, f, hi = self.enabled[q, p], self.first[q, p], self.last[q, p]
        if not ok:
            return np.array([], dtype=np.intp)
        ranges = [np.arange(a, b + 1) for a, b in zip(f, hi)]
        grid = np.meshgrid(*ranges, indexing="ij")
        return np.ravel_multi_index(tuple(g.ravel() for g in grid), self.X.shape)

    def state_mask(self, predicate: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """Boolean mask over state cells from a predicate on cell corners (lo, hi)."""
        lo, hi = self.X.cell_bounds(self.X.all_indices())
        return np.asarray(predicate(lo, hi), dtype=bool)

    def cells_inside(self, box_lo, box_hi) -> np.ndarray:
        """Mask of state cells contained in the box [box_lo, box_hi]."""
        box_lo, box_hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
        return self.state_mask(lambda lo, hi: np.all((lo >= box_lo) & (hi <= box_hi), axis=1))

    def cells_meeting(self, box_lo, box_hi) -> np.ndarray:
        """Mask of state cells whose closure intersects the box [box_lo, box_hi].

        Conservative (closed cells), so it is the right choice for obstacles.
        """
        box_lo, box_hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
        return self.state_mask(lambda lo, hi: np.all((hi >= box_lo) & (lo <= box_hi), axis=1))
