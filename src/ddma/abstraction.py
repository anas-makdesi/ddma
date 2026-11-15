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
from .partitioned import PartitionedSimulatingMap, _core

__all__ = ["AffineBounds", "SymbolicAbstraction", "affine_bounds", "transitions_to_data"]

BoundsFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple]


class AffineBounds:
    """Interval image of g(x, u) = A x + B u + c over boxes (see :func:`affine_bounds`).

    Calling it with (x_lo, x_hi, u_lo, u_hi) returns (g_lo, g_hi). :meth:`parts` returns the
    state and input parts separately, so that the compiled abstraction kernel can combine
    them per cell pair with the same arithmetic.
    """

    def __init__(self, A, B=None, c=None):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = None if B is None else np.atleast_2d(np.asarray(B, dtype=float))
        self.c = np.zeros(self.A.shape[0]) if c is None else np.asarray(c, dtype=float)

    def parts(self, x_lo, x_hi, u_lo, u_hi):
        """(x_mid, x_rad, u_mid, u_rad): mid = x_mid + u_mid, rad = x_rad + u_rad
        (u_mid = u_rad = None when B is None: mid = x_mid, rad = x_rad)."""
        x_mid = 0.5 * (x_lo + x_hi) @ self.A.T + self.c
        x_rad = 0.5 * (x_hi - x_lo) @ np.abs(self.A).T
        if self.B is None:
            return x_mid, x_rad, None, None
        u_mid = 0.5 * (u_lo + u_hi) @ self.B.T
        u_rad = 0.5 * (u_hi - u_lo) @ np.abs(self.B).T
        return x_mid, x_rad, u_mid, u_rad

    def __call__(self, x_lo, x_hi, u_lo, u_hi):
        mid, rad, u_mid, u_rad = self.parts(x_lo, x_hi, u_lo, u_hi)
        if u_mid is not None:
            mid = mid + u_mid
            rad = rad + u_rad
        return mid - rad, mid + rad


def affine_bounds(A, B=None, c=None) -> BoundsFn:
    """Exact interval image of g(x, u) = A x + B u + c over boxes.

    Returns a function (x_lo, x_hi, u_lo, u_hi) -> (g_lo, g_hi) usable as ``g_bounds``
    (an :class:`AffineBounds`, which the compiled abstraction kernel evaluates per cell
    pair without building the bounds of every pair in NumPy).
    """
    return AffineBounds(A, B, c)


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
    backend : "auto" (compiled kernel if installed, else NumPy), "cpp" or "numpy". The
        compiled kernel computes the transitions on all cores (see
        :func:`ddma.set_num_threads`); both backends give identical abstractions.
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
        backend: str = "auto",
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
        if backend not in ("auto", "cpp", "numpy"):
            raise ValueError("backend must be 'auto', 'cpp' or 'numpy'")
        if backend == "cpp" and _core is None:
            raise RuntimeError("compiled kernel ddma._core is not installed")
        self.backend = backend
        self._chunk = chunk
        self._spec = self._kernel_spec() if (backend != "numpy" and _core is not None) else None
        if self.lazy:
            # transitions are computed on demand (pair_boxes); no |Q| x |P| arrays
            self.first = self.last = self.enabled = None
            return

        self.first = np.zeros((nq, npi, n), dtype=self.index_dtype)
        self.last = np.zeros((nq, npi, n), dtype=self.index_dtype)
        self.enabled = np.zeros((nq, npi), dtype=bool)
        self._fill_rows(np.arange(nq))

    def pair_boxes(self, Q, P):
        """Transitions of the state-input pairs (Q[k], P[k]) (flat cell indices).

        Returns (enabled, first, last): enabled (k,) and the successor box of cells
        first..last (k, n) (zeros where not enabled). This is what the eager
        abstraction stores for every pair; a lazy abstraction calls it on demand.
        """
        Q = np.asarray(Q, dtype=np.intp).ravel()
        P = np.asarray(P, dtype=np.intp).ravel()
        self.n_evaluated += Q.size
        if self._spec is not None:
            spec = self._spec_for(Q, P)
            return _core.abstraction_pairs(spec, Q.astype(np.int64), P.astype(np.int64),
                                           np.dtype(self.index_dtype))
        return self._pair_boxes_numpy(Q, P)

    def _successor_boxes(self, Q, P):
        """Successor boxes [lo, hi] (k, n) of the pairs, before clipping (NumPy)."""
        joint_idx = np.hstack([self._q_all[Q], self._p_all[P]])
        x_lo, x_hi, u_lo, u_hi = self._xq_lo[Q], self._xq_hi[Q], self._up_lo[P], self._up_hi[P]
        if self._is_model:
            return self.sim_map.successor_bounds(joint_idx, x_lo, x_hi, u_lo, u_hi)
        lo, hi = self.sim_map.bounds_at(joint_idx[:, self.features])
        if self._g_bounds is not None:
            g_lo, g_hi = self._g_bounds(x_lo, x_hi, u_lo, u_hi)
            lo, hi = lo + g_lo, hi + g_hi
        return lo, hi

    def _pair_boxes_numpy(self, Q, P):
        n = self.X.ndim
        lo, hi = self._successor_boxes(Q, P)
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

    # ------------------------------------------------------------------ compiled kernel
    def _kernel_spec(self) -> dict:
        """What the compiled kernel needs, computed once (see cpp/ddma_abstraction.hpp).

        mode 0: simulating map (+ g); mode 1: two-matrix model; mode 2: boxes computed
        here per pair (other models and maps), the kernel does the clipping and cell
        ranges. g in parts (AffineBounds) is evaluated by the kernel; any other g is
        evaluated here per pair.
        """
        n = self.X.ndim
        spec = dict(mode=2, n=n, nq=self.X.n_cells, np=self.U.n_cells,
                    edges=[np.ascontiguousarray(e, float) for e in self.X.edges],
                    gathers=[], gm=None, gp=None, g=None, g_lo=None, g_hi=None,
                    box_lo=None, box_hi=None, assumption=None)
        if self._assumption is not None:
            a_lo, a_hi, c_lo, c_hi = self._assumption
            spec["assumption"] = (list(map(float, a_lo)), list(map(float, a_hi)),
                                  list(map(int, c_lo)), list(map(int, c_hi)))
        self._g_user = None  # g evaluated here per pair (not in parts)
        sm = self.sim_map
        if not self._is_model and _is_grid_map(sm, n):
            spec["mode"] = 0
            spec["gathers"] = [self._gather(self.features, sm, range(n), 0)]
            g = self._g_bounds
        elif (self._is_model and getattr(sm, "components", None) is None
              and isinstance(getattr(sm, "_gm", None), AffineBounds)
              and isinstance(getattr(sm, "_gp", None), AffineBounds)
              and all(_is_grid_map(m, len(outs)) for _, outs, a, b in sm.groups for m in (a, b))):
            spec["mode"] = 1
            for feats, outs, s_minus, s_plus in sm.groups:
                spec["gathers"].append(self._gather(feats, s_minus, outs, 1))
                spec["gathers"].append(self._gather(feats, s_plus, outs, 2))
            spec["gm"] = self._affine_parts(sm._gm)
            spec["gp"] = self._affine_parts(sm._gp)
            g = sm.g_bounds
        else:
            return spec
        if isinstance(g, AffineBounds):
            spec["g"] = self._affine_parts(g)
        elif g is not None:
            self._g_user = g
        return spec

    def _affine_parts(self, a: AffineBounds):
        x_mid, x_rad, u_mid, u_rad = a.parts(self._xq_lo, self._xq_hi, self._up_lo, self._up_hi)
        def c(a):
            return None if a is None else np.ascontiguousarray(a, float)

        return c(x_mid), c(x_rad), c(u_mid), c(u_rad)

    def _gather(self, feats, sm, outs, target):
        """Offsets of the state and input cells into the flat grid of ``sm`` (on the joint
        coordinates ``feats``): flat index = offq[q] + offp[p]."""
        n = self.X.ndim
        shape = sm.lower.shape[:-1]
        strides = np.ones(len(shape), dtype=np.int64)
        for k in range(len(shape) - 2, -1, -1):
            strides[k] = strides[k + 1] * shape[k + 1]
        offq = np.zeros(self.X.n_cells, dtype=np.int64)
        offp = np.zeros(self.U.n_cells, dtype=np.int64)
        for k, j in enumerate(feats):
            if j < n:
                offq += self._q_all[:, j].astype(np.int64) * strides[k]
            else:
                offp += self._p_all[:, j - n].astype(np.int64) * strides[k]
        lo = np.ascontiguousarray(sm.lower)
        hi = np.ascontiguousarray(sm.upper)
        return (offq, offp, lo, hi, [int(o) for o in outs], target)

    def _spec_for(self, Q, P) -> dict:
        """The spec with the per-pair inputs of the pairs (Q, P) (if any)."""
        spec = self._spec
        if spec["mode"] == 2:
            lo, hi = self._successor_boxes(Q, P)
            spec = dict(spec, box_lo=np.ascontiguousarray(lo, float),
                        box_hi=np.ascontiguousarray(hi, float))
        elif self._g_user is not None:
            g_lo, g_hi = self._g_user(self._xq_lo[Q], self._xq_hi[Q],
                                      self._up_lo[P], self._up_hi[P])
            spec = dict(spec, g_lo=np.ascontiguousarray(g_lo, float),
                        g_hi=np.ascontiguousarray(g_hi, float))
        return spec

    def _fill_rows(self, qs, chunk=None):
        """Compute the transitions of the state cells ``qs`` (all inputs) in place."""
        n, npi = self.X.ndim, self.U.n_cells
        qs = np.asarray(qs, dtype=np.intp)
        if qs.size == 0:
            return
        chunk = self._chunk if chunk is None else chunk
        rows_per_q = max(1, chunk // npi)
        spec = self._spec
        if spec is not None and spec["mode"] != 2 and self._g_user is None:
            # nothing to evaluate per pair in Python: one call, threads over all states
            self.n_evaluated += qs.size * npi
            _core.abstraction_rows(spec, qs.astype(np.int64), self.enabled, self.first, self.last)
            return
        for s in range(0, qs.size, rows_per_q):
            q = qs[s : s + rows_per_q]
            Q = np.repeat(q, npi)
            P = np.tile(np.arange(npi), q.size)
            if spec is not None:
                self.n_evaluated += Q.size
                _core.abstraction_rows(self._spec_for(Q, P), q.astype(np.int64),
                                       self.enabled, self.first, self.last)
                continue
            ok, first, last = self.pair_boxes(Q, P)
            self.enabled[q] = ok.reshape(q.size, npi)
            self.first[q] = first.reshape(q.size, npi, n)
            self.last[q] = last.reshape(q.size, npi, n)

    def refresh(self, states=None) -> np.ndarray:
        """Recompute the transitions of some state cells after the model was refined.

        ``states``: flat indices or boolean mask of the state cells to recompute. By
        default, the cells changed by the model's last ``update_local`` calls (its
        ``changed_boxes``, which are then cleared): a state cell is recomputed when its
        coordinates lie in a changed box (for all inputs). A lazy abstraction stores no
        transitions, so only the mask is returned. Returns the boolean mask of the
        recomputed states. After a refinement the successor boxes only shrink, so the
        previous winning set stays controlled invariant: pass it to
        :func:`ddma.safety` as ``known_winning`` for an exact, cheaper update.
        """
        n = self.X.ndim
        mask = np.zeros(self.n_states, dtype=bool)
        if states is None:
            boxes = getattr(self.sim_map, "changed_boxes", [])
            for feats, first, last in boxes:
                ok = np.ones(self.n_states, dtype=bool)
                for k, j in enumerate(feats):
                    if j < n:
                        ok &= (self._q_all[:, j] >= first[k]) & (self._q_all[:, j] <= last[k])
                mask |= ok
            if hasattr(self.sim_map, "changed_boxes"):
                self.sim_map.changed_boxes = []
        else:
            st = np.asarray(states)
            if st.dtype == bool:
                mask = st.ravel().copy()
            else:
                mask[st.ravel()] = True
        if self.lazy or not mask.any():
            return mask
        self._fill_rows(np.flatnonzero(mask), chunk=1_000_000)
        return mask

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


def _is_grid_map(m, n_out: int) -> bool:
    """Whether ``m`` stores its bounds as grids lower/upper of shape (*cells, n_out)."""
    lo, hi = getattr(m, "lower", None), getattr(m, "upper", None)
    return (isinstance(lo, np.ndarray) and isinstance(hi, np.ndarray) and lo.shape == hi.shape
            and lo.dtype == hi.dtype and lo.dtype in (np.float32, np.float64)
            and lo.ndim >= 2 and lo.shape[-1] == n_out
            and type(m).bounds_at is PartitionedSimulatingMap.bounds_at)
