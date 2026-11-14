"""Systems with bounded derivatives, and estimation of the bounds from data.

TAC paper, Sections III-C and IV. The unknown part f of

    x+ in f(x, u) + g(x, u) + W

need not be monotone; its partial derivatives are assumed to lie in known intervals

    df_i/dx_j in [A_lo_ij, A_hi_ij],   df_i/du_j in [B_lo_ij, B_hi_ij].

With A- = min(A_lo, 0), A+ = max(A_hi, 0) (same for B), the functions

    f- = f - A- x - B- u   and   f+ = A+ x + B+ u - f

are monotone. Their simulating maps S-, S+ are learned from the transformed data

    y-_k = x'_k - g-(x_k, u_k),   y+_k = g+(x_k, u_k) - x'_k,
    g- = g + A- x + B- u,         g+ = g + A+ x + B+ u,

and give the model (Proposition 8)

    f + g + W  in  (g- + S-)  intersected with  (g+ - S+).

This is approach 1 ("2 matrix") of the thesis, Section 3.4. The model offers the
other four with ``method=`` (thesis Chapter 3; see :mod:`ddma._bd_components`):

    method          thesis      idea
    "two_matrix"    approach 1  the two transforms above, interval final step (default)
    "all_matrix"    approach 2  all 2^n slope choices per output (coordinate flips)
    "lin_mono"      approach 3  as "all_matrix", linear-programming final step
    "growth"        approach 4  growth cones evaluated at the grid vertices
    "lin_growth"    approach 5  as "growth", linear-programming final step

The bounds themselves can be estimated with the scenario approach (Section IV):
:func:`estimate_derivative_bounds`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ._bd_components import GrowthComponent, SignMatrixComponent
from .abstraction import affine_bounds
from .partition import RectangularPartition
from .partitioned import PartitionedSimulatingMap

METHODS = ("two_matrix", "all_matrix", "lin_mono", "growth", "lin_growth")

__all__ = [
    "METHODS",
    "BoundedDerivativeModel",
    "estimate_derivative_bounds",
    "scenario_confidence",
]


def _split(lo, hi):
    lo, hi = np.atleast_2d(np.asarray(lo, float)), np.atleast_2d(np.asarray(hi, float))
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError("derivative bounds must have equal shapes and satisfy lo <= hi")
    return np.minimum(lo, 0.0), np.maximum(hi, 0.0)


class BoundedDerivativeModel:
    """Data-driven model of f + g + W for f with bounded partial derivatives.

    Use :meth:`from_data` (and :meth:`update` to add data in batches). The model is defined
    on a rectangular partition of the joint (x, u) space and can be passed to
    :class:`ddma.SymbolicAbstraction` in place of a simulating map (``g_bounds`` is then not
    used: g is already part of the model).

    ``features[i]`` (optional) lists the (x, u) coordinates that output i depends on. Each
    output is then learned on that sub-grid only, which can save orders of magnitude of
    memory (e.g. Lorenz: x+ depends on (x, y) only). Derivative bounds outside the listed
    coordinates are taken as zero: listing features asserts that f_i does not depend on
    the other coordinates.

    ``method`` selects one of the five approaches of the thesis (module docstring).
    All of them are sound; "lin_*" are the tightest and the slowest (one small linear
    program per cell). "two_matrix" leaves cells unbounded where no data point lies on
    the right side; the other methods bound every cell as soon as there is data. For
    methods other than "two_matrix", :meth:`update` refits exactly (same result as
    fitting all data at once), and for "growth" / "lin_growth" :meth:`update_local`
    offers the cheaper windowed update.
    """

    def __init__(self, partition, n_x, groups, Am, Ap, Bm, Bp, g, g_bounds, d_lo, d_hi,
                 backend, compact, method="two_matrix", components=None):
        self.partition = partition
        self.n_x = n_x
        self.groups = groups  # list of (features, outputs, map_minus, map_plus)
        self.Am, self.Ap, self.Bm, self.Bp = Am, Ap, Bm, Bp
        self.g, self.g_bounds = g, g_bounds
        self.d_lo, self.d_hi = d_lo, d_hi
        self.backend, self.compact = backend, compact
        self.method = method
        self.components = components  # list of (features, component), one per output
        self.changed_boxes = []  # cell boxes changed by update_local, for abstraction.refresh
        self._gm = affine_bounds(Am, Bm)
        self._gp = affine_bounds(Ap, Bp)

    @property
    def m(self) -> int:
        return self.n_x

    @property
    def sigma_minus(self):
        """Map of f- (only when all outputs share one set of inputs)."""
        if self.components is not None or len(self.groups) != 1:
            raise AttributeError("only for method='two_matrix' with one feature group")
        return self.groups[0][2]

    @property
    def sigma_plus(self):
        if self.components is not None or len(self.groups) != 1:
            raise AttributeError("only for method='two_matrix' with one feature group")
        return self.groups[0][3]

    @classmethod
    def from_data(
        cls,
        X,
        U,
        Xnext,
        partition: RectangularPartition,
        A_lo,
        A_hi,
        B_lo,
        B_hi,
        d_lo=0.0,
        d_hi=0.0,
        g: Optional[Callable] = None,
        g_bounds: Optional[Callable] = None,
        backend: str = "auto",
        features=None,
        compact: bool = False,
        method: str = "two_matrix",
        local: Optional[str] = None,
    ) -> BoundedDerivativeModel:
        """Learn the model from transitions (x_k, u_k, x'_k).

        ``partition`` partitions the joint (x, u) box. ``g`` (vectorized, (X, U) -> R^n)
        and ``g_bounds`` (interval bounds of g over boxes, see :func:`ddma.affine_bounds`)
        describe the known part; leave both None when g = 0. ``features``: see the class
        docstring. ``compact=True`` stores the bounds as float32, rounded outward
        ("two_matrix", "all_matrix", "lin_mono"). ``method``: see the class docstring.
        ``local`` ("growth" / "lin_growth" only): also intersect with the cones of the data
        points in each cell (``"cell"``) or in the cell's slabs (``"slab"``, tighter and
        never looser than min over data of the cone's max over the cell); see
        :class:`ddma._bd_components.GrowthComponent`.
        """
        if local is not None and method not in ("growth", "lin_growth"):
            raise ValueError("local is available for method='growth' or 'lin_growth'")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        X = np.atleast_2d(np.asarray(X, float))
        U = np.atleast_2d(np.asarray(U, float))
        n_x, n_u = X.shape[1], U.shape[1]
        if (g is None) != (g_bounds is None):
            raise ValueError("give both g and g_bounds, or neither")
        Am, Ap = _split(A_lo, A_hi)
        Bm, Bp = _split(B_lo, B_hi)
        if features is None:
            features = [list(range(n_x + n_u))] * n_x
        if len(features) != n_x:
            raise ValueError("features needs one list of coordinates per state output")
        for i, f in enumerate(features):  # outside its features, output i is constant
            mask = np.ones(n_x + n_u, bool)
            mask[list(f)] = False
            Am[i, mask[:n_x]] = Ap[i, mask[:n_x]] = 0.0
            Bm[i, mask[n_x:]] = Bp[i, mask[n_x:]] = 0.0
        d_lo = np.broadcast_to(np.asarray(d_lo, float), (n_x,)).copy()
        d_hi = np.broadcast_to(np.asarray(d_hi, float), (n_x,)).copy()
        by_feats = {}
        for i, f in enumerate(features):
            by_feats.setdefault(tuple(int(j) for j in f), []).append(i)
        groups = [(list(f), outs, None, None) for f, outs in by_feats.items()]
        components = None
        if method != "two_matrix":
            g_lo_all = np.hstack([np.atleast_2d(np.asarray(A_lo, float)),
                                  np.atleast_2d(np.asarray(B_lo, float))])
            g_hi_all = np.hstack([np.atleast_2d(np.asarray(A_hi, float)),
                                  np.atleast_2d(np.asarray(B_hi, float))])
            final = "lp" if method.startswith("lin_") else "box"
            components = []
            for i, f in enumerate(features):
                f = [int(j) for j in f]
                P = partition.project(f)
                gl, gh = g_lo_all[i, f], g_hi_all[i, f]
                if method in ("growth", "lin_growth"):
                    comp = GrowthComponent(P, gl, gh, d_lo[i], d_hi[i], final=final,
                                           local=local, backend=backend)
                else:
                    comp = SignMatrixComponent(P, gl, gh, d_lo[i], d_hi[i], final=final,
                                               backend=backend, compact=compact)
                components.append((f, comp))
            groups = []
        model = cls(partition, n_x, groups, Am, Ap, Bm, Bp, g, g_bounds, d_lo, d_hi,
                    backend, compact, method, components)
        model.update(X, U, Xnext)
        return model

    def _transformed(self, X, U, Xn):
        gx = np.zeros_like(Xn) if self.g is None else self.g(X, U)
        y_minus = Xn - (gx + X @ self.Am.T + U @ self.Bm.T)
        y_plus = (gx + X @ self.Ap.T + U @ self.Bp.T) - Xn
        return np.hstack([X, U]), y_minus, y_plus

    def _component_data(self, X, U, Xn):
        gx = np.zeros_like(Xn) if self.g is None else self.g(X, U)
        return np.hstack([X, U]), Xn - gx

    def update(self, X, U, Xnext) -> BoundedDerivativeModel:
        """Add a batch of transitions (in place; maps are intersected, TAC Prop. 3)."""
        X = np.atleast_2d(np.asarray(X, float))
        U = np.atleast_2d(np.asarray(U, float))
        Xn = np.atleast_2d(np.asarray(Xnext, float))
        if self.components is not None:
            Z, Y = self._component_data(X, U, Xn)
            for i, (f, comp) in enumerate(self.components):
                comp.update(Z[:, f], Y[:, i])
            return self
        Z, ym, yp = self._transformed(X, U, Xn)
        for k, (feats, outs, sm, sp) in enumerate(self.groups):
            P = self.partition.project(feats)
            new_m = PartitionedSimulatingMap.from_data(
                Z[:, feats], ym[:, outs], P, self.d_lo[outs], self.d_hi[outs],
                backend=self.backend, compact=self.compact)
            new_p = PartitionedSimulatingMap.from_data(
                Z[:, feats], yp[:, outs], P, -self.d_hi[outs], -self.d_lo[outs],
                backend=self.backend, compact=self.compact)
            sm = new_m if sm is None else sm.intersect_(new_m)
            del new_m
            sp = new_p if sp is None else sp.intersect_(new_p)
            del new_p
            self.groups[k] = (feats, outs, sm, sp)
        return self

    def update_local(self, X, U, Xnext, window) -> BoundedDerivativeModel:
        """Windowed update for "growth" / "lin_growth" (thesis Section 3.2, Prop. 13).

        Each transition tightens the model only on the cells within ``window`` cells of
        its own cell (an int, or one int per coordinate of the output's features). Meant
        for online use, one or a few points at a time. Sound; not as tight as
        :meth:`update`, which spreads each point over the whole grid.
        """
        if self.method not in ("growth", "lin_growth"):
            raise ValueError("update_local is available for method='growth' or 'lin_growth'")
        X = np.atleast_2d(np.asarray(X, float))
        U = np.atleast_2d(np.asarray(U, float))
        Xn = np.atleast_2d(np.asarray(Xnext, float))
        Z, Y = self._component_data(X, U, Xn)
        for i, (f, comp) in enumerate(self.components):
            w = window if np.ndim(window) == 0 else np.asarray(window)[f]
            for first, last in comp.update_local(Z[:, f], Y[:, i], w):
                self.changed_boxes.append((f, first, last))
        return self

    def f_bounds_at(self, joint_idx):
        """Bounds of f(x, u) + W per cell, without g (methods other than "two_matrix")."""
        if self.components is None:
            raise AttributeError("available for methods other than 'two_matrix'")
        joint_idx = np.atleast_2d(np.asarray(joint_idx, np.intp))
        lo = np.empty((joint_idx.shape[0], self.n_x))
        hi = np.empty_like(lo)
        for i, (f, comp) in enumerate(self.components):
            ix = tuple(joint_idx[:, f].T)
            lo[:, i], hi[:, i] = comp.lower[ix], comp.upper[ix]
        return lo, hi

    def successor_bounds(self, joint_idx, x_lo, x_hi, u_lo, u_hi):
        """Box containing f(x, u) + g(x, u) + W for all (x, u) in the given cells."""
        joint_idx = np.atleast_2d(joint_idx)
        if self.components is not None:
            lo, hi = self.f_bounds_at(joint_idx)
            if self.g_bounds is not None:
                g_lo, g_hi = self.g_bounds(x_lo, x_hi, u_lo, u_hi)
                lo, hi = lo + g_lo, hi + g_hi
            return lo, hi
        k = joint_idx.shape[0]
        sm_lo = np.empty((k, self.n_x))
        sm_hi = np.empty((k, self.n_x))
        sp_lo = np.empty((k, self.n_x))
        sp_hi = np.empty((k, self.n_x))
        for feats, outs, sm, sp in self.groups:
            sm_lo[:, outs], sm_hi[:, outs] = sm.bounds_at(joint_idx[:, feats])
            sp_lo[:, outs], sp_hi[:, outs] = sp.bounds_at(joint_idx[:, feats])
        gm_lo, gm_hi = self._gm(x_lo, x_hi, u_lo, u_hi)
        gp_lo, gp_hi = self._gp(x_lo, x_hi, u_lo, u_hi)
        if self.g_bounds is not None:
            g_lo, g_hi = self.g_bounds(x_lo, x_hi, u_lo, u_hi)
            gm_lo, gm_hi, gp_lo, gp_hi = gm_lo + g_lo, gm_hi + g_hi, gp_lo + g_lo, gp_hi + g_hi
        with np.errstate(invalid="ignore"):
            lo = np.maximum(gm_lo + sm_lo, gp_lo - sp_hi)
            hi = np.minimum(gm_hi + sm_hi, gp_hi - sp_lo)
        return lo, hi

    def bounds(self, x, u):
        """Box containing f(x, u) + g(x, u) + W at given points (via their cells)."""
        x, u = np.atleast_2d(x), np.atleast_2d(u)
        idx = self.partition.locate(np.hstack([x, u]))
        lo, hi = self.partition.cell_bounds(idx)
        n = self.n_x
        return self.successor_bounds(idx, lo[:, :n], hi[:, :n], lo[:, n:], hi[:, n:])


def estimate_derivative_bounds(
    Z,
    Y,
    n_lp_pairs: int = 50_000,
    n_w_pairs: int = 1_000_000,
    rng: Optional[np.random.Generator] = None,
    w_weight: float = 1.0,
):
    """Estimate derivative and disturbance bounds from data (TAC paper, Section IV).

    Data (z_k, y_k) with y_k in f(z_k) + [-w, w]. Random pairs of samples are drawn;
    for each output i the linear program (32) on ``n_lp_pairs`` pairs gives the
    derivative bounds a_lo_i, a_hi_i, then (33) on ``n_w_pairs`` fresh pairs gives w_i.

    ``w_weight`` multiplies the disturbance term of the LP cost (1 = equation (32) of the
    paper). Larger values trade slightly wider derivative bounds for a smaller disturbance
    bound; the original MATLAB experiments used 50. Any value keeps the guarantee of
    Proposition 9, which only concerns the second step (33).

    Returns (a_lo, a_hi, w) with shapes (m, n), (m, n), (m,). The disturbance interval is
    [-w, w]. See :func:`scenario_confidence` for the probabilistic guarantee.
    """
    from scipy.optimize import linprog

    rng = np.random.default_rng() if rng is None else rng
    Z = np.atleast_2d(np.asarray(Z, float))
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    K, n = Z.shape
    m = Y.shape[1]

    def pairs(count):
        a = rng.integers(0, K, count)
        b = rng.integers(0, K, count)
        dz = Z[a] - Z[b]
        return np.maximum(dz, 0), np.minimum(dz, 0), Y[a] - Y[b]

    P1, N1, dY1 = pairs(n_lp_pairs)
    P2, N2, dY2 = pairs(n_w_pairs)
    a_lo = np.zeros((m, n))
    a_hi = np.zeros((m, n))
    w = np.zeros(m)
    # variables: [a_lo (n), a_hi (n), w]
    c = np.concatenate([N1.sum(0), P1.sum(0), [2.0 * n_lp_pairs * w_weight]])
    A_ub = np.vstack([
        -np.hstack([N1, P1, 2 * np.ones((n_lp_pairs, 1))]),
        np.hstack([np.eye(n), -np.eye(n), np.zeros((n, 1))]),
    ])
    bounds = [(None, None)] * (2 * n) + [(0, None)]
    for i in range(m):
        b_ub = np.concatenate([-dY1[:, i], np.zeros(n)])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP for output {i} failed: {res.message}")
        a_hi[i] = res.x[n : 2 * n]
        a_lo[i] = np.minimum(res.x[:n], a_hi[i])  # a_lo <= a_hi up to solver tolerance
        slack = dY2[:, i] - P2 @ a_hi[i] - N2 @ a_lo[i]
        w[i] = max(0.5 * slack.max(), 0.0)
    return a_lo, a_hi, w


def scenario_confidence(n_w_pairs: int, beta: float) -> float:
    """Proposition 9: with probability >= 1 - beta, a fresh pair of samples satisfies the
    estimated bounds with probability >= beta ** (1 / n_w_pairs)."""
    return float(beta ** (1.0 / n_w_pairs))
