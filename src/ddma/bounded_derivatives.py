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

The bounds themselves can be estimated with the scenario approach (Section IV):
:func:`estimate_derivative_bounds`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .abstraction import affine_bounds
from .partition import RectangularPartition
from .partitioned import PartitionedSimulatingMap

__all__ = [
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
    """

    def __init__(self, partition, n_x, groups, Am, Ap, Bm, Bp, g, g_bounds, d_lo, d_hi,
                 backend, compact):
        self.partition = partition
        self.n_x = n_x
        self.groups = groups  # list of (features, outputs, map_minus, map_plus)
        self.Am, self.Ap, self.Bm, self.Bp = Am, Ap, Bm, Bp
        self.g, self.g_bounds = g, g_bounds
        self.d_lo, self.d_hi = d_lo, d_hi
        self.backend, self.compact = backend, compact
        self._gm = affine_bounds(Am, Bm)
        self._gp = affine_bounds(Ap, Bp)

    @property
    def m(self) -> int:
        return self.n_x

    @property
    def sigma_minus(self):
        """Map of f- (only when all outputs share one set of inputs)."""
        if len(self.groups) != 1:
            raise AttributeError("several feature groups: use model.groups")
        return self.groups[0][2]

    @property
    def sigma_plus(self):
        if len(self.groups) != 1:
            raise AttributeError("several feature groups: use model.groups")
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
    ) -> BoundedDerivativeModel:
        """Learn the model from transitions (x_k, u_k, x'_k).

        ``partition`` partitions the joint (x, u) box. ``g`` (vectorized, (X, U) -> R^n)
        and ``g_bounds`` (interval bounds of g over boxes, see :func:`ddma.affine_bounds`)
        describe the known part; leave both None when g = 0. ``features``: see the class
        docstring. ``compact=True`` stores the bounds as float32, rounded outward.
        """
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
        model = cls(partition, n_x, groups, Am, Ap, Bm, Bp, g, g_bounds, d_lo, d_hi,
                    backend, compact)
        model.update(X, U, Xnext)
        return model

    def _transformed(self, X, U, Xn):
        gx = np.zeros_like(Xn) if self.g is None else self.g(X, U)
        y_minus = Xn - (gx + X @ self.Am.T + U @ self.Bm.T)
        y_plus = (gx + X @ self.Ap.T + U @ self.Bp.T) - Xn
        return np.hstack([X, U]), y_minus, y_plus

    def update(self, X, U, Xnext) -> BoundedDerivativeModel:
        """Add a batch of transitions (in place; maps are intersected, TAC Prop. 3)."""
        X = np.atleast_2d(np.asarray(X, float))
        U = np.atleast_2d(np.asarray(U, float))
        Xn = np.atleast_2d(np.asarray(Xnext, float))
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

    def successor_bounds(self, joint_idx, x_lo, x_hi, u_lo, u_hi):
        """Box containing f(x, u) + g(x, u) + W for all (x, u) in the given cells."""
        joint_idx = np.atleast_2d(joint_idx)
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
