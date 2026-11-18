"""Assumption monitor: are the data consistent with the model's assumptions?

Every ddma model rests on assumptions that are given, not learned: a monotonicity sign
pattern, bounds on partial derivatives, bounds on the disturbance. If one of them is
wrong, the model (and every safe set built on it) can be wrong too. This module checks
data against these assumptions: the learning data before learning (:func:`check_data`),
and new transitions while the system runs (:class:`AssumptionMonitor`).

**One form for all models.** Each assumption is written as an interval of slopes per
output i and input j, plus a disturbance width per output:

    sign +1         ->  df_i/dz_j in [0, +inf)
    sign -1         ->  df_i/dz_j in (-inf, 0]
    sign 0          ->  df_i/dz_j in [0, 0]       (f_i does not depend on z_j)
    bounded slopes  ->  df_i/dz_j in [g_lo_ij, g_hi_ij]
    disturbance     ->  theta_i = d_hi_i - d_lo_i

Two data points a, b (with y = f(z) + d, d in [d_lo, d_hi]) are consistent with these
assumptions if and only if, for every output i,

    y_a,i - y_b,i  <=  theta_i + sum_j s_ij(z_a,j - z_b,j),
    s_ij(t) = g_hi_ij t (t > 0),  g_lo_ij t (t < 0),  0 (t = 0)                   (*)

(the growth cone of point b contains point a; for the sign patterns this is exactly the
consistency condition of the minimal simulating map). A pair that violates (*) is a
proof that at least one assumption is wrong (or that the data are corrupted): there are
no false alarms.

**Which assumption?** For a violating pair, the gap of (*) can be closed by changing a
single assumption, all others fixed:

    theta_i            must grow by the gap;
    g_hi_ij (dz_j > 0) must reach  r_j / dz_j,
    g_lo_ij (dz_j < 0) must reach  r_j / dz_j,
    with r_j = y_a,i - y_b,i - theta_i - sum_{k != j} s_ik(dz_k);
    a bound whose dz_j has the other sign (or is 0) cannot close the gap.

Widening one bound never breaks a pair that was consistent before, so the extreme of
these values over all violating pairs is *exactly* the smallest change of that one
assumption that makes all data consistent (:meth:`AssumptionMonitor.needed`).

**Three levels** (each alarm is a proof; :class:`Violation` records the level):

1. ``"cells"``: the new point against the model's bounds at its cell (cheap, catches large
   violations);
2. ``"sample"``: against a fresh random sample of stored points (only with
   ``full="later"``);
3. ``"full"``: against *every* earlier point, in the same step (``full="step"``, the
   default) or later, all at once, with :meth:`AssumptionMonitor.flush` (``full="later"``:
   collect during a run, check back at base).

The pair checks run in the C++ core (OpenMP, bit-identical for any number of threads),
with exact pruning by a grid of cells (``prune=True``): a cell is skipped only when no pair
with it can violate (*), so pruning changes the work, never the result.

**Quarantine.** A point that fails a check is kept aside: it counts for the diagnosis, but
later points are not blamed for disagreeing with it. When a pair conflicts, the earlier
point is trusted (unless it is itself quarantined).

**Responses** (``on_violation``): ``"alarm"`` records; ``"fallback"`` latches
:attr:`AssumptionMonitor.fallback_active` until :meth:`AssumptionMonitor.reset`;
``"widen"`` applies the smallest change of the chosen assumption; ``"widen_then_fallback"``
widens while the needed change stays within a limit, and falls back beyond it. After a
widening the model must be refit from :meth:`AssumptionMonitor.data`.

Example::

    print(ddma.check_data(Z, Y, d_lo=d_lo, d_hi=d_hi, signs=signs).report())
    sigma = PartitionedSimulatingMap.from_data(Z, Y, part, d_lo, d_hi)
    mon = AssumptionMonitor.for_model(sigma, Z, Y, on_violation="widen_then_fallback")
    ...
    mon.check(z_new, y_new)          # every new transition
    if mon.fallback_active:
        u = backup(x)
    print(mon.report())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import _monitor_kernels as _np_kernels
from ._signs import normalize_signs
from .partitioned import _core

__all__ = ["AssumptionMonitor", "Violation", "DataCheck", "check_data", "slopes_from_signs"]

_P = 2147483647  # MINSTD random numbers (same in Python and MATLAB): x <- 48271 x mod P
_A = 48271


def slopes_from_signs(signs, m: int, n: int):
    """Slope intervals (m, n) of a sign pattern: +1 -> [0, inf), -1 -> (-inf, 0], 0 -> [0, 0]."""
    s = normalize_signs(signs, m, n)
    lo = np.where(s < 0, -np.inf, 0.0)
    hi = np.where(s > 0, np.inf, 0.0)
    return lo, hi


def _slopes(slope_lo, slope_hi, m=None):
    lo = np.asarray(slope_lo, float)
    hi = np.asarray(slope_hi, float)
    if lo.ndim == 1:
        lo = np.broadcast_to(lo, (m or 1, lo.size))
    if hi.ndim == 1:
        hi = np.broadcast_to(hi, (m or 1, hi.size))
    if lo.shape != hi.shape or lo.ndim != 2:
        raise ValueError("slope_lo and slope_hi must have the same shape (m, n)")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("slope bounds must satisfy slope_lo <= slope_hi")
    if np.any(lo == np.inf) or np.any(hi == -np.inf):
        raise ValueError("slope_lo must be < +inf and slope_hi > -inf")
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


def _kernels(backend):
    if backend not in ("auto", "cpp", "numpy"):
        raise ValueError("backend must be 'auto', 'cpp' or 'numpy'")
    if backend == "cpp" and _core is None:
        raise RuntimeError("compiled kernel ddma._core is not installed")
    return _np_kernels if (backend == "numpy" or _core is None) else _core


def _merge_need(need, width, nlo, nhi, theta):
    """Fold a kernel's need tuple into the needed values (in place)."""
    any_, w, nh, nho, nl, nlo_ = need
    for i in np.flatnonzero(any_):
        width[i] = max(width[i], theta[i] + w[i])
        for j in range(nhi.shape[1]):
            if nho[i, j] or np.isnan(nhi[i, j]):
                nhi[i, j] = np.nan  # ruled out for good
            else:
                nhi[i, j] = max(nhi[i, j], nh[i, j])
            if nlo_[i, j] or np.isnan(nlo[i, j]):
                nlo[i, j] = np.nan
            else:
                nlo[i, j] = min(nlo[i, j], nl[i, j])


def _explain(i, need, theta, slope_lo, slope_hi, tol, out, inp):
    """Plain-words lines for output i: the single changes that explain all data, and the
    bounds ruled out."""
    cands, ruled = [], []
    w = need["width"][i]
    if w > theta[i] + tol:
        cands.append(f"disturbance width {theta[i]:.4g} -> {w:.4g}")
    for j in range(slope_lo.shape[1]):
        for side, arr, cur in (("lower", need["slope_lo"], slope_lo),
                               ("upper", need["slope_hi"], slope_hi)):
            v, c = arr[i, j], cur[i, j]
            name = f"{side} slope bound w.r.t. {inp(j)}"
            if np.isnan(v):
                ruled.append(name)
            elif v != c:
                note = ""
                if side == "lower" and c == 0 and v < 0:
                    note = (" (contradicts 'does not depend on it')" if slope_hi[i, j] == 0
                            else " (contradicts 'non-decreasing')")
                if side == "upper" and c == 0 and v > 0:
                    note = (" (contradicts 'does not depend on it')" if slope_lo[i, j] == 0
                            else " (contradicts 'non-increasing')")
                cands.append(f"{name} {c:.4g} -> {v:.4g}{note}")
    lines = []
    if cands:
        lines.append("  Any one of these changes alone makes all data consistent:")
        lines += [f"   - {c}" for c in cands]
    if ruled:
        lines.append("  Cannot explain the data on their own: " + ", ".join(ruled) + ".")
    return lines


def _names(names, prefix):
    return (lambda k: names[k]) if names else (lambda k: f"{prefix} {k}")


# ====================================================================== check_data
@dataclass
class DataCheck:
    """Result of :func:`check_data`.

    ok : no pair of data points contradicts the assumptions.
    violating : (m,) number of conflicting pairs per output.
    worst : (m,) largest gap per output (``-inf`` if none); worst_pair : (m, 2) the two
        points of that pair (later, earlier), ``-1`` if none.
    conflicts : (K,) number of points each point conflicts with (any output); points with
        many conflicts are the first suspects for corrupted data.
    needed : smallest single change of each assumption that makes all data consistent
        (same as :meth:`AssumptionMonitor.needed`; ``nan`` = this bound alone cannot).
    pairs_evaluated : pairs actually computed (fewer than K(K-1)/2 with pruning).
    """

    ok: bool
    n_points: int
    pairs_evaluated: int
    violating: np.ndarray
    worst: np.ndarray
    worst_pair: np.ndarray
    conflicts: np.ndarray
    needed: dict
    slope_lo: np.ndarray
    slope_hi: np.ndarray
    theta: np.ndarray
    tol: float = 1e-9
    input_names: Optional[list] = None
    output_names: Optional[list] = None

    def suspects(self, k: Optional[int] = None) -> np.ndarray:
        """Indices of points with conflicts, most conflicts first (``k``: the first k)."""
        idx = np.flatnonzero(self.conflicts)
        idx = idx[np.argsort(-self.conflicts[idx], kind="stable")]
        return idx if k is None else idx[:k]

    def report(self) -> str:
        out = _names(self.output_names, "output")
        inp = _names(self.input_names, "input")
        nbad = int(np.count_nonzero(self.conflicts))
        lines = [f"Data check: {self.n_points} points, "
                 + ("all consistent with the assumptions." if self.ok else
                    f"{nbad} involved in pairs that contradict the assumptions.")]
        for i in range(len(self.violating)):
            if not self.violating[i]:
                continue
            a, b = self.worst_pair[i]
            lines.append(f"{out(i)}: {self.violating[i]} conflicting pairs, worst "
                         f"{self.worst[i]:.4g} beyond the assumptions (points {a} and {b}).")
            lines += _explain(i, self.needed, self.theta, self.slope_lo, self.slope_hi,
                              self.tol, out, inp)
        if not self.ok:
            s = self.suspects(5)
            lines.append("Most conflicts: " + ", ".join(
                f"point {k} ({self.conflicts[k]})" for k in s) + ".")
        return "\n".join(lines)


def check_data(Z, Y, slope_lo=None, slope_hi=None, d_lo=0.0, d_hi=0.0, *, signs=None,
               tol: float = 1e-9, prune: bool = True, backend: str = "auto",
               input_names=None, output_names=None) -> DataCheck:
    """Check every pair of data points against the assumptions, before learning.

    Needs only the data and the assumptions (no model). Give the slope intervals
    ``slope_lo``, ``slope_hi`` (m, n), or a sign pattern ``signs`` (as for the monotone
    models). Y: (K, m), or (K,) for one output. Exact: every conflicting pair is found.
    ``prune`` skips cells of a grid in which no pair can conflict (same result, less work).
    Runs in the C++ core on all cores (``ddma.set_num_threads``); ``backend="numpy"``
    checks all pairs in NumPy (slow, same result).
    """
    Z = np.ascontiguousarray(np.atleast_2d(np.asarray(Z, float)))
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Y = np.ascontiguousarray(Y)
    K, n = Z.shape
    m = Y.shape[1]
    if Y.shape[0] != K:
        raise ValueError("Z and Y must have the same number of rows")
    if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(Y))):
        raise ValueError("data must be finite")
    if signs is not None or slope_lo is None:
        lo, hi = slopes_from_signs(signs, m, n)
    else:
        lo, hi = _slopes(slope_lo, slope_hi, m)
    if lo.shape != (m, n):
        raise ValueError(f"slope bounds must be ({m}, {n})")
    d_lo = np.broadcast_to(np.asarray(d_lo, float), (m,))
    d_hi = np.broadcast_to(np.asarray(d_hi, float), (m,))
    theta = np.ascontiguousarray(d_hi - d_lo)
    if np.any(theta < 0):
        raise ValueError("disturbance bounds must satisfy d_lo <= d_hi")
    kern = _kernels(backend)
    need, count, worst, pair, conflicts, pairs = kern.monitor_all(lo, hi, theta, float(tol),
                                                                  Z, Y, bool(prune))
    width, nlo, nhi = theta.copy(), lo.copy(), hi.copy()
    _merge_need(need, width, nlo, nhi, theta)
    return DataCheck(ok=not np.any(count), n_points=K, pairs_evaluated=int(pairs),
                     violating=np.asarray(count), worst=np.asarray(worst),
                     worst_pair=np.asarray(pair), conflicts=np.asarray(conflicts),
                     needed={"width": width, "slope_lo": nlo, "slope_hi": nhi},
                     slope_lo=lo, slope_hi=hi, theta=theta, tol=tol,
                     input_names=input_names, output_names=output_names)


# ====================================================================== monitor
@dataclass
class Violation:
    """One transition that contradicts the assumptions.

    index : position of the transition among all checked ones (0-based).
    output : output i whose assumption is contradicted.
    level : which check found it: ``"cells"`` (the model's bounds at its cell), ``"sample"``
        (a random sample of stored points) or ``"full"`` (all earlier points).
    gap : by how much (in output units) the assumptions are exceeded.
    witness : the stored point it conflicts with (seed points: -1, -2, ... in the order
        added; checked points: their index), ``None`` for ``"cells"``.
    found_at : index of the last checked transition when it was found (``found_at -
        index`` = delay; with ``full="later"`` the delay is that of the :meth:`flush`).
    """

    index: int
    output: int
    level: str
    gap: float
    witness: Optional[int] = None
    found_at: Optional[int] = None


_LEVELS = ("cells", "sample", "full")
_RESPONSES = ("alarm", "widen", "fallback", "widen_then_fallback")


class AssumptionMonitor:
    """Check new transitions against the assumptions of a ddma model.

    Parameters
    ----------
    slope_lo, slope_hi : (m, n) slope intervals (``-inf`` / ``+inf`` allowed), see the
        module docstring. ``(n,)`` arrays apply to every output; give ``m`` then.
    d_lo, d_hi : disturbance bounds, scalars or (m,).
    on_violation : ``"alarm"`` (default), ``"widen"``, ``"fallback"`` or
        ``"widen_then_fallback"``.
    widen : what widening changes: ``"disturbance"`` (default) or ``("slope_lo", i, j)`` /
        ``("slope_hi", i, j)``. A slope bound can be widened only if both of its ends are
        finite (a sign has no scale) and only with ``slope_limit``.
    widen_limit : ``"widen_then_fallback"``: largest disturbance width allowed, as a
        multiple of the original width (scalar or (m,)); beyond it: fallback.
    slope_limit : absolute limit of the widened slope bound (required for a slope target).
    margin : relative extra widening (0.1 = 10 % more than the smallest change).
    full : ``"step"`` (default): each new point is checked against all earlier points in
        the same step. ``"later"``: during a run only levels 1 and 2; :meth:`flush` then
        checks all pending points at once.
    sample : level 2, points per step (``full="later"`` only; 0 = off).
    prune : skip grid cells where no pair can conflict (same result, less work).
    model_bounds : optional function Z (N, n) -> (lo, hi) (N, m) with bounds of y at Z from
        the model, and ``inside`` (optional) Z -> bool mask of points where it applies.
        Set by :meth:`for_model`. Level 1 checks them.
    callback : called with the list of :class:`Violation` found at once.
    tol : violations smaller than ``tol`` are ignored (rounding).
    g, features : how :meth:`observe` turns transitions (x, u, x') into data:
        z = (x, u)[features], y = x' - g(x, u).
    input_names, output_names : used in :meth:`report`.
    seed : seed of the level-2 samples (same samples in Python and MATLAB).
    backend : ``"auto"`` (C++ if installed), ``"cpp"`` or ``"numpy"`` (same results).
    """

    def __init__(self, slope_lo, slope_hi, d_lo=0.0, d_hi=0.0, *, m: Optional[int] = None,
                 on_violation: str = "alarm", widen="disturbance", widen_limit=2.0,
                 slope_limit: Optional[float] = None, margin: float = 0.0,
                 full: str = "step", sample: int = 2000, prune: bool = True,
                 model_bounds: Optional[Callable] = None, inside: Optional[Callable] = None,
                 callback: Optional[Callable] = None, tol: float = 1e-9,
                 g: Optional[Callable] = None, features=None, input_names=None,
                 output_names=None, seed: int = 0, backend: str = "auto"):
        self.slope_lo, self.slope_hi = _slopes(slope_lo, slope_hi, m)
        self.slope_lo, self.slope_hi = self.slope_lo.copy(), self.slope_hi.copy()
        mm, n = self.slope_lo.shape
        self.d_lo = np.broadcast_to(np.asarray(d_lo, float), (mm,)).copy()
        self.d_hi = np.broadcast_to(np.asarray(d_hi, float), (mm,)).copy()
        if np.any(self.d_lo > self.d_hi):
            raise ValueError("disturbance bounds must satisfy d_lo <= d_hi")
        if on_violation not in _RESPONSES:
            raise ValueError("on_violation must be one of " + ", ".join(_RESPONSES))
        if full not in ("step", "later"):
            raise ValueError("full must be 'step' or 'later'")
        self.on_violation = on_violation
        self.widen_target = widen
        self.slope_limit = slope_limit
        self._check_target(widen)
        self.widen_limit = np.broadcast_to(np.asarray(widen_limit, float), (mm,)).copy()
        self.margin = float(margin)
        self.full = full
        self.sample = int(sample)
        self.prune = bool(prune)
        self.backend = backend
        self._kern = _kernels(backend)
        self.model_bounds, self.inside = model_bounds, inside
        self.callback = callback
        self.tol = float(tol)
        self.g, self.features = g, features
        self.input_names = list(input_names) if input_names is not None else None
        self.output_names = list(output_names) if output_names is not None else None
        # storage (spare capacity): data, ids, status (0 clean, 2 quarantined)
        self._Zb = np.empty((64, n))
        self._Yb = np.empty((64, mm))
        self._ib = np.empty(64, np.int64)
        self._st = np.zeros(64, np.uint8)
        self._K = 0
        self._n_seed = 0
        self._pending_from = 0  # points from here on wait for their full check
        self._grid = None  # cell grid for pruning (C++ backend), see rebuild_index
        self._grid_K = 0
        self.n_checked = 0
        self.violations: list[Violation] = []
        self.violating: list[int] = []
        self.changes: list[str] = []
        self.notes: list[str] = []
        self.fallback_active = False
        self.fallback_reason: Optional[str] = None
        self._fallback_since = None
        self.needs_refit = False
        self._theta0 = self.theta.copy()
        self._x = int(seed) % (_P - 1) + 1
        self._pw = None
        self._reset_needed()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def monotone(cls, signs, d_lo=0.0, d_hi=0.0, *, m: int, n: int, **kwargs):
        """Monitor of a monotone model with sign pattern ``signs`` ((m, n), (n,) or None)."""
        lo, hi = slopes_from_signs(signs, m, n)
        return cls(lo, hi, d_lo, d_hi, **kwargs)

    @classmethod
    def for_model(cls, model, Z=None, Y=None, **kwargs):
        """Monitor of a ddma model, seeded with the data (Z, Y) it was learned from.

        ``model``: :class:`ddma.MinimalSimulatingMap` (its data are used when Z is None),
        :class:`ddma.PartitionedSimulatingMap`, :class:`ddma.GrowthMap` or
        :class:`ddma.BoundedDerivativeModel` (data Z = [X U], Y = X' - g(X, U), as in
        :func:`ddma.transitions_to_data`). The model's cell bounds are level 1 (inside
        its partition). For ``BoundedDerivativeModel`` the slope intervals are those the
        method relies on: [min(g_lo, 0), max(g_hi, 0)] for "two_matrix", "all_matrix"
        and "lin_mono", [g_lo, g_hi] for the growth methods; zero outside ``features``.
        Monotone models can only widen the disturbance.
        """
        from .bounded_derivatives import BoundedDerivativeModel
        from .growth import GrowthMap
        from .msm import MinimalSimulatingMap
        from .partitioned import PartitionedSimulatingMap

        if isinstance(model, (MinimalSimulatingMap, PartitionedSimulatingMap)):
            if kwargs.get("widen", "disturbance") != "disturbance":
                raise ValueError("a monotone model can only widen the disturbance (widening a "
                                 "sign makes the model non-monotone: use a "
                                 "BoundedDerivativeModel)")
        if isinstance(model, MinimalSimulatingMap):
            lo, hi = slopes_from_signs(model.signs, model.m, model.n)
            if Z is None:
                Z, Y = model.Z, model.Y
            mon = cls(lo, hi, model.d_lo, model.d_hi, **kwargs)
        elif isinstance(model, PartitionedSimulatingMap):
            lo, hi = slopes_from_signs(model.signs, model.m, model.n)
            kw = dict(model_bounds=model.bounds, inside=model.partition.contains)
            mon = cls(lo, hi, model.d_lo, model.d_hi, **kw, **kwargs)
        elif isinstance(model, GrowthMap):
            def mb(Zq):
                lo_, hi_ = model.bounds(Zq)
                return lo_[:, None], hi_[:, None]

            kw = dict(model_bounds=mb, inside=model.partition.contains)
            mon = cls(model.g_lo[None, :], model.g_hi[None, :], model.d_lo, model.d_hi,
                      **kw, **kwargs)
            if Y is not None:
                Y = np.asarray(Y, float).reshape(-1, 1)
        elif isinstance(model, BoundedDerivativeModel):
            n_x = model.n_x
            n = model.partition.ndim
            if model.components is None:
                lo = np.hstack([model.Am, model.Bm])
                hi = np.hstack([model.Ap, model.Bp])
            else:
                lo = np.zeros((n_x, n))
                hi = np.zeros((n_x, n))
                for i, (f, comp) in enumerate(model.components):
                    if model.method in ("growth", "lin_growth"):
                        lo[i, f], hi[i, f] = comp.g_lo, comp.g_hi
                    else:
                        lo[i, f] = np.minimum(comp.g_lo, 0.0)
                        hi[i, f] = np.maximum(comp.g_hi, 0.0)
            gfun = model.g

            def mb(Zq):
                X, U = Zq[:, :n_x], Zq[:, n_x:]
                lo_, hi_ = model.bounds(X, U)
                if gfun is not None:
                    gx = gfun(X, U)
                    lo_, hi_ = lo_ - gx, hi_ - gx
                return lo_, hi_

            kw = dict(model_bounds=mb, inside=model.partition.contains)
            kwargs.setdefault("g", gfun)
            mon = cls(lo, hi, model.d_lo, model.d_hi, **kw, **kwargs)
        else:
            raise TypeError(f"no monitor for {type(model).__name__}")
        if Z is not None:
            mon.add(Z, Y)
        return mon

    def set_model(self, model_bounds: Optional[Callable], inside: Optional[Callable] = None):
        """Use new model bounds (e.g. after a refit); clears :attr:`needs_refit`."""
        self.model_bounds, self.inside = model_bounds, inside
        self.needs_refit = False

    # ------------------------------------------------------------------ properties
    @property
    def m(self) -> int:
        return self.slope_lo.shape[0]

    @property
    def n(self) -> int:
        return self.slope_lo.shape[1]

    @property
    def theta(self) -> np.ndarray:
        return self.d_hi - self.d_lo

    @property
    def ok(self) -> bool:
        """No violation since the start (or the last :meth:`reset`)."""
        return not self.violating

    @property
    def n_points(self) -> int:
        """Number of stored data points (clean and quarantined)."""
        return self._K

    @property
    def pending(self) -> int:
        """Checked points whose full check (level 3) has not run yet (``full="later"``)."""
        return self._K - self._pending_from

    def status(self) -> dict:
        """Counts: stored points, pending full checks, quarantined points, grid cells."""
        return {"points": self._K, "pending": self.pending,
                "quarantined": int(np.count_nonzero(self._st[: self._K] == 2)),
                "full": self.full,
                "cells": int(self._grid[7].size) if self._grid is not None else 0}

    # ------------------------------------------------------------------ storage
    def _append(self, Z, Y, ids, status):
        k = Z.shape[0]
        if self._K + k > self._Zb.shape[0]:
            cap = max(2 * self._Zb.shape[0], self._K + k)
            for name in ("_Zb", "_Yb", "_ib", "_st"):
                old = getattr(self, name)
                new = np.zeros((cap,) + old.shape[1:], old.dtype)
                new[: self._K] = old[: self._K]
                setattr(self, name, new)
            if self._grid is not None:
                nxt = np.full(cap, -1, np.int64)
                nxt[: self._K] = self._grid[9][: self._K]
                self._grid = self._grid[:9] + (nxt,)
        a, b = self._K, self._K + k
        self._Zb[a:b], self._Yb[a:b], self._ib[a:b], self._st[a:b] = Z, Y, ids, status
        self._K = b
        if self._grid is not None:
            for t in range(a, b):
                self._grid_insert(t)

    def data(self, quarantined: bool = False):
        """Stored data (Z, Y), e.g. for a refit: clean points only, or all with
        ``quarantined=True``."""
        keep = slice(None) if quarantined else (self._st[: self._K] != 2)
        return self._Zb[: self._K][keep].copy(), self._Yb[: self._K][keep].copy()

    def quarantine(self) -> dict:
        """The quarantined points: ``ids`` (seed: negative; checked: transition index),
        ``Z``, ``Y`` and the ``violations`` that put them there."""
        q = np.flatnonzero(self._st[: self._K] == 2)
        ids = self._ib[q].copy()
        s = set(ids.tolist())
        return {"ids": ids, "Z": self._Zb[q].copy(), "Y": self._Yb[q].copy(),
                "violations": [v for v in self.violations if v.index in s]}

    def _as_data(self, Z, Y):
        Z = np.atleast_2d(np.asarray(Z, float))
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None] if self.m == 1 else Y[None, :]
        if Z.shape[1] != self.n or Y.shape[1] != self.m or Z.shape[0] != Y.shape[0]:
            raise ValueError(f"expected Z (K, {self.n}) and Y (K, {self.m}), "
                             f"got {Z.shape} and {Y.shape}")
        if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(Y))):
            raise ValueError("data must be finite")
        return Z, Y

    def add(self, Z, Y):
        """Store data without checking them (e.g. the data the model was learned from;
        check those with :func:`check_data`). Pending full checks are run first."""
        Z, Y = self._as_data(Z, Y)
        if self.pending:
            self.flush()
        k = Z.shape[0]
        self._append(Z, Y, -(self._n_seed + 1 + np.arange(k)), 0)
        self._n_seed += k
        self._pending_from = self._K
        self.rebuild_index()
        return self

    # ------------------------------------------------------------------ grid
    def rebuild_index(self):
        """Rebuild the cell grid used for pruning from all stored points (done by
        :meth:`add`, and when the data have doubled since the last build)."""
        if not (self.prune and self._kern is _core) or self._K == 0:
            self._grid = None
            return
        cap = self._Zb.shape[0]
        g = _core.monitor_grid(self._Zb[: self._K], self._Yb[: self._K])
        nxt = np.full(cap, -1, np.int64)
        nxt[: self._K] = g[9]
        self._grid = tuple(g[:9]) + (nxt,)
        self._grid_K = self._K

    def _grid_insert(self, k):
        cdim, glo, ginv, zlo, zhi, ylo, yhi, cnt, head, nxt = self._grid
        z, y = self._Zb[k], self._Yb[k]
        c = 0
        for j in range(self.n):
            t = 0
            if cdim[j] > 1:
                u = np.floor((z[j] - glo[j]) * ginv[j])
                t = int(min(max(u, 0), cdim[j] - 1))
            c = c * int(cdim[j]) + t
        np.minimum(zlo[c], z, out=zlo[c])
        np.maximum(zhi[c], z, out=zhi[c])
        np.minimum(ylo[c], y, out=ylo[c])
        np.maximum(yhi[c], y, out=yhi[c])
        cnt[c] += 1
        nxt[k] = head[c]
        head[c] = k

    # ------------------------------------------------------------------ checks
    def observe(self, X, U, Xnext):
        """:meth:`check` for transitions (x, u, x'): z = (x, u)[features], y = x' - g(x, u)."""
        X = np.atleast_2d(np.asarray(X, float))
        U = np.atleast_2d(np.asarray(U, float))
        Xn = np.atleast_2d(np.asarray(Xnext, float))
        Z = np.hstack([X, U])
        Y = Xn if self.g is None else Xn - self.g(X, U)
        if self.features is not None:
            Z = Z[:, list(self.features)]
        return self.check(Z, Y)

    def _sample(self, V):
        """S indices in [0, V) from the MINSTD sequence (same in Python and MATLAB)."""
        S = self.sample
        if self._pw is None or self._pw.size != S:
            pw = np.empty(S, np.int64)
            x = 1
            for t in range(S):
                x = (x * _A) % _P
                pw[t] = x
            self._pw = pw
        u = (self._pw * self._x) % _P  # products < 2**62: exact in int64
        self._x = int(u[-1])
        return u % V

    def check(self, Z, Y) -> list:
        """Check new transitions one by one, store them, and respond. Returns the
        violations found during this call (with ``full="later"``: levels 1 and 2 only;
        :meth:`flush` runs level 3)."""
        Z, Y = self._as_data(Z, Y)
        found = []
        for k in range(Z.shape[0]):
            found += self._check_one(Z[k], Y[k])
        return found

    def _check_one(self, z, y):
        idx = self.n_checked
        this = []
        theta = self.theta
        # level 1: the model's bounds at the point's cell
        if self.model_bounds is not None and (self.inside is None
                                              or bool(self.inside(z[None])[0])):
            lo, hi = self.model_bounds(z[None])
            gap = np.maximum(np.asarray(lo, float).ravel() - y, y - np.asarray(hi, float).ravel())
            for i in np.flatnonzero(gap > self.tol):
                this.append(Violation(idx, int(i), "cells", float(gap[i]), None, idx))
                self._need_width_model[i] = max(self._need_width_model[i], theta[i] + gap[i])
        # level 2: a fresh random sample of the points whose full check is done
        V = self._pending_from
        if self.full == "later" and self.sample > 0 and V > 0:
            sidx = self._sample(V)
            need, worst, wit = self._kern.monitor_list(
                self.slope_lo, self.slope_hi, theta, self.tol, self._Zb[:V], self._Yb[:V],
                z, y, sidx, self._st[:V])
            _merge_need(need, self._need_width, self._need_lo, self._need_hi, theta)
            done = {v.output for v in this}
            for i in np.flatnonzero(worst > self.tol):
                if int(i) not in done:
                    this.append(Violation(idx, int(i), "sample", float(worst[i]),
                                          int(self._ib[wit[i]]), idx))
        # store (quarantined if levels 1-2 caught it)
        self._append(z[None], y[None], [idx], 2 if this else 0)
        self.n_checked += 1
        if self._grid is not None and self._K > 2 * self._grid_K:
            self.rebuild_index()
        elif self._grid is None and self.prune and self._kern is _core:
            self.rebuild_index()
        # level 3 in the same step
        if self.full == "step":
            this += self._full_check()
        if this:
            self._record(this)
            self._respond(this)
        return this

    def _full_check(self):
        """Level 3 for the pending points [pending_from, K), each against all earlier
        points; blame in arrival order. Returns the new violations (recorded by caller)."""
        first, last = self._pending_from, self._K
        if first >= last:
            return []
        theta = self.theta
        need, worst, wit, inr, gaps, _pairs = self._kern.monitor_full(
            self.slope_lo, self.slope_hi, theta, self.tol, self._Zb[:last], self._Yb[:last],
            self._st[:last], first, last, self._grid, self.prune)
        _merge_need(need, self._need_width, self._need_lo, self._need_hi, theta)
        found_at = self.n_checked - 1
        out = []
        r = 0
        R = inr.shape[0]
        for q in range(first, last):
            row_g = worst[q - first].copy()
            row_k = wit[q - first].copy()
            while r < R and inr[r, 0] == q:  # partners checked in this same batch
                k, i, gp = int(inr[r, 1]), int(inr[r, 2]), gaps[r]
                if self._st[k] == 0 and (gp > row_g[i] or (gp == row_g[i] and k < row_k[i])):
                    row_g[i], row_k[i] = gp, k
                r += 1
            if self._st[q] == 2:
                continue  # already quarantined (levels 1-2): diagnosis only
            hit = np.flatnonzero(row_g > self.tol)
            if hit.size:
                self._st[q] = 2
                for i in hit:
                    out.append(Violation(int(self._ib[q]), int(i), "full", float(row_g[i]),
                                         int(self._ib[row_k[i]]), found_at))
        self._pending_from = last
        return out

    def flush(self) -> list:
        """Run the full check (level 3) of all pending points now, record and respond.
        Returns the violations found."""
        v = self._full_check()
        if v:
            self._record(v)
            self._respond(v)
        return v

    def _record(self, viols):
        self.violations.extend(viols)
        for idx in sorted({v.index for v in viols}):
            if idx not in self.violating:
                self.violating.append(idx)
        if self.callback is not None:
            self.callback(viols)

    # ------------------------------------------------------------------ responses
    def _respond(self, viols):
        mode = self.on_violation
        when = viols[0].found_at
        if mode == "alarm":
            return
        if mode == "fallback":
            self._fall(when, "assumption violated")
            return
        if self.fallback_active:  # widen_then_fallback after falling back: nothing more
            return
        v = self._full_check()  # the needed changes are exact only after a full check
        if v:
            self._record(v)
        target = self.widen_target
        if target != "disturbance" and np.isnan(self._target_need(target)):
            if mode == "widen_then_fallback":
                self._fall(when, f"{self._target_name(target)} alone cannot explain the data")
                return
            target = "disturbance"  # as in round 1: the chosen bound cannot explain this
        if mode == "widen_then_fallback":
            reason = self._beyond_limit(target)
            if reason:
                self._fall(when, reason)
                return
        self.widen(target, at=when)

    def _fall(self, when, reason):
        if not self.fallback_active:
            self.fallback_active = True
            self._fallback_since = when
            self.fallback_reason = reason

    def _beyond_limit(self, target):
        """Reason why widening ``target`` exceeds its limit, or None."""
        if target == "disturbance":
            need = self._disturbance_need()
            lim = self.widen_limit * self._theta0
            for i in range(self.m):
                if need[i] > self.theta[i] + self.tol and need[i] > lim[i] + self.tol:
                    return (f"disturbance width of {self._out(i)} would need {need[i]:.4g}, "
                            f"beyond the limit {lim[i]:.4g} ({self.widen_limit[i]:g} x the "
                            f"original {self._theta0[i]:.4g})")
            return None
        need = self._target_need(target)
        if (target[0] == "slope_hi" and need > self.slope_limit) or (
                target[0] == "slope_lo" and need < self.slope_limit):
            return (f"{self._target_name(target)} would need {need:.4g}, beyond the limit "
                    f"{self.slope_limit:.4g}")
        return None

    # ------------------------------------------------------------------ diagnosis
    def _reset_needed(self):
        self._need_width = self.theta.copy()
        self._need_lo = self.slope_lo.copy()
        self._need_hi = self.slope_hi.copy()
        self._need_width_model = self.theta.copy()

    def needed(self):
        """Smallest single changes that make all stored data consistent.

        Returns a dict with ``width`` (m,): the disturbance width d_hi - d_lo needed,
        ``slope_lo``, ``slope_hi`` (m, n): the slope bound needed, each with all other
        assumptions unchanged. Entries equal to the current value need no change; ``nan``
        means that this bound alone cannot explain the violations. Exact once no full
        check is pending (:attr:`pending` = 0, see :meth:`flush`); before that, the values
        cover the pairs checked so far. Violations found only by level 1 (the model's
        bounds) are not included; widening also covers them.
        """
        return {"width": self._need_width.copy(), "slope_lo": self._need_lo.copy(),
                "slope_hi": self._need_hi.copy()}

    def _disturbance_need(self):
        return np.maximum(self._need_width, self._need_width_model)

    def _check_target(self, target):
        if target == "disturbance":
            return
        if (not isinstance(target, tuple) or len(target) != 3
                or target[0] not in ("slope_lo", "slope_hi")):
            raise ValueError("widen target must be 'disturbance', ('slope_lo', i, j) "
                             "or ('slope_hi', i, j)")
        _, i, j = target
        if not (np.isfinite(self.slope_lo[i, j]) and np.isfinite(self.slope_hi[i, j])):
            raise ValueError("only a finite slope interval can be widened (a sign has no "
                             "scale: use a bounded-derivative model)")
        if self.slope_limit is None or not np.isfinite(self.slope_limit):
            raise ValueError("widening a slope bound needs an explicit finite slope_limit")

    def _target_need(self, target):
        kind, i, j = target
        return (self._need_lo if kind == "slope_lo" else self._need_hi)[i, j]

    def _target_name(self, target):
        kind, i, j = target
        side = "lower" if kind == "slope_lo" else "upper"
        return f"{side} slope bound of {self._out(i)} w.r.t. {self._inp(j)}"

    def widen(self, target="disturbance", at: Optional[int] = None) -> bool:
        """Apply the smallest change of one assumption (times 1 + margin) that makes all
        stored data consistent; pending full checks are run first. Quarantined points
        then rejoin the data. Returns whether anything changed.

        After widening, the model must be refit from :meth:`data` with the new
        assumptions (:attr:`needs_refit`); its guarantees then hold for assumptions
        fitted to the data seen so far, not given in advance. A slope bound is never
        widened beyond ``slope_limit``.
        """
        self._check_target(target)
        v = self._full_check()
        if v:
            self._record(v)
        changed = False
        when = f" at transition {at}" if at is not None else ""
        if target == "disturbance":
            need = self._disturbance_need()
            for i in range(self.m):
                extra = need[i] - self.theta[i]
                if extra > 0:
                    extra *= 1.0 + self.margin
                    old = self.theta[i]
                    self.d_lo[i] -= extra / 2
                    self.d_hi[i] += extra / 2
                    self.changes.append(f"disturbance width of {self._out(i)}: {old:.4g} -> "
                                        f"{self.theta[i]:.4g}{when}")
                    changed = True
        else:
            kind, i, j = target
            need = self._target_need(target)
            if np.isnan(need):
                raise ValueError(f"{self._target_name(target)} alone cannot explain the data")
            arr = self.slope_lo if kind == "slope_lo" else self.slope_hi
            old = arr[i, j]
            if need != old:
                new = old + (1.0 + self.margin) * (need - old)
                lim = self.slope_limit
                beyond = need > lim if kind == "slope_hi" else need < lim
                if beyond:
                    self.notes.append(f"{self._target_name(target)} not widened: it would "
                                      f"need {need:.4g}, beyond the limit {lim:.4g}{when}")
                else:
                    new = min(new, lim) if kind == "slope_hi" else max(new, lim)
                    arr[i, j] = new
                    self.changes.append(f"{self._target_name(target)}: {old:.4g} -> "
                                        f"{new:.4g}{when}")
                    changed = True
        if changed:
            self.needs_refit = True
            self.model_bounds = None  # built on the old assumptions
            self.inside = None
            self._st[: self._K] = 0  # all data consistent now: the quarantine is lifted
            self._reset_needed()
        return changed

    def consistent(self) -> bool:
        """Whether all stored data (quarantined too) are consistent with the current
        assumptions (all pairs, exact)."""
        if self._K < 2:
            return True
        need, count, *_ = self._kern.monitor_all(
            self.slope_lo, self.slope_hi, self.theta, self.tol, self._Zb[: self._K],
            self._Yb[: self._K], self.prune)
        return not np.any(count)

    def reset(self):
        """Clear the violation record, the fallback flag and the needed changes
        (assumptions and quarantine stay as they are)."""
        self.violations = []
        self.violating = []
        self.fallback_active = False
        self.fallback_reason = None
        self._fallback_since = None
        self._reset_needed()

    # ------------------------------------------------------------------ report
    def _out(self, i):
        return self.output_names[i] if self.output_names else f"output {i}"

    def _inp(self, j):
        return self.input_names[j] if self.input_names else f"input {j}"

    def summary(self) -> dict:
        """Counts per output (violations, worst gap, per level) and the monitor's state."""
        out = []
        for i in range(self.m):
            vs = [v for v in self.violations if v.output == i]
            out.append({"violations": len(vs),
                        "worst_gap": max((v.gap for v in vs), default=0.0),
                        "levels": {lv: sum(v.level == lv for v in vs) for lv in _LEVELS}})
        st = self.status()
        return {"checked": self.n_checked, "violating": len(self.violating),
                "first": self.violating[0] if self.violating else None,
                "levels": {lv: sum(v.level == lv for v in self.violations) for lv in _LEVELS},
                "outputs": out, "pending": st["pending"], "quarantined": st["quarantined"],
                "fallback_active": self.fallback_active,
                "fallback_reason": self.fallback_reason,
                "changes": list(self.changes), "notes": list(self.notes),
                "needs_refit": self.needs_refit}

    def report(self) -> str:
        """Plain-words summary of the checks, the violations and what would explain them."""
        s = self.summary()
        lines = [f"Assumption monitor: {s['checked']} transitions checked, "
                 f"{s['violating']} inconsistent with the assumptions"
                 + (f" (first: #{s['first']})." if s["first"] is not None else ".")]
        if s["violating"]:
            lv = s["levels"]
            lines.append(f"Found by: cells {lv['cells']}, sample {lv['sample']}, "
                         f"full check {lv['full']}. Quarantined points: {s['quarantined']}.")
        if s["pending"]:
            lines.append(f"{s['pending']} full checks pending (flush() runs them); the "
                         "needed changes below cover the pairs checked so far.")
        need = self.needed()
        for i, o in enumerate(s["outputs"]):
            if not o["violations"]:
                continue
            nv = o["violations"]
            lines.append(f"{self._out(i)}: {nv} violation{'s' if nv != 1 else ''}, worst "
                         f"{o['worst_gap']:.4g} beyond the assumptions.")
            ex = _explain(i, need, self.theta, self.slope_lo, self.slope_hi, self.tol,
                          self._out, self._inp)
            if not ex and o["levels"]["cells"] == nv:
                ex = ["  Found through the model's cell bounds only: at least the disturbance "
                      "width must grow by the gap."]
            lines += ex
        if self.fallback_active:
            lines.append(f"Fallback active since #{self._fallback_since}"
                         + (f": {self.fallback_reason}." if self.fallback_reason else "."))
        if self.changes:
            lines.append("Assumptions widened (the model must be refit; its guarantees "
                         "now rest on bounds fitted to the data):")
            lines += [f"   - {c}" for c in self.changes]
        if self.notes:
            lines += [f"Note: {t}." for t in self.notes]
        return "\n".join(lines)
