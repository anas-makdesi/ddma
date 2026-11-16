"""Assumption monitor: are new data still consistent with the model's assumptions?

Every ddma model rests on assumptions that are given, not learned: a monotonicity sign
pattern, bounds on partial derivatives, bounds on the disturbance. If one of them is
wrong, the model (and every safe set built on it) can be wrong too. This module checks
new transitions against these assumptions online.

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
    s_ij(t) = max(g_hi_ij t, g_lo_ij t)                                         (*)

(the growth cone of point b contains point a; for the sign patterns this is exactly the
consistency condition of the minimal simulating map). A pair that violates (*) is a
proof that at least one assumption is wrong (or that the data are corrupted): the monitor
never raises an alarm on correct assumptions.

**Which assumption?** For a violating pair, the gap of (*) can be closed by changing a
single assumption, all others fixed:

    theta_i            must grow by the gap;
    g_hi_ij (dz_j > 0) must reach  r_j / dz_j,
    g_lo_ij (dz_j < 0) must reach  r_j / dz_j,
    with r_j = y_a,i - y_b,i - theta_i - sum_{k != j} s_ik(dz_k);
    a bound whose dz_j has the other sign (or is 0) cannot close the gap.

Widening one bound never breaks a pair that was consistent before, so the extreme of
these values over all violating pairs seen so far is *exactly* the smallest change of that
one assumption that makes every stored data point consistent (:meth:`AssumptionMonitor.needed`).
Several assumptions can often explain the same data (a wrong sign or a large disturbance);
the report lists each with the change it needs, and those that are ruled out. The size of
the change usually tells them apart (e.g. a slope of exactly -2 vs a disturbance 100 times
the assumed one).

**Responses** (``on_violation``): ``"alarm"`` records the violation and calls
``callback``; ``"widen"`` applies the smallest change of the chosen assumption at once
(the guarantees then rest on bounds fitted to the data, not given in advance, and the
model must be refit from :meth:`AssumptionMonitor.data`); ``"fallback"`` latches
:attr:`AssumptionMonitor.fallback_active` until :meth:`AssumptionMonitor.reset`, for a
switch to a backup controller.

Example::

    sigma = PartitionedSimulatingMap.from_data(Z, Y, part, d_lo, d_hi)
    mon = AssumptionMonitor.for_model(sigma, Z, Y, on_violation="fallback")
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

from ._signs import normalize_signs

__all__ = ["AssumptionMonitor", "Violation", "slopes_from_signs"]


def slopes_from_signs(signs, m: int, n: int):
    """Slope intervals (m, n) of a sign pattern: +1 -> [0, inf), -1 -> (-inf, 0], 0 -> [0, 0]."""
    s = normalize_signs(signs, m, n)
    lo = np.where(s < 0, -np.inf, 0.0)
    hi = np.where(s > 0, np.inf, 0.0)
    return lo, hi


def _cone(dz, lo, hi):
    """s(t) = max(hi t, lo t) per entry, with 0 * inf = 0. dz (K, n), lo/hi (n,) -> (K, n)."""
    with np.errstate(invalid="ignore"):
        return np.where(dz > 0, hi * dz, np.where(dz < 0, lo * dz, 0.0))


@dataclass
class Violation:
    """One transition that contradicts the assumptions.

    index : position of the transition among all checked ones (0-based).
    output : output i whose assumption is contradicted.
    kind : ``"data"`` (inconsistent with a stored data point, the witness) or ``"model"``
        (outside the model's bounds at its cell).
    gap : by how much (in output units) the assumptions are exceeded.
    witness : index of the stored data point it conflicts with (``-1`` for seed data
        without index, or for ``kind == "model"``). Seed points have negative indices
        -1, -2, ... in the order they were added.
    """

    index: int
    output: int
    kind: str
    gap: float
    witness: Optional[int] = None


class AssumptionMonitor:
    """Online check of new transitions against the assumptions of a ddma model.

    Parameters
    ----------
    slope_lo, slope_hi : (m, n) slope intervals (``-inf`` / ``+inf`` allowed), see the
        module docstring. ``(n,)`` arrays apply to every output; give ``m`` then.
    d_lo, d_hi : disturbance bounds, scalars or (m,).
    on_violation : ``"alarm"`` (default), ``"widen"`` or ``"fallback"``.
    widen : what ``"widen"`` changes: ``"disturbance"`` (default) or
        ``("slope_lo", i, j)`` / ``("slope_hi", i, j)``.
    margin : relative extra widening (0.1 = 10 % more than the smallest change).
    max_points : keep at most this many data points (uniform random sample of all data,
        reservoir sampling); ``None`` (default) keeps all. Fewer points: cheaper checks,
        fewer violations detected (each one detected is still a proof).
    model_bounds : optional function Z (N, n) -> (lo, hi) (N, m) with bounds of y at Z from
        the model, and ``inside`` (optional) Z -> bool mask of points where it applies.
        Set by :meth:`for_model`. A transition outside these bounds is also a violation.
    callback : called with the list of :class:`Violation` of each violating transition.
    tol : violations smaller than ``tol`` are ignored (rounding).
    g, features : how :meth:`observe` turns transitions (x, u, x') into data:
        z = (x, u)[features], y = x' - g(x, u).
    input_names, output_names : used in :meth:`report`.
    seed : random seed for ``max_points``.
    """

    def __init__(self, slope_lo, slope_hi, d_lo=0.0, d_hi=0.0, *, m: Optional[int] = None,
                 on_violation: str = "alarm", widen="disturbance", margin: float = 0.0,
                 max_points: Optional[int] = None, model_bounds: Optional[Callable] = None,
                 inside: Optional[Callable] = None, callback: Optional[Callable] = None,
                 tol: float = 1e-9, g: Optional[Callable] = None, features=None,
                 input_names=None, output_names=None, seed: int = 0):
        lo = np.asarray(slope_lo, float)
        hi = np.asarray(slope_hi, float)
        if lo.ndim == 1:
            lo = np.broadcast_to(lo, (m or 1, lo.size))
        if hi.ndim == 1:
            hi = np.broadcast_to(hi, (m or 1, hi.size))
        if lo.shape != hi.shape or lo.ndim != 2:
            raise ValueError("slope_lo and slope_hi must have the same shape (m, n)")
        if np.any(lo > hi) or np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
            raise ValueError("slope bounds must satisfy slope_lo <= slope_hi")
        self.slope_lo, self.slope_hi = lo.copy(), hi.copy()
        mm, n = lo.shape
        self.d_lo = np.broadcast_to(np.asarray(d_lo, float), (mm,)).copy()
        self.d_hi = np.broadcast_to(np.asarray(d_hi, float), (mm,)).copy()
        if np.any(self.d_lo > self.d_hi):
            raise ValueError("disturbance bounds must satisfy d_lo <= d_hi")
        if on_violation not in ("alarm", "widen", "fallback"):
            raise ValueError("on_violation must be 'alarm', 'widen' or 'fallback'")
        self.on_violation = on_violation
        self.widen_target = widen
        self._check_target(widen)
        self.margin = float(margin)
        self.max_points = max_points
        self.model_bounds, self.inside = model_bounds, inside
        self.callback = callback
        self.tol = float(tol)
        self.g, self.features = g, features
        self.input_names = list(input_names) if input_names is not None else None
        self.output_names = list(output_names) if output_names is not None else None
        self._rng = np.random.default_rng(seed)
        self._Zb = np.empty((64, n))  # storage with spare capacity
        self._Yb = np.empty((64, mm))
        self._ib = np.empty(64, np.int64)
        self._K = 0
        self._n_seen = 0  # data points offered to the buffer (for reservoir sampling)
        self._n_seed = 0
        self.n_checked = 0
        self.violations: list[Violation] = []
        self.violating = []  # indices of violating transitions
        self.changes: list[str] = []  # log of widenings
        self.fallback_active = False
        self.needs_refit = False
        self._fallback_since = None
        self._initial = (self.slope_lo.copy(), self.slope_hi.copy(),
                         self.d_lo.copy(), self.d_hi.copy())
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
        :func:`ddma.transitions_to_data`). The model's cell bounds are also checked
        (inside its partition). For ``BoundedDerivativeModel`` the slope intervals are
        those the method relies on: [min(g_lo, 0), max(g_hi, 0)] for "two_matrix",
        "all_matrix" and "lin_mono", [g_lo, g_hi] for the growth methods; zero outside
        ``features``.
        """
        from .bounded_derivatives import BoundedDerivativeModel
        from .growth import GrowthMap
        from .msm import MinimalSimulatingMap
        from .partitioned import PartitionedSimulatingMap

        kw = {}
        if isinstance(model, MinimalSimulatingMap):
            lo, hi = slopes_from_signs(model.signs, model.m, model.n)
            if Z is None:
                Z, Y = model.Z, model.Y
            mon = cls(lo, hi, model.d_lo, model.d_hi, **kwargs)
        elif isinstance(model, PartitionedSimulatingMap):
            lo, hi = slopes_from_signs(model.signs, model.m, model.n)
            P = model.partition
            kw = dict(model_bounds=model.bounds, inside=P.contains)
            mon = cls(lo, hi, model.d_lo, model.d_hi, **kw, **kwargs)
        elif isinstance(model, GrowthMap):
            P = model.partition

            def mb(Zq):
                lo_, hi_ = model.bounds(Zq)
                return lo_[:, None], hi_[:, None]

            kw = dict(model_bounds=mb, inside=P.contains)
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
            P = model.partition
            gfun = model.g

            def mb(Zq):
                X, U = Zq[:, :n_x], Zq[:, n_x:]
                lo_, hi_ = model.bounds(X, U)
                if gfun is not None:
                    gx = gfun(X, U)
                    lo_, hi_ = lo_ - gx, hi_ - gx
                return lo_, hi_

            kw = dict(model_bounds=mb, inside=P.contains)
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
        """Number of stored data points."""
        return self._K

    @property
    def _Z(self):
        return self._Zb[: self._K]

    @property
    def _Y(self):
        return self._Yb[: self._K]

    @property
    def _ids(self):
        return self._ib[: self._K]

    def _append(self, Z, Y, ids):
        k = Z.shape[0]
        if self._K + k > self._Zb.shape[0]:
            cap = max(2 * self._Zb.shape[0], self._K + k)
            for name in ("_Zb", "_Yb", "_ib"):
                old = getattr(self, name)
                new = np.empty((cap,) + old.shape[1:], old.dtype)
                new[: self._K] = old[: self._K]
                setattr(self, name, new)
        self._Zb[self._K : self._K + k] = Z
        self._Yb[self._K : self._K + k] = Y
        self._ib[self._K : self._K + k] = ids
        self._K += k

    def data(self):
        """Stored data (Z, Y) (all data unless ``max_points`` is set), e.g. for a refit."""
        return self._Z.copy(), self._Y.copy()

    # ------------------------------------------------------------------ data
    def _as_data(self, Z, Y):
        Z = np.atleast_2d(np.asarray(Z, float))
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None] if self.m == 1 else Y[None, :]
        if Z.shape[1] != self.n or Y.shape[1] != self.m or Z.shape[0] != Y.shape[0]:
            raise ValueError(f"expected Z (K, {self.n}) and Y (K, {self.m}), "
                             f"got {Z.shape} and {Y.shape}")
        return Z, Y

    def _store(self, z, y, ident):
        self._n_seen += 1
        if self.max_points is None or self.n_points < self.max_points:
            self._append(z[None], y[None], [ident])
        else:
            k = int(self._rng.integers(0, self._n_seen))
            if k < self.max_points:
                self._Zb[k], self._Yb[k], self._ib[k] = z, y, ident

    def add(self, Z, Y):
        """Store data without checking them (e.g. the data the model was learned from)."""
        Z, Y = self._as_data(Z, Y)
        if self.max_points is None:
            self._append(Z, Y, -(self._n_seed + 1 + np.arange(Z.shape[0])))
            self._n_seen += Z.shape[0]
        else:
            for k in range(Z.shape[0]):
                self._store(Z[k], Y[k], -(self._n_seed + 1 + k))
        self._n_seed += Z.shape[0]
        return self

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

    # ------------------------------------------------------------------ checks
    def _pair_gaps(self, z, y):
        """Gaps of (*) between (z, y) and every stored point, both orders.

        Returns gap_new (K, m) for "new above stored" and gap_old (K, m) for "stored
        above new", and dz = z - Z (K, n)."""
        dz = z[None, :] - self._Z  # (K, n)
        dy = y[None, :] - self._Y  # (K, m)
        th = self.theta
        gap_new = np.empty_like(dy)
        gap_old = np.empty_like(dy)
        for i in range(self.m):
            lo, hi = self.slope_lo[i], self.slope_hi[i]
            gap_new[:, i] = dy[:, i] - th[i] - _cone(dz, lo, hi).sum(1)
            gap_old[:, i] = -dy[:, i] - th[i] - _cone(-dz, lo, hi).sum(1)
        return gap_new, gap_old, dz, dy

    def _accumulate(self, i, dz, dy_up):
        """Update the needed changes of output i with violating pairs: dz = z_a - z_b
        (P, n), dy_up = y_a,i - y_b,i (P,), where a is the point above."""
        lo, hi = self.slope_lo[i], self.slope_hi[i]
        s = _cone(dz, lo, hi)  # (P, n), all finite for violating pairs
        total = s.sum(1)
        r = dy_up - self.theta[i] - total  # gap > 0
        self._need_theta[i] = max(self._need_theta[i], self.theta[i] + r.max())
        rest = dy_up[:, None] - self.theta[i] - (total[:, None] - s)  # (P, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rest / dz
        pos, neg = dz > 0, dz < 0
        # g_hi_j can close the gap only if dz_j > 0 for every violating pair
        ok_hi = np.all(pos, axis=0)
        ok_lo = np.all(neg, axis=0)
        for j in range(self.n):
            cur = self._need_hi[i, j]
            if ok_hi[j] and not np.isnan(cur):
                self._need_hi[i, j] = max(cur, ratio[:, j].max())
            else:
                self._need_hi[i, j] = np.nan  # ruled out for good
            cur = self._need_lo[i, j]
            if ok_lo[j] and not np.isnan(cur):
                self._need_lo[i, j] = min(cur, ratio[:, j].min())
            else:
                self._need_lo[i, j] = np.nan

    def check(self, Z, Y) -> list:
        """Check new transitions one by one, then store them. Returns the violations.

        Each transition is checked against every stored data point (exact consistency
        condition (*)) and, if the monitor has model bounds, against the model's bounds at
        its cell. The response (``on_violation``) is applied after each violating
        transition, so later ones are judged under the widened assumptions.
        """
        Z, Y = self._as_data(Z, Y)
        found = []
        for k in range(Z.shape[0]):
            z, y = Z[k], Y[k]
            idx = self.n_checked
            this = []
            if self.n_points:
                gap_new, gap_old, dz, dy = self._pair_gaps(z, y)
                for i in range(self.m):
                    vn = gap_new[:, i] > self.tol
                    vo = gap_old[:, i] > self.tol
                    if not (vn.any() or vo.any()):
                        continue
                    if vn.any():
                        self._accumulate(i, dz[vn], dy[vn, i])
                    if vo.any():
                        self._accumulate(i, -dz[vo], -dy[vo, i])
                    worst = np.maximum(gap_new[:, i], gap_old[:, i])
                    w = int(np.argmax(worst))
                    this.append(Violation(idx, i, "data", float(worst[w]), int(self._ids[w])))
            if self.model_bounds is not None:
                use = True if self.inside is None else bool(self.inside(z[None])[0])
                if use:
                    lo, hi = self.model_bounds(z[None])
                    gap = np.maximum(np.asarray(lo, float).ravel() - y,
                                     y - np.asarray(hi, float).ravel())
                    done = {v.output for v in this}
                    for i in range(self.m):
                        if gap[i] > self.tol and i not in done:
                            this.append(Violation(idx, i, "model", float(gap[i]), -1))
            self._store(z, y, idx)
            self.n_checked += 1
            if this:
                found.extend(this)
                self.violations.extend(this)
                self.violating.append(idx)
                self._respond(this)
        return found

    def _respond(self, viols):
        if self.callback is not None:
            self.callback(viols)
        if self.on_violation == "fallback" and not self.fallback_active:
            self.fallback_active = True
            self._fallback_since = viols[0].index
        elif self.on_violation == "widen":
            if all(v.kind == "model" for v in viols):
                # no witness among the stored data: widen the disturbance by the model gap
                for v in viols:
                    self._need_theta[v.output] = max(self._need_theta[v.output],
                                                     self.theta[v.output] + v.gap)
            target = self.widen_target
            if target != "disturbance" and np.isnan(self._target_need(target)):
                target = "disturbance"  # the chosen bound cannot explain this violation
            self.widen(target, at=viols[0].index)

    # ------------------------------------------------------------------ diagnosis
    def _reset_needed(self):
        self._need_theta = self.theta.copy()
        self._need_lo = self.slope_lo.copy()
        self._need_hi = self.slope_hi.copy()

    def needed(self):
        """Smallest single changes that make all stored data consistent.

        Returns a dict with ``width`` (m,): the disturbance width d_hi - d_lo needed,
        ``slope_lo``, ``slope_hi`` (m, n): the slope bound needed, each with all other
        assumptions unchanged. Entries equal to the current value need no change; ``nan``
        means that this bound alone cannot explain the violations. Violations found only
        through the model bounds (no witness among the stored data) are not included.
        """
        return {"width": self._need_theta.copy(), "slope_lo": self._need_lo.copy(),
                "slope_hi": self._need_hi.copy()}

    def _check_target(self, target):
        if target == "disturbance":
            return
        if (not isinstance(target, tuple) or len(target) != 3
                or target[0] not in ("slope_lo", "slope_hi")):
            raise ValueError("widen target must be 'disturbance', ('slope_lo', i, j) "
                             "or ('slope_hi', i, j)")

    def _target_need(self, target):
        kind, i, j = target
        return (self._need_lo if kind == "slope_lo" else self._need_hi)[i, j]

    def widen(self, target="disturbance", at: Optional[int] = None) -> bool:
        """Apply the smallest change of one assumption (times 1 + margin) that makes all
        stored data consistent. Returns whether anything changed.

        After widening, the model must be refit from :meth:`data` with the new
        assumptions (:attr:`needs_refit`); its guarantees then hold for assumptions
        fitted to the data seen so far, not given in advance.
        """
        self._check_target(target)
        changed = False
        when = f" at transition {at}" if at is not None else ""
        if target == "disturbance":
            for i in range(self.m):
                extra = self._need_theta[i] - self.theta[i]
                if extra > 0:
                    extra *= 1.0 + self.margin
                    old = self.theta[i]
                    self.d_lo[i] -= extra / 2
                    self.d_hi[i] += extra / 2
                    self.changes.append(
                        f"disturbance width of {self._out(i)}: {old:.4g} -> "
                        f"{self.theta[i]:.4g}{when}")
                    changed = True
        else:
            kind, i, j = target
            need = self._target_need(target)
            if np.isnan(need):
                raise ValueError(f"{kind}[{i}, {j}] alone cannot explain the violations")
            arr = self.slope_lo if kind == "slope_lo" else self.slope_hi
            old = arr[i, j]
            if need != old:
                new = old + (1.0 + self.margin) * (need - old)
                arr[i, j] = new
                side = "lower" if kind == "slope_lo" else "upper"
                self.changes.append(
                    f"{side} slope bound of {self._out(i)} w.r.t. {self._inp(j)}: "
                    f"{old:.4g} -> {new:.4g}{when}")
                changed = True
        if changed:
            self.needs_refit = True
            self.model_bounds = None  # built on the old assumptions
            self.inside = None
            self._reset_needed()
        return changed

    def consistent(self, chunk: int = 2000) -> bool:
        """Whether all stored data are consistent with the current assumptions (O(K^2))."""
        K = self.n_points
        for s in range(0, K, chunk):
            a = slice(s, min(K, s + chunk))
            dz = self._Z[a, None, :] - self._Z[None, :, :]  # (B, K, n)
            dy = self._Y[a, None, :] - self._Y[None, :, :]
            for i in range(self.m):
                lo, hi = self.slope_lo[i], self.slope_hi[i]
                with np.errstate(invalid="ignore"):
                    c = np.where(dz > 0, hi * dz, np.where(dz < 0, lo * dz, 0.0)).sum(-1)
                if np.any(dy[..., i] - self.theta[i] - c > self.tol):
                    return False
        return True

    def reset(self):
        """Clear the violation record and the fallback flag (assumptions stay as they are)."""
        self.violations = []
        self.violating = []
        self.fallback_active = False
        self._fallback_since = None
        self._reset_needed()

    # ------------------------------------------------------------------ report
    def _out(self, i):
        return self.output_names[i] if self.output_names else f"output {i}"

    def _inp(self, j):
        return self.input_names[j] if self.input_names else f"input {j}"

    def summary(self) -> dict:
        """Counts per output: number of violations and worst gap."""
        out = []
        for i in range(self.m):
            vs = [v for v in self.violations if v.output == i]
            out.append({"violations": len(vs),
                        "worst_gap": max((v.gap for v in vs), default=0.0),
                        "model_only": sum(v.kind == "model" for v in vs)})
        return {"checked": self.n_checked, "violating": len(self.violating),
                "first": self.violating[0] if self.violating else None,
                "outputs": out, "fallback_active": self.fallback_active,
                "changes": list(self.changes), "needs_refit": self.needs_refit}

    def report(self) -> str:
        """Plain-words summary of the checks, the violations and what would explain them."""
        s = self.summary()
        lines = [f"Assumption monitor: {s['checked']} transitions checked, "
                 f"{s['violating']} inconsistent with the assumptions"
                 + (f" (first: #{s['first']})." if s["first"] is not None else ".")]
        need = self.needed()
        for i, o in enumerate(s["outputs"]):
            if not o["violations"]:
                continue
            nv = o["violations"]
            lines.append(f"{self._out(i)}: {nv} violation{'s' if nv != 1 else ''}, worst "
                         f"{o['worst_gap']:.4g} beyond the assumptions"
                         + (f" ({o['model_only']} found through the model bounds only)"
                            if o["model_only"] else "") + ".")
            cands, ruled = [], []
            w = need["width"][i]
            if w > self.theta[i] + self.tol:
                cands.append(f"disturbance width {self.theta[i]:.4g} -> {w:.4g}")
            for j in range(self.n):
                for side, arr, cur in (("lower", need["slope_lo"], self.slope_lo),
                                       ("upper", need["slope_hi"], self.slope_hi)):
                    v, c = arr[i, j], cur[i, j]
                    name = f"{side} slope bound w.r.t. {self._inp(j)}"
                    if np.isnan(v):
                        ruled.append(name)
                    elif v != c:
                        note = ""
                        if side == "lower" and c == 0 and v < 0:
                            note = (" (contradicts 'does not depend on it')"
                                    if self.slope_hi[i, j] == 0 else
                                    " (contradicts 'non-decreasing')")
                        if side == "upper" and c == 0 and v > 0:
                            note = (" (contradicts 'does not depend on it')"
                                    if self.slope_lo[i, j] == 0 else
                                    " (contradicts 'non-increasing')")
                        cands.append(f"{name} {c:.4g} -> {v:.4g}{note}")
            if cands:
                lines.append("  Any one of these changes alone makes all stored data "
                             "consistent:")
                lines += [f"   - {c}" for c in cands]
            elif o["model_only"] == o["violations"]:
                lines.append("  No witness among the stored data (model bounds only): "
                             "at least the disturbance width must grow by the gap.")
            if ruled:
                lines.append("  Cannot explain the data on their own: " + ", ".join(ruled)
                             + ".")
        if self.fallback_active:
            lines.append(f"Fallback active since #{self._fallback_since}.")
        if self.changes:
            lines.append("Assumptions widened (the model must be refit; its guarantees "
                         "now rest on bounds fitted to the data):")
            lines += [f"   - {c}" for c in self.changes]
        return "\n".join(lines)
