"""Controller synthesis on symbolic abstractions (safety and reach-avoid).

Successor sets are boxes of cells, so "all successors lie in W" is checked in O(2^n)
per transition with an n-dimensional summed-area table of the complement of W.

The fixed points run in the compiled kernel (cpp/ddma_synthesis.hpp) when it is installed
and in NumPy otherwise; both give the same sets as the textbook iterations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .abstraction import SymbolicAbstraction
from .partitioned import _core

__all__ = ["Controller", "reach_avoid", "safety", "simulate"]


def _summed_area(abs_: SymbolicAbstraction, W: np.ndarray) -> np.ndarray:
    """n-D summed-area table of the complement of W (cells outside W count 1)."""
    shape = abs_.X.shape
    sat = np.zeros(tuple(s + 1 for s in shape), dtype=np.int64)
    sat[tuple(slice(1, None) for _ in shape)] = ~W.reshape(shape)
    for ax in range(len(shape)):
        np.cumsum(sat, axis=ax, out=sat)
    return sat


def _bad_in_boxes(sat, first, last) -> np.ndarray:
    """Number of cells outside W in each box [first, last] (arrays (..., n))."""
    n = first.shape[-1]
    first = first.astype(np.intp)
    last = last.astype(np.intp) + 1
    count = np.zeros(first.shape[:-1], dtype=np.int64)
    for corner in itertools.product((0, 1), repeat=n):
        idx = tuple(np.where(c, last[..., j], first[..., j]) for j, c in enumerate(corner))
        sign = -1 if (n - sum(corner)) % 2 else 1
        count += sign * sat[idx]
    return count


_CHUNK = 2_000_000  # state-input pairs per vectorized block (bounds temporary memory)


def _use_cpp(backend: str) -> bool:
    if backend not in ("auto", "cpp", "numpy"):
        raise ValueError("backend must be 'auto', 'cpp' or 'numpy'")
    if backend == "cpp" and _core is None:
        raise RuntimeError("compiled kernel ddma._core is not installed")
    return backend != "numpy" and _core is not None


def _core_args(abs_: SymbolicAbstraction):
    return (abs_.first, abs_.last, abs_.enabled.view(np.uint8), list(abs_.X.shape))


def _box_all_in(abs_: SymbolicAbstraction, W: np.ndarray, backend: str = "auto") -> np.ndarray:
    """good[q, p]: p enabled at q and every successor cell lies in W (row-chunked)."""
    if _use_cpp(backend):
        W8 = np.ascontiguousarray(W, dtype=bool).view(np.uint8)
        return _core.box_all_in(*_core_args(abs_), W8).view(bool)
    sat = _summed_area(abs_, W)
    good = np.zeros(abs_.enabled.shape, dtype=bool)
    step = max(1, _CHUNK // max(1, abs_.n_inputs))
    for s in range(0, abs_.n_states, step):
        r = slice(s, s + step)
        good[r] = abs_.enabled[r] & (_bad_in_boxes(sat, abs_.first[r], abs_.last[r]) == 0)
    return good


@dataclass
class Controller:
    """Symbolic controller: allowed[q, p] says input cell p may be used in state cell q.

    ``domain`` is the set of state cells where the controller is defined (the winning
    set); ``value`` is the number of steps to the target for reach-avoid controllers.
    """

    abstraction: SymbolicAbstraction
    allowed: Optional[np.ndarray]
    domain: np.ndarray
    value: Optional[np.ndarray] = None

    def allowed_row(self, q: int) -> np.ndarray:
        """allowed[q, :]. For a lazy abstraction (allowed is None) it is computed on
        demand from the transitions of q and the domain."""
        if self.allowed is not None:
            return self.allowed[q]
        if not self.domain[q]:
            return np.zeros(self.abstraction.n_inputs, dtype=bool)
        if getattr(self, "_sat", None) is None:
            self._sat = _summed_area(self.abstraction, self.domain)
        P = np.arange(self.abstraction.n_inputs)
        ok, first, last = self.abstraction.pair_boxes(np.full(P.size, q), P)
        return ok & (_bad_in_boxes(self._sat, first, last) == 0)

    def winning_fraction(self) -> float:
        return float(np.mean(self.domain))

    def domain_grid(self) -> np.ndarray:
        """Domain as a boolean array over the state grid shape."""
        return self.domain.reshape(self.abstraction.X.shape)

    def input_cells(self, x) -> np.ndarray:
        """Allowed input-cell indices at the concrete state x (empty outside the domain)."""
        X = self.abstraction.X
        x = np.atleast_2d(x)
        if not X.contains(x)[0]:
            return np.array([], dtype=np.intp)
        q = np.ravel_multi_index(tuple(X.locate(x)[0]), X.shape)
        return np.flatnonzero(self.allowed_row(q))

    def __call__(self, x, pick: str = "center", rng: Optional[np.random.Generator] = None):
        """A concrete input for state x: a point of an allowed input cell.

        ``pick`` = "center" (centre of the first allowed cell), "random" (uniform in a
        random allowed cell), or "min_norm" (the allowed cell centre with smallest norm).
        Any point of an allowed cell is admissible. Returns None outside the domain.
        """
        ps = self.input_cells(x)
        if ps.size == 0:
            return None
        U = self.abstraction.U
        idx = np.stack(np.unravel_index(ps, U.shape), axis=1)
        lo, hi = U.cell_bounds(idx)
        if pick == "random":
            rng = np.random.default_rng() if rng is None else rng
            k = rng.integers(ps.size)
            return rng.uniform(lo[k], hi[k])
        centers = 0.5 * (lo + hi)
        if pick == "min_norm":
            return centers[np.argmin(np.linalg.norm(centers, axis=1))]
        return centers[0]

    def filter(self, x, u_desired):
        """Safety filter: the admissible input closest to ``u_desired`` (Euclidean).

        Returns ``u_desired`` unchanged when it lies in an allowed input cell, the
        nearest point of the allowed cells otherwise, and None outside the domain.
        """
        ps = self.input_cells(x)
        if ps.size == 0:
            return None
        U = self.abstraction.U
        u = np.asarray(u_desired, dtype=float).ravel()
        lo, hi = U.cell_bounds(np.stack(np.unravel_index(ps, U.shape), axis=1))
        # keep a margin inside each cell so the returned point is not on a boundary
        # that belongs to a neighbouring (possibly forbidden) half-open cell
        eps = 1e-9 * (hi - lo)
        cand = np.clip(u, lo, hi - eps)
        return cand[np.argmin(np.linalg.norm(cand - u, axis=1))]


def safety(
    abs_: SymbolicAbstraction,
    safe: np.ndarray,
    max_iter: int = 100_000,
    backend: str = "auto",
) -> Controller:
    """Maximal controlled invariant subset of ``safe`` (flat boolean mask over state cells).

    Greatest fixed point: remove cells with no input whose successors all stay in the
    current set, until nothing changes. Each state keeps a pointer to its first input not
    yet ruled out; the set only shrinks, so an input that fails once fails for ever and the
    pointer only moves forward. Over the whole fixed point every (state, input) pair is
    examined at most once, plus one check of the current pointer per state and round.

    ``backend``: "auto" (compiled kernel if installed, else NumPy), "cpp" or "numpy".
    Both give the same result as the textbook iteration.

    With a lazy abstraction (``SymbolicAbstraction(..., lazy=True)``) the same scan runs
    in NumPy and evaluates a transition only when the scan reaches it: states outside
    ``safe`` are never expanded and each state stops at its first good input. Memory is
    O(|Q|) instead of O(|Q| |P|); the controller computes its allowed inputs on demand.
    Same result.
    """
    safe = np.asarray(safe, dtype=bool).ravel()
    if safe.size != abs_.n_states:
        raise ValueError("safe must have one entry per state cell")
    if getattr(abs_, "lazy", False):
        return _safety_lazy(abs_, safe, max_iter)
    if _use_cpp(backend):
        W, allowed, _ = _core.safety(*_core_args(abs_), safe.view(np.uint8))
        return Controller(abs_, allowed.view(bool), W.view(bool))
    W = safe.copy()
    nq, npi = abs_.enabled.shape
    ptr = np.zeros(nq, dtype=np.int64)  # first input of each state not yet ruled out
    block = 16  # inputs scanned per state in one vectorized step
    for _ in range(max_iter):
        sat = _summed_area(abs_, W)
        rows = np.flatnonzero(W)
        # 1. check the current pointers
        r = rows[ptr[rows] < npi]
        ok = np.zeros(r.size, dtype=bool)
        for s in range(0, r.size, _CHUNK):
            rr = r[s : s + _CHUNK]
            pp = ptr[rr]
            ok[s : s + _CHUNK] = abs_.enabled[rr, pp] & (
                _bad_in_boxes(sat, abs_.first[rr, pp], abs_.last[rr, pp]) == 0
            )
        todo = r[~ok]
        ptr[todo] += 1
        # 2. move the failed pointers forward, `block` inputs at a time
        while todo.size:
            nxt = []
            for s in range(0, todo.size, max(1, _CHUNK // block)):
                t = todo[s : s + max(1, _CHUNK // block)]
                P = ptr[t][:, None] + np.arange(block)
                valid = P < npi
                Pc = np.minimum(P, npi - 1)
                R = np.broadcast_to(t[:, None], P.shape)
                good = valid & abs_.enabled[R, Pc]
                good &= _bad_in_boxes(sat, abs_.first[R, Pc], abs_.last[R, Pc]) == 0
                found = good.any(axis=1)
                ptr[t[found]] = P[found, good[found].argmax(axis=1)]
                lost = t[~found]
                ptr[lost] = np.minimum(ptr[lost] + block, npi)
                nxt.append(lost[ptr[lost] < npi])
            todo = np.concatenate(nxt)
        dead = rows[ptr[rows] >= npi]
        if dead.size == 0:
            break
        W[dead] = False
    else:
        raise RuntimeError("safety fixed point did not converge")
    allowed = _box_all_in(abs_, W, "numpy") & W[:, None]
    return Controller(abs_, allowed, W)


def _safety_lazy(abs_, safe, max_iter):
    """Pointer scan on a lazy abstraction: transitions evaluated on demand."""
    nq, npi, n = abs_.n_states, abs_.n_inputs, abs_.X.ndim
    W = safe.copy()
    ptr = np.zeros(nq, dtype=np.int64)  # current candidate input of each state
    wf = np.zeros((nq, n), dtype=np.intp)  # its successor box, when known
    wl = np.zeros((nq, n), dtype=np.intp)
    known = np.zeros(nq, dtype=bool)
    for _ in range(max_iter):
        sat = _summed_area(abs_, W)
        rows = np.flatnonzero(W & known)
        bad = _bad_in_boxes(sat, wf[rows], wl[rows]) != 0
        failed = rows[bad]
        ptr[failed] += 1
        known[failed] = False
        todo = np.flatnonzero(W & ~known & (ptr < npi))
        block = 1
        while todo.size:  # scan forward, evaluating 1, 2, 4, ... transitions per state
            P = ptr[todo][:, None] + np.arange(block)
            valid = P < npi
            ok, f, la = abs_.pair_boxes(np.broadcast_to(todo[:, None], P.shape)[valid], P[valid])
            f, la = f.astype(np.intp), la.astype(np.intp)
            gv = ok.copy()
            gv[ok] = _bad_in_boxes(sat, f[ok], la[ok]) == 0
            g = np.zeros(P.shape, dtype=bool)
            g[valid] = gv
            found = g.any(axis=1)
            j = g[found].argmax(axis=1)
            hit = todo[found]
            ptr[hit] += j
            where = np.full(P.size, -1)  # (row, column) of P -> index in the evaluated pairs
            where[np.flatnonzero(valid.ravel())] = np.arange(gv.size)
            sel = where[np.flatnonzero(found) * block + j]
            wf[hit], wl[hit] = f[sel], la[sel]
            known[hit] = True
            miss = todo[~found]
            ptr[miss] = np.minimum(ptr[miss] + block, npi)
            todo = miss[ptr[miss] < npi]
            block = min(2 * block, 64)
        dead = np.flatnonzero(W & (ptr >= npi))
        if dead.size == 0:
            break
        W[dead] = False
    else:
        raise RuntimeError("safety fixed point did not converge")
    return Controller(abs_, None, W)


def reach_avoid(
    abs_: SymbolicAbstraction,
    target: np.ndarray,
    avoid: Optional[np.ndarray] = None,
    max_iter: int = 100_000,
    backend: str = "auto",
) -> Controller:
    """States that can be driven to ``target`` while avoiding ``avoid`` (least fixed point).

    ``value[q]`` is the number of steps needed (inf if unreachable). At a state with
    value k > 0 the allowed inputs are those whose successors all have value < k.
    Only undecided states are examined in each round; the compiled kernel also skips a
    state when no cell that entered the reach set in the last round lies in the hull of
    its successor boxes. ``backend`` as in :func:`safety`.
    """
    if getattr(abs_, "lazy", False):
        raise ValueError("reach_avoid needs an eager abstraction (lazy=False)")
    target = np.asarray(target, dtype=bool).ravel()
    avoid = np.zeros_like(target) if avoid is None else np.asarray(avoid, dtype=bool).ravel()
    if target.size != abs_.n_states or avoid.size != abs_.n_states:
        raise ValueError("target and avoid must have one entry per state cell")
    if _use_cpp(backend):
        R, value, allowed, _ = _core.reach_avoid(
            *_core_args(abs_), target.view(np.uint8), avoid.view(np.uint8)
        )
        return Controller(abs_, allowed.view(bool), R.view(bool), value)
    R = target & ~avoid
    value = np.where(R, 0.0, np.inf)
    allowed = np.zeros(abs_.enabled.shape, dtype=bool)
    undecided = np.flatnonzero(~R & ~avoid & abs_.enabled.any(axis=1))
    step = max(1, _CHUNK // max(1, abs_.n_inputs))
    for k in range(1, max_iter + 1):
        sat = _summed_area(abs_, R)
        new = []
        for s in range(0, undecided.size, step):
            r = undecided[s : s + step]
            good = abs_.enabled[r] & (_bad_in_boxes(sat, abs_.first[r], abs_.last[r]) == 0)
            won = good.any(axis=1)
            allowed[r[won]] = good[won]
            new.append(r[won])
        new = np.concatenate(new) if new else np.zeros(0, dtype=np.intp)
        if new.size == 0:
            break
        value[new] = k
        R[new] = True
        undecided = undecided[~R[undecided]]
    else:
        raise RuntimeError("reach-avoid fixed point did not converge")
    return Controller(abs_, allowed, R, value)


def simulate(
    step: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray],
    controller: Controller,
    x0,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
    pick: str = "center",
    stop_in: Optional[np.ndarray] = None,
    nominal: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """Closed-loop simulation of the concrete system ``step(x, u, rng) -> x+``.

    With ``nominal`` (a function x -> desired input) the controller acts as a safety
    filter around it (:meth:`Controller.filter`); otherwise inputs are chosen by ``pick``.
    Stops early if the state leaves the controller domain (returns what was simulated)
    or, when ``stop_in`` (a state-cell mask) is given, once the state enters it.
    Returns (states (T+1, n), inputs (T, nu)).
    """
    rng = np.random.default_rng() if rng is None else rng
    X = controller.abstraction.X
    xs, us = [np.asarray(x0, dtype=float)], []
    for _ in range(n_steps):
        x = xs[-1]
        if stop_in is not None and X.contains(x[None])[0]:
            q = np.ravel_multi_index(tuple(X.locate(x[None])[0]), X.shape)
            if stop_in[q]:
                break
        if nominal is not None:
            u = controller.filter(x, nominal(x))
        else:
            u = controller(x, pick=pick, rng=rng)
        if u is None:
            break
        us.append(np.asarray(u, dtype=float))
        xs.append(np.asarray(step(x, u, rng), dtype=float))
    nu = controller.abstraction.U.ndim
    return np.array(xs), (np.array(us) if us else np.zeros((0, nu)))
