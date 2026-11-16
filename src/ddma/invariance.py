"""Safe operating envelopes (controlled invariant sets) straight from data, without a grid.

Two methods, both deterministic and both working on raw transitions (x_k, u_k, x_k+):

* :func:`monotone_envelope` for monotone systems with an additive bounded disturbance,

      x+ = f(x, u) + w,   f monotone,   w in [w_lo, w_hi],

  and a lower-closed safety specification (e.g. "every temperature at most 26 C").
  Each valid combination of data points gives a corner T whose lower set {x <= T} is
  robustly invariant with a constant input; the envelope is the union of these lower
  sets, stored as its maximal corners (a *skyline*) with the largest allowed input of
  each. Method of A. Makdesi, M. Zamani, "Data-driven synthesis of controlled invariant
  sets for monotone systems", IFAC World Congress 2026.

* :func:`embedding_envelope` for general nonlinear systems with bounded Jacobians,

      x+ = f(x, u) + w,   df_i/dx_j in [a_ij, b_ij],  df_i/du_k in [c_ik, d_ik].

  Every pair of data points spans a box of states and a box of inputs. The growth cones
  of the two points bound where the whole box can go (an LP with two cone constraints,
  solved here exactly in closed form); if that bound fits back inside the box, the box
  is robustly invariant for every input in its input box. Method of A. Makdesi,
  M. Zamani, S. Jafarpour, "Data-driven controlled invariance via monotone embeddings",
  IFAC World Congress 2026 (the box test is the paper's monotone-embedding condition
  z' <= z on the pair data).

Coordinates: monotone systems are often monotone only after flipping some coordinates
(e.g. distance -> -distance, cooling -> -cooling). Pass ``state_signs`` / ``input_signs``
(+1 or -1 per coordinate) and work in the original coordinates everywhere else.

Disturbances are in state units (the change they cause in x+ in one step). In the data,
x_k+ = f(x_k, u_k) + w_k with w_k unknown; the guarantee holds for every w in the bounds.

Example::

    env = monotone_envelope(X, Xn, U, w_lo=[0, 0, 0], w_hi=[0.15] * 3,
                            input_signs=[-1], x_hi=[26, 26, 26])
    env.contains(x)            # in the certified envelope?
    u = env.safe_input(x, u_desired)   # closest certified input (None outside)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_DEBUG = False

__all__ = [
    "MonotoneEnvelope",
    "BoxEnvelope",
    "monotone_envelope",
    "embedding_envelope",
    "maximal_points",
    "max_min_cones",
]


# ----------------------------------------------------------------------------- helpers


def maximal_points(P, return_index: bool = False):
    """Rows of P that no other row dominates (componentwise >=; duplicates kept once).

    Sort-filter skyline: rows sorted by decreasing sum can only be dominated by earlier
    rows, so each block is compared with the current skyline and with its own earlier rows.
    """
    P = np.asarray(P, float)
    if P.ndim != 2:
        raise ValueError("P must be 2-D")
    if len(P) == 0:
        return (P, np.zeros(0, int)) if return_index else P
    order = np.lexsort(tuple(-P[:, ::-1].T) + (-P.sum(axis=1),))
    S = P[order]
    keep = []
    front = np.empty((0, P.shape[1]))
    block = 256
    for start in range(0, len(S), block):
        C = S[start:start + block]
        idx = np.arange(start, start + len(C))
        if len(front):
            dom = np.zeros(len(C), bool)
            for f0 in range(0, len(front), 4096):
                F = front[f0:f0 + 4096]
                dom |= np.all(F[None, :, :] >= C[:, None, :], axis=2).any(axis=1)
            C, idx = C[~dom], idx[~dom]
        if len(C) == 0:
            continue
        # within the block: j < i dominates i
        ge = np.all(C[None, :, :] >= C[:, None, :], axis=2)  # ge[i, j]: C_j >= C_i
        ge = np.tril(ge, k=-1)
        ok = ~ge.any(axis=1)
        C, idx = C[ok], idx[ok]
        front = np.vstack([front, C])
        keep.append(idx)
    keep = order[np.concatenate(keep)] if keep else np.zeros(0, int)
    return (P[keep], keep) if return_index else P[keep]


def max_min_cones(p, q, alpha, beta):
    """max over s in [0, 1]^d of min(p + alpha . s, q + beta . (1 - s)), exactly.

    Vectorized over leading axes (p, q: (...), alpha, beta: (..., d), all alpha, beta >= 0).
    This is the LP of the embedding method: two cones with apexes at opposite corners of a
    box, each increasing towards the other corner. By LP duality it equals
    min over lam in [0, 1] of  lam p + (1 - lam)(q + sum beta) + sum max(0, lam(alpha+beta) - beta),
    a convex piecewise-linear function whose minimum sits at 0, 1 or a breakpoint
    beta_j / (alpha_j + beta_j); the slope test below finds it after one sort.
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    alpha = np.asarray(alpha, float)
    beta = np.asarray(beta, float)
    tot = alpha + beta
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.where(tot > 0, beta / np.where(tot > 0, tot, 1.0), 2.0)  # 2: never reached
    order = np.argsort(lam, axis=-1)
    lam_s = np.take_along_axis(lam, order, -1)
    tot_s = np.take_along_axis(tot, order, -1)
    # slope just after the k-th breakpoint: s_k = p - q - sum(beta) + cumsum(tot)_k
    s0 = p - q - beta.sum(-1)
    slope = s0[..., None] + np.cumsum(tot_s, -1)
    # the minimum is at the first breakpoint after which the slope is >= 0 (or at 0 / 1)
    nonneg = slope >= 0
    first = np.argmax(nonneg, axis=-1)
    has = nonneg.any(axis=-1)
    lam_star = np.take_along_axis(lam_s, first[..., None], -1)[..., 0]
    lam_star = np.where(s0 >= 0, 0.0, np.where(has, lam_star, 1.0))
    lam_star = np.clip(lam_star, 0.0, 1.0)
    L = lam_star[..., None]
    val = (lam_star * p + (1 - lam_star) * (q + beta.sum(-1))
           + np.maximum(0.0, L * tot - beta).sum(-1))
    return val


def _vec(v, n, name, fill):
    if v is None:
        return np.full(n, fill, float)
    v = np.broadcast_to(np.asarray(v, float), (n,)).copy()
    if v.shape != (n,):
        raise ValueError(f"{name} must have length {n}")
    return v


def _signs(s, n, name):
    s = _vec(s, n, name, 1.0)
    if not np.all(np.isin(s, (-1.0, 1.0))):
        raise ValueError(f"{name} must contain only +1 and -1")
    return s


def _as_2d(A, name):
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    return A


# ------------------------------------------------------------------ monotone envelope


@dataclass
class MonotoneEnvelope:
    """Union of lower sets {x : s * x <= s * T} (s = state signs), with allowed inputs.

    Attributes
    ----------
    corners : (L, n) maximal corners T, in original coordinates (the skyline).
    pairs_T, pairs_U : (P, n), (P, m) every non-dominated (corner, input limit) pair:
        for x below T (in the order of ``state_signs``), every input u with
        input_signs * u <= input_signs * U and inside [u_lo, u_hi] is certified.
    state_signs, input_signs : the coordinate flips that make f monotone increasing.
    u_lo, u_hi : the input range in which monotonicity was assumed (default: the data's).
    n_valid : number of valid corner candidates found by the search (before the skyline).
    """

    corners: np.ndarray
    pairs_T: np.ndarray
    pairs_U: np.ndarray
    state_signs: np.ndarray
    input_signs: np.ndarray
    u_lo: np.ndarray
    u_hi: np.ndarray
    n_valid: int = 0

    @property
    def n_corners(self) -> int:
        """Number of maximal corners."""
        return len(self.corners)

    def contains(self, x) -> np.ndarray:
        """Boolean per state (rows of x): inside the certified envelope."""
        x = np.atleast_2d(np.asarray(x, float)) * self.state_signs
        if self.n_corners == 0:
            return np.zeros(len(x), bool)
        T = self.corners * self.state_signs
        out = np.zeros(len(x), bool)
        for s in range(0, len(x), 4096):
            out[s:s + 4096] = np.all(x[s:s + 4096, None, :] <= T[None] + 0.0, axis=2).any(1)
        return out

    def input_limits(self, x) -> np.ndarray:
        """Maximal input limits U valid at one state x (rows; empty if x is outside)."""
        z = np.asarray(x, float).reshape(-1) * self.state_signs
        T = self.pairs_T * self.state_signs
        ok = np.all(z[None] <= T, axis=1)
        return self.pairs_U[ok]

    def safe_input(self, x, u_desired=None) -> Optional[np.ndarray]:
        """A certified input at x, as close as possible to ``u_desired`` (None outside).

        Without ``u_desired`` the input closest to the middle of [u_lo, u_hi] is returned.
        """
        lim = self.input_limits(x)
        if len(lim) == 0:
            return None
        su = self.input_signs
        v_lo = np.minimum(self.u_lo * su, self.u_hi * su)
        if u_desired is None:
            u_desired = 0.5 * (self.u_lo + self.u_hi)
        vd = np.asarray(u_desired, float).reshape(-1) * su
        cand = np.maximum(np.minimum(vd[None], lim * su), v_lo[None])
        best = np.argmin(np.sum((cand - vd[None]) ** 2, axis=1))
        return cand[best] * su


def _monotone_search(Z, Zn_eff, V, order, max_states, z_max):
    """All non-dominated valid (T, U) from data, transformed coordinates (maximize all).

    State of a partial choice (dims order[:l] chosen): m = min of chosen x (n),
    mu = min of chosen u (mv), t = thresholds x_eff+ of the chosen point in its own dim.
    Final validity: t_j <= m_j for every dim j; m only decreases, so it is checked at every
    layer. States dominated in (m, mu, -t) are dropped: they can only lead to dominated or
    invalid corners.
    """
    n = Z.shape[1]
    mv = V.shape[1]
    m = np.full((1, n), np.inf)
    mu = np.full((1, mv), np.inf)
    t = np.zeros((1, 0))
    # candidate walls per dimension: the worst-case successor did not increase in that
    # coordinate, and it already meets the specification there (t <= T <= z_max)
    cands = [np.flatnonzero((Zn_eff[:, d] <= Z[:, d]) & (Zn_eff[:, d] <= z_max[d]))
             for d in order]
    # lowest value each coordinate can still reach from the remaining layers
    floor = [np.min(np.vstack([Z[c] for c in cands[i + 1:]]), axis=0) if i + 1 < len(order)
             and all(len(c) for c in cands[i + 1:]) else np.full(n, np.inf)
             for i in range(len(order))]
    for layer, dim in enumerate(order):
        cand = cands[layer]
        if len(cand) == 0:
            return np.zeros((0, n)), np.zeros((0, mv)), 0
        Zc, Vc, tc = Z[cand], V[cand], Zn_eff[cand, dim]
        new_m, new_mu, new_t = [], [], []
        chosen = np.asarray(order[:layer + 1])
        block = max(1, 2_000_000 // (len(cand) * (n + mv + 1)))
        for s in range(0, len(m), block):
            M = np.minimum(m[s:s + block, None, :], Zc[None])          # (P, Q, n)
            MU = np.minimum(mu[s:s + block, None, :], Vc[None])       # (P, Q, mv)
            Tt = np.concatenate([np.broadcast_to(t[s:s + block, None, :],
                                                 (len(M), len(cand), layer)),
                                 np.broadcast_to(tc[None, :, None], (len(M), len(cand), 1))],
                                axis=2)
            ok = np.all(Tt <= M[:, :, chosen], axis=2)
            new_m.append(M[ok])
            new_mu.append(MU[ok])
            new_t.append(Tt[ok])
        m = np.concatenate(new_m)
        mu = np.concatenate(new_mu)
        t = np.concatenate(new_t)
        # a threshold at or below every value still to come can never bind again
        slack = t <= floor[layer][chosen][None, :]
        t = np.where(slack, -np.inf, t)
        last = layer == len(order) - 1
        if last:  # the corner is final: it must satisfy the safety specification
            inside = np.all(m <= z_max, axis=1)
            m, mu, t = m[inside], mu[inside], t[inside]
        if len(m) == 0:
            return np.zeros((0, n)), np.zeros((0, mv)), 0
        key = np.hstack([m, mu] if last else [m, mu, -t])  # thresholds are spent at the end
        key = np.where(np.isfinite(key), key, np.where(key > 0, 1e300, -1e300))
        _, first = np.unique(key, axis=0, return_index=True)  # exact duplicates are common
        _, idx = maximal_points(key[first], return_index=True)
        idx = first[idx]
        m, mu, t = m[idx], mu[idx], t[idx]
        if _DEBUG:
            print(f"layer {layer} (dim {dim}): {len(key)} expanded, {len(first)} distinct, "
                  f"{len(idx)} kept")
        if len(m) > max_states:
            raise MemoryError(
                f"{len(m)} partial corners after dimension {dim}; reduce the data or raise "
                "max_states")
    return m, mu, len(m)


def monotone_envelope(X, Xn, U=None, w_lo=None, w_hi=None, x_lo=None, x_hi=None,
                      state_signs=None, input_signs=None, u_lo=None, u_hi=None,
                      max_states: int = 2_000_000) -> MonotoneEnvelope:
    """Certified safe envelope of a monotone system from transitions (no model, no grid).

    Parameters
    ----------
    X, Xn : (N, n) states and measured successors, x_k+ = f(x_k, u_k) + w_k.
    U : (N, m) inputs, or None for an autonomous system.
    w_lo, w_hi : bounds on the additive disturbance, in state units (default 0).
    x_lo, x_hi : safety specification as a box (use +-inf for "no bound"). It must be
        lower-closed in the flipped coordinates: only x_hi where ``state_signs`` is +1,
        only x_lo where it is -1.
    state_signs, input_signs : +1 / -1 per coordinate so that f is increasing in
        (state_signs * x, input_signs * u).
    u_lo, u_hi : input range in which f is known to be monotone; certified inputs stay in
        it (default: the range of the data).

    Returns
    -------
    MonotoneEnvelope. Every state in it can be kept in it forever, for every disturbance
    sequence, with the inputs it certifies.
    """
    X = _as_2d(X, "X")
    Xn = _as_2d(Xn, "Xn")
    N, n = X.shape
    if Xn.shape != (N, n):
        raise ValueError("X and Xn must have the same shape")
    if U is None:
        U = np.zeros((N, 0))
    U = _as_2d(U, "U")
    if len(U) != N:
        raise ValueError("U must have one row per transition")
    m = U.shape[1]
    s = _signs(state_signs, n, "state_signs")
    su = _signs(input_signs, m, "input_signs") if m else np.zeros(0)
    w_lo = _vec(w_lo, n, "w_lo", 0.0)
    w_hi = _vec(w_hi, n, "w_hi", 0.0)
    if np.any(w_lo > w_hi):
        raise ValueError("w_lo must be <= w_hi")
    x_lo = _vec(x_lo, n, "x_lo", -np.inf)
    x_hi = _vec(x_hi, n, "x_hi", np.inf)
    if np.any(np.isfinite(x_lo) & (s > 0)) or np.any(np.isfinite(x_hi) & (s < 0)):
        raise ValueError("the safety box must be lower-closed in the flipped coordinates: "
                         "give x_hi where state_signs = +1 and x_lo where it is -1")
    u_lo = _vec(u_lo, m, "u_lo", np.nan) if m else np.zeros(0)
    u_hi = _vec(u_hi, m, "u_hi", np.nan) if m else np.zeros(0)
    if m:
        u_lo = np.where(np.isnan(u_lo), U.min(axis=0), u_lo)
        u_hi = np.where(np.isnan(u_hi), U.max(axis=0), u_hi)

    # transformed coordinates: f increasing, safe set = {z <= z_max}
    Z = X * s
    V = U * su if m else U
    z_max = np.where(s > 0, x_hi, -x_lo)
    # worst-case successor: f(x_k) <= x_k+ - w_lo, so f + w <= x_k+ + (w_hi - w_lo)
    Zn_eff = Xn * s + (w_hi - w_lo)

    counts = [(Zn_eff[:, j] <= Z[:, j]).sum() for j in range(n)]
    order = list(np.argsort(counts, kind="stable"))
    T, Uv, n_valid = _monotone_search(Z, Zn_eff, V, order, max_states, z_max)
    keep = np.all(T <= z_max, axis=1)
    T, Uv = T[keep], Uv[keep]
    if len(T):
        TU, _ = maximal_points(np.hstack([T, Uv]), return_index=True)
        T, Uv = TU[:, :n], TU[:, n:]
        corners = maximal_points(T)
    else:
        corners = np.zeros((0, n))
    return MonotoneEnvelope(
        corners=corners * s, pairs_T=T * s, pairs_U=Uv * su if m else Uv,
        state_signs=s, input_signs=su, u_lo=u_lo, u_hi=u_hi, n_valid=int(n_valid))


# ----------------------------------------------------------------- embedding envelope


@dataclass
class BoxEnvelope:
    """Union of robustly invariant boxes, each with a box of certified inputs.

    Attributes
    ----------
    lo, hi : (K, n) the state boxes.
    ulo, uhi : (K, m) for every state in box k, every input in [ulo_k, uhi_k] keeps the
        successor in box k, for every disturbance.
    pairs : (K, 2) indices of the data points spanning each box.
    n_pairs : number of pairs examined.
    """

    lo: np.ndarray
    hi: np.ndarray
    ulo: np.ndarray
    uhi: np.ndarray
    pairs: np.ndarray
    n_pairs: int = 0

    @property
    def n_boxes(self) -> int:
        """Number of certified boxes."""
        return len(self.lo)

    def boxes_containing(self, x) -> np.ndarray:
        """Indices of the boxes that contain the state x."""
        x = np.asarray(x, float).reshape(-1)
        return np.flatnonzero(np.all((self.lo <= x) & (x <= self.hi), axis=1))

    def contains(self, x) -> np.ndarray:
        """Boolean per state (rows of x): inside some certified box."""
        x = np.atleast_2d(np.asarray(x, float))
        out = np.zeros(len(x), bool)
        if self.n_boxes == 0:
            return out
        for s in range(0, len(x), 2048):
            xs = x[s:s + 2048, None, :]
            out[s:s + 2048] = np.all((self.lo[None] <= xs) & (xs <= self.hi[None]), 2).any(1)
        return out

    def safe_input(self, x, u_desired=None) -> Optional[np.ndarray]:
        """Certified input at x closest to ``u_desired`` (None if x is in no box)."""
        k = self.boxes_containing(x)
        if len(k) == 0:
            return None
        lo, hi = self.ulo[k], self.uhi[k]
        if u_desired is None:
            u_desired = 0.5 * (lo[0] + hi[0])
        ud = np.asarray(u_desired, float).reshape(-1)
        cand = np.clip(ud[None], lo, hi)
        return cand[np.argmin(np.sum((cand - ud) ** 2, axis=1))]

    def bounding_box(self):
        """(lo, hi) of the smallest box containing every certified box."""
        return self.lo.min(axis=0), self.hi.max(axis=0)


def _cone_rates(delta, a, b):
    """Rates of the upper/lower cones along the box, for every pair and output.

    delta: (P, d) displacement from point mu to point nu; a, b: (n, d) derivative bounds.
    Returns alpha_up, beta_up, alpha_lo, beta_lo: (P, n, d), all >= 0.
    """
    ad = np.abs(delta)[:, None, :]
    pos = (delta > 0)[:, None, :]
    bp = np.maximum(b, 0.0)[None]
    am = np.maximum(-a, 0.0)[None]
    alpha_up = ad * np.where(pos, bp, am)   # upper cone from mu, growing towards nu
    beta_up = ad * np.where(pos, am, bp)    # upper cone from nu, growing towards mu
    alpha_lo = ad * np.where(pos, am, bp)   # -(lower cone) from mu
    beta_lo = ad * np.where(pos, bp, am)
    return alpha_up, beta_up, alpha_lo, beta_lo


def embedding_envelope(X, Xn, U=None, jx_lo=None, jx_hi=None, ju_lo=None, ju_hi=None,
                       w_lo=None, w_hi=None, x_lo=None, x_hi=None, pairs=None,
                       chunk: int = 20_000) -> BoxEnvelope:
    """Certified safe envelope of a nonlinear system with bounded Jacobians, from data.

    Parameters
    ----------
    X, Xn : (N, n) states and measured successors, x_k+ = f(x_k, u_k) + w_k.
    U : (N, m) inputs, or None.
    jx_lo, jx_hi : (n, n) bounds on df_i/dx_j (scalars broadcast).
    ju_lo, ju_hi : (n, m) bounds on df_i/du_k (needed when U is given).
    w_lo, w_hi : additive disturbance bounds, in state units (default 0).
    x_lo, x_hi : safety box (default unbounded); every certified box lies inside it.
    pairs : (P, 2) index pairs to examine (default: all pairs i < j).

    Returns
    -------
    BoxEnvelope: boxes [min(x_i, x_j), max(x_i, x_j)] whose image under f + w, for every
    input in [min(u_i, u_j), max(u_i, u_j)], provably stays inside them.
    """
    X = _as_2d(X, "X")
    Xn = _as_2d(Xn, "Xn")
    N, n = X.shape
    if Xn.shape != (N, n):
        raise ValueError("X and Xn must have the same shape")
    if U is None:
        U = np.zeros((N, 0))
    U = _as_2d(U, "U")
    m = U.shape[1]
    if jx_lo is None or jx_hi is None:
        raise ValueError("jx_lo and jx_hi are required")
    jx_lo = np.broadcast_to(np.asarray(jx_lo, float), (n, n))
    jx_hi = np.broadcast_to(np.asarray(jx_hi, float), (n, n))
    if m:
        if ju_lo is None or ju_hi is None:
            raise ValueError("ju_lo and ju_hi are required when inputs are given")
        ju_lo = np.broadcast_to(np.asarray(ju_lo, float), (n, m))
        ju_hi = np.broadcast_to(np.asarray(ju_hi, float), (n, m))
    else:
        ju_lo = ju_hi = np.zeros((n, 0))
    if np.any(jx_lo > jx_hi) or np.any(ju_lo > ju_hi):
        raise ValueError("derivative bounds must satisfy lo <= hi")
    a = np.hstack([jx_lo, ju_lo])
    b = np.hstack([jx_hi, ju_hi])
    w_lo = _vec(w_lo, n, "w_lo", 0.0)
    w_hi = _vec(w_hi, n, "w_hi", 0.0)
    if np.any(w_lo > w_hi):
        raise ValueError("w_lo must be <= w_hi")
    x_lo = _vec(x_lo, n, "x_lo", -np.inf)
    x_hi = _vec(x_hi, n, "x_hi", np.inf)
    width = w_hi - w_lo
    y_up = Xn + width   # f(x_k) + w <= x_k+ - w_lo + w_hi
    y_lo = Xn - width   # f(x_k) + w >= x_k+ - w_hi + w_lo
    # a box contains both endpoints, so each endpoint's successor bound must fit in it
    W = np.hstack([X, U])

    if pairs is None:
        first, second = np.triu_indices(N, k=1)
    else:
        pairs = np.asarray(pairs, int)
        first, second = pairs[:, 0], pairs[:, 1]
    n_pairs = len(first)
    out_lo, out_hi, out_ulo, out_uhi, out_p = [], [], [], [], []
    for s in range(0, len(first), chunk):
        i, j = first[s:s + chunk], second[s:s + chunk]
        lo = np.minimum(X[i], X[j])
        hi = np.maximum(X[i], X[j])
        ok = np.all(lo >= x_lo, 1) & np.all(hi <= x_hi, 1)
        ok &= np.all((y_up[i] <= hi) & (y_up[j] <= hi) & (y_lo[i] >= lo) & (y_lo[j] >= lo), 1)
        i, j, lo, hi = i[ok], j[ok], lo[ok], hi[ok]
        if len(i) == 0:
            continue
        delta = W[j] - W[i]
        au, bu, al, bl = _cone_rates(delta, a, b)
        g_hi = max_min_cones(y_up[i], y_up[j], au, bu)          # (P, n)
        g_lo = -max_min_cones(-y_lo[i], -y_lo[j], al, bl)
        ok = np.all((g_hi <= hi) & (g_lo >= lo), axis=1)
        if not ok.any():
            continue
        i, j = i[ok], j[ok]
        out_lo.append(lo[ok])
        out_hi.append(hi[ok])
        out_ulo.append(np.minimum(U[i], U[j]))
        out_uhi.append(np.maximum(U[i], U[j]))
        out_p.append(np.stack([i, j], 1))
    if out_lo:
        cat = np.concatenate
        return BoxEnvelope(cat(out_lo), cat(out_hi), cat(out_ulo), cat(out_uhi), cat(out_p),
                           n_pairs)
    z = np.zeros((0, n))
    zu = np.zeros((0, m))
    return BoxEnvelope(z, z.copy(), zu, zu.copy(), np.zeros((0, 2), int), n_pairs)
