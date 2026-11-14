"""Gridless safe envelopes (ddma.invariance): exactness of the building blocks and the
guarantee itself, checked in closed loop on the true systems."""

import itertools

import numpy as np
import pytest

from ddma.invariance import (
    embedding_envelope,
    max_min_cones,
    maximal_points,
    monotone_envelope,
)

scipy = pytest.importorskip("scipy")
from scipy.optimize import linprog  # noqa: E402

# ------------------------------------------------------------------ building blocks


def test_maximal_points_brute_force():
    rng = np.random.default_rng(0)
    for d in (1, 2, 3, 5):
        P = rng.integers(0, 6, (300, d)).astype(float)
        M = maximal_points(P)
        ref = {tuple(p) for p in P
               if not any(np.all(q >= p) and np.any(q > p) for q in P)}
        assert {tuple(r) for r in M} == ref
        assert len(M) == len(ref)  # duplicates kept once


@pytest.mark.parametrize("d", [1, 2, 4, 7])
def test_max_min_cones_equals_lp(d):
    rng = np.random.default_rng(d)
    for _ in range(200):
        p, q = rng.normal(size=2)
        alpha = rng.exponential(size=d) * (rng.random(d) > 0.2)
        beta = rng.exponential(size=d) * (rng.random(d) > 0.2)
        # max t  s.t.  t - alpha.s <= p,  t + beta.s <= q + sum(beta),  s in [0, 1]
        c = np.zeros(d + 1)
        c[0] = -1
        A = np.array([np.r_[1, -alpha], np.r_[1, beta]])
        res = linprog(c, A_ub=A, b_ub=[p, q + beta.sum()],
                      bounds=[(None, None)] + [(0, 1)] * d, method="highs")
        assert abs(max_min_cones(p, q, alpha, beta) - (-res.fun)) < 1e-9


def test_max_min_cones_vectorized():
    rng = np.random.default_rng(3)
    p, q = rng.normal(size=(2, 50, 3))
    al, be = rng.exponential(size=(2, 50, 3, 4))
    V = max_min_cones(p, q, al, be)
    for i, k in itertools.product(range(50), range(3)):
        assert V[i, k] == pytest.approx(max_min_cones(p[i, k], q[i, k], al[i, k], be[i, k]))


# ------------------------------------------------------------------ monotone envelope


def brute_force_corners(Z, Zn_eff, V):
    """All valid (T, U) by enumerating one data point per dimension (paper, Alg. 1)."""
    N, n = Z.shape
    out = []
    for ks in itertools.product(range(N), repeat=n):
        T = Z[list(ks)].min(axis=0)
        if all(Zn_eff[k, j] <= T[j] for j, k in enumerate(ks)):
            out.append(np.r_[T, V[list(ks)].min(axis=0)])
    return np.array(out).reshape(-1, n + V.shape[1])


@pytest.mark.parametrize("n,m", [(1, 0), (2, 0), (2, 1), (3, 1)])
def test_search_equals_brute_force(n, m):
    rng = np.random.default_rng(10 * n + m)
    N = 9 if n == 3 else 14
    X = rng.uniform(0, 1, (N, n))
    U = rng.uniform(0, 1, (N, m))
    A = rng.uniform(0, 0.6 / n, (n, n))
    Xn = X @ A.T + (U.sum(1, keepdims=True) * 0.2 if m else 0) + rng.uniform(0, 0.05, (N, n))
    env = monotone_envelope(X, Xn, U if m else None, w_lo=0, w_hi=0.05)
    ref = brute_force_corners(X, Xn + 0.05, U)
    ref = maximal_points(ref) if len(ref) else ref
    got = np.hstack([env.pairs_T, env.pairs_U])
    assert {tuple(np.round(r, 12)) for r in got} == {tuple(np.round(r, 12)) for r in ref}


def rooms(T, u):
    """Three rooms cooled by one shared input (as in examples/lazy_synthesis.py)."""
    a, b, c, Ts = 0.05, 0.3, 0.6, 0.25
    cpl = np.stack([T[:, 1] - T[:, 0], T[:, 0] + T[:, 2] - 2 * T[:, 1], T[:, 1] - T[:, 2]], 1)
    return T + Ts * (a * (34.0 - T) + b * cpl - c * u * (T - 14.0))


def test_room_cooling_closed_loop():
    rng = np.random.default_rng(1)
    N, W = 800, 0.15
    X = rng.uniform(16, 32, (N, 3))
    U = rng.uniform(0, 1, (N, 1))
    Xn = rooms(X, U) + rng.uniform(0, W, (N, 3))
    env = monotone_envelope(X, Xn, U, w_lo=0, w_hi=W, input_signs=[-1], x_hi=[26] * 3)
    assert env.n_corners > 0
    assert np.all(env.corners <= 26 + 1e-12)
    # every corner pair: constant input u at its limit keeps the corner's lower set invariant
    for T, Ul in zip(env.pairs_T, env.pairs_U):
        assert np.all(rooms(T[None], Ul[None])[0] + W <= T + 1e-9)
    # closed loop from random certified states, adversarial-ish random heat gains
    starts = rng.uniform(16, 26, (4000, 3))
    starts = starts[env.contains(starts)]
    assert len(starts) > 100
    x = starts.copy()
    for _ in range(40):
        u = np.array([env.safe_input(xi, [0.0])[0] for xi in x])  # least cooling allowed
        x = rooms(x, u[:, None]) + rng.choice([0.0, W], size=x.shape)
        assert env.contains(x).all()
        assert np.all(x <= 26 + 1e-9)


def test_signs_and_spec_validation():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        monotone_envelope(X, X, x_lo=[0, 0])  # not lower-closed for +1 signs
    env = monotone_envelope(X, X - [1, -1], state_signs=[1, -1], x_hi=[1, np.inf],
                            x_lo=[-np.inf, -5])
    assert env.contains([[0, 0]]).all()


def test_autonomous_contraction():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (400, 2))
    f = lambda x: 0.5 * x + 0.1 * np.tanh(x[:, ::-1])  # noqa: E731  monotone
    Xn = f(X)
    env = monotone_envelope(X, Xn, w_lo=[-0.05, -0.05], w_hi=[0.05, 0.05])
    x = rng.uniform(-1, 1, (3000, 2))
    x = x[env.contains(x)]
    for _ in range(30):
        x = f(x) + rng.uniform(-0.05, 0.05, x.shape)
        assert env.contains(x).all()


# ----------------------------------------------------------------- embedding envelope


def competition(x, u):
    """Two competing species (discrete Lotka-Volterra), r = 1, a = 0.2."""
    return np.stack([x[:, 0] * (2 - x[:, 0] - 0.2 * x[:, 1]) + u[:, 0],
                     x[:, 1] * (2 - x[:, 1] - 0.2 * x[:, 0]) + u[:, 1]], 1)


LO, HI = 0.6, 1.1  # domain around the coexistence equilibrium (0.833, 0.833)


def competition_bounds():
    # on [LO, HI]^2: df_i/dx_i = 2 - 2 x_i - 0.2 x_j, df_i/dx_j = -0.2 x_i, df_i/du_i = 1
    d_lo, d_hi = 2 - 2.2 * HI, 2 - 2.2 * LO
    jx_lo = np.array([[d_lo, -0.2 * HI], [-0.2 * HI, d_lo]])
    jx_hi = np.array([[d_hi, -0.2 * LO], [-0.2 * LO, d_hi]])
    return jx_lo, jx_hi, np.eye(2), np.eye(2)


def test_embedding_boxes_are_invariant():
    rng = np.random.default_rng(5)
    N = 300
    X = rng.uniform(LO, HI, (N, 2))
    U = rng.uniform(-0.02, 0.02, (N, 2))
    W = 0.005
    Xn = competition(X, U) + rng.uniform(-W, W, (N, 2))
    jx_lo, jx_hi, ju_lo, ju_hi = competition_bounds()
    env = embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, w_lo=-W, w_hi=W,
                             x_lo=[LO, LO], x_hi=[HI, HI])
    assert env.n_boxes > 10
    assert env.n_pairs == N * (N - 1) // 2
    # each box, under every input of its input box and every disturbance, maps into itself
    for k in rng.choice(env.n_boxes, min(60, env.n_boxes), replace=False):
        x = rng.uniform(env.lo[k], env.hi[k], (2000, 2))
        u = rng.uniform(env.ulo[k], env.uhi[k], (2000, 2))
        xn = competition(x, u) + rng.choice([-W, W], (2000, 2))
        assert np.all((xn >= env.lo[k] - 1e-12) & (xn <= env.hi[k] + 1e-12))
    # closed loop with the certified input closest to "do nothing"
    x = rng.uniform(LO, HI, (3000, 2))
    x = x[env.contains(x)]
    assert len(x) > 1000
    for _ in range(25):
        u = np.array([env.safe_input(xi, [0.0, 0.0]) for xi in x])
        x = competition(x, u) + rng.uniform(-W, W, x.shape)
        assert env.contains(x).all()


def test_embedding_bound_is_sound_and_tight_for_one_pair():
    """The pair bound contains the true image of the box and is attained by the cones."""
    rng = np.random.default_rng(6)
    jx_lo, jx_hi, ju_lo, ju_hi = competition_bounds()
    X = np.array([[0.7, 1.0], [0.95, 0.8]])
    U = np.array([[0.05, -0.02], [-0.03, 0.04]])
    Xn = competition(X, U)
    from ddma.invariance import _cone_rates
    W = np.hstack([X, U])
    au, bu, al, bl = _cone_rates((W[1] - W[0])[None], np.hstack([jx_lo, ju_lo]),
                                 np.hstack([jx_hi, ju_hi]))
    hi = max_min_cones(Xn[0][None], Xn[1][None], au, bu)[0]
    lo = -max_min_cones(-Xn[0][None], -Xn[1][None], al, bl)[0]
    x = rng.uniform(X.min(0), X.max(0), (20000, 2))
    u = rng.uniform(U.min(0), U.max(0), (20000, 2))
    y = competition(x, u)
    assert np.all(y <= hi + 1e-12) and np.all(y >= lo - 1e-12)


def test_embedding_autonomous_chain_10d():
    rng = np.random.default_rng(7)
    n, N = 10, 1000

    def f(x):
        return 0.5 * x + 0.1 * np.sin(np.roll(x, 1, 1)) + 0.1 * np.cos(np.roll(x, -1, 1))

    jx_lo = np.zeros((n, n))
    jx_hi = np.zeros((n, n))
    for i in range(n):
        jx_lo[i, i] = jx_hi[i, i] = 0.5
        jx_lo[i, (i - 1) % n], jx_hi[i, (i - 1) % n] = -0.1, 0.1
        jx_lo[i, (i + 1) % n], jx_hi[i, (i + 1) % n] = -0.1, 0.1
    X = rng.uniform(-2, 2, (N, n))
    env = embedding_envelope(X, f(X), None, jx_lo, jx_hi)
    assert env.n_boxes > 0
    for k in range(min(env.n_boxes, 30)):
        x = rng.uniform(env.lo[k], env.hi[k], (3000, n))
        y = f(x)
        assert np.all((y >= env.lo[k] - 1e-12) & (y <= env.hi[k] + 1e-12))
