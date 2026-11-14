"""GrowthMap: the growth-cone model of one scalar function, used on its own."""

import numpy as np
import pytest

from ddma import BoundedDerivativeModel, GrowthMap, RectangularPartition


def random_function(rng, n):
    a = rng.uniform(-1, 1, n)
    b = rng.uniform(0.5, 2.0, n)
    c = rng.uniform(-1, 1, n)

    def f(Z):
        return (np.sin(b * Z) * a).sum(axis=1) + Z @ c

    return f, c - np.abs(a) * b, c + np.abs(a) * b


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("local", [None, "cell", "slab"])
def test_sound(n, local):
    rng = np.random.default_rng(10 + n)
    f, g_lo, g_hi = random_function(rng, n)
    P = RectangularPartition.uniform(-np.ones(n), np.ones(n), [10] * n)
    d = 0.05
    Z = rng.uniform(-1, 1, (300, n))
    G = GrowthMap.from_data(Z, f(Z) + rng.uniform(-d, d, 300), P, g_lo, g_hi, -d, d,
                            local=local)
    T = rng.uniform(-1, 1, (20000, n))
    lo, hi = G.bounds(T)
    assert np.all(f(T) - d >= lo - 1e-12) and np.all(f(T) + d <= hi + 1e-12)
    f_lo, f_hi = G.f_bounds()
    idx = tuple(G.cell_index(T).T)
    assert np.all(f(T) >= f_lo[idx] - 1e-12) and np.all(f(T) <= f_hi[idx] + 1e-12)
    assert G.n_data == 300


@pytest.mark.parametrize("local", [None, "cell", "slab"])
def test_one_point_at_a_time_equals_batch(local):
    """The per-point path (online use) and the sweep path (batches) give the same model."""
    rng = np.random.default_rng(1)
    f, g_lo, g_hi = random_function(rng, 2)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [15, 12])
    Z = rng.uniform(-1.2, 1.2, (500, 2))
    y = f(Z)
    A = GrowthMap.from_data(Z, y, P, g_lo, g_hi, -0.01, 0.01, local=local)
    B = GrowthMap(P, g_lo, g_hi, -0.01, 0.01, local=local)
    for k in range(len(y)):
        B.update(Z[k], [y[k]])
    for attr in ("lower", "upper", "v_lower", "v_upper"):
        np.testing.assert_allclose(getattr(B, attr), getattr(A, attr), atol=1e-12, rtol=0)


def test_more_data_only_shrinks_and_copy_is_independent():
    rng = np.random.default_rng(2)
    f, g_lo, g_hi = random_function(rng, 2)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [10, 10])
    Z = rng.uniform(-1, 1, (200, 2))
    G1 = GrowthMap.from_data(Z[:50], f(Z[:50]), P, g_lo, g_hi, 0, 0)
    G2 = G1.copy().update(Z[50:], f(Z[50:]))
    assert np.all(G2.upper <= G1.upper) and np.all(G2.lower >= G1.lower)
    assert np.any(G2.upper < G1.upper)        # the copy did not change G1 ...
    assert G1.n_data == 50 and G2.n_data == 200


def test_window_update_is_sound_and_never_tighter():
    rng = np.random.default_rng(3)
    f, g_lo, g_hi = random_function(rng, 2)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [16, 16])
    Z = rng.uniform(-1, 1, (300, 2))
    full = GrowthMap.from_data(Z, f(Z), P, g_lo, g_hi, 0, 0)
    win = GrowthMap(P, g_lo, g_hi, 0, 0).update_local(Z, f(Z), window=3)
    assert np.all(win.upper >= full.upper - 1e-12) and np.all(win.lower <= full.lower + 1e-12)
    T = rng.uniform(-1, 1, (5000, 2))
    lo, hi = win.bounds(T)
    assert np.all(f(T) >= lo - 1e-12) and np.all(f(T) <= hi + 1e-12)


def test_same_as_bounded_derivative_model_growth():
    """GrowthMap is the growth model of BoundedDerivativeModel for one output."""
    rng = np.random.default_rng(4)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [8, 6])     # joint (x, u) grid
    X = rng.uniform(-1, 1, (400, 1))
    U = rng.uniform(-1, 1, (400, 1))
    Xn = 0.5 * np.sin(2 * X) + 0.3 * U
    A_lo, A_hi = np.array([[-1.0]]), np.array([[1.0]])
    B_lo, B_hi = np.array([[0.3]]), np.array([[0.3]])
    M = BoundedDerivativeModel.from_data(X, U, Xn, P, A_lo, A_hi, B_lo, B_hi, d_lo=0, d_hi=0,
                                         method="growth", local="slab")
    G = GrowthMap.from_data(np.c_[X, U], Xn[:, 0], P, [-1.0, 0.3], [1.0, 0.3], 0, 0)
    q = np.array([[0.1], [0.5], [-0.7]]), np.array([[0.2], [-0.4], [0.9]])
    lo_m, hi_m = M.bounds(*q)
    lo_g, hi_g = G.bounds(np.c_[q[0], q[1]])
    np.testing.assert_allclose(lo_g, lo_m[:, 0], atol=1e-12)
    np.testing.assert_allclose(hi_g, hi_m[:, 0], atol=1e-12)
