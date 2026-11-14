import numpy as np
import pytest

from ddma import MinimalSimulatingMap


def monotone_f(Z, W):
    """Random monotone map R^n -> R^m: nonnegative weights through increasing nonlinearities."""
    return np.tanh(Z) @ W[0] + Z**3 @ W[1] + Z @ W[2]


def random_problem(rng, K=60, n=2, m=2, theta=0.3):
    W = rng.uniform(0, 1, size=(3, n, m))
    Z = rng.uniform(-1, 1, size=(K, n))
    d_lo, d_hi = -theta / 2 * np.ones(m), theta / 2 * np.ones(m)
    Y = monotone_f(Z, W) + rng.uniform(d_lo, d_hi, size=(K, m))
    return W, Z, Y, d_lo, d_hi


def test_scalar_example_from_paper():
    # Data {(1, 1.2), (2, 4.3), (3, 8.6)}, D = [-0.5, 0.5]: the map S_2 of the paper.
    S = MinimalSimulatingMap([[1.0], [2.0], [3.0]], [1.2, 4.3, 8.6], -0.5, 0.5)
    lo, hi = S.bounds([[0.5], [1.5], [2.5], [3.5]])
    np.testing.assert_allclose(lo[:, 0], [-np.inf, 0.2, 3.3, 7.6])
    np.testing.assert_allclose(hi[:, 0], [2.2, 5.3, 9.6, np.inf])


def test_soundness_contains_every_disturbed_output():
    rng = np.random.default_rng(0)
    W, Z, Y, d_lo, d_hi = random_problem(rng)
    S = MinimalSimulatingMap(Z, Y, d_lo, d_hi)
    zq = rng.uniform(-1.2, 1.2, size=(500, 2))
    lo, hi = S.bounds(zq)
    for d in (d_lo, d_hi, 0.5 * (d_lo + d_hi)):
        y = monotone_f(zq, W) + d
        assert np.all(lo <= y + 1e-12) and np.all(y <= hi + 1e-12)
    flo, fhi = S.f_bounds(zq)
    assert np.all(flo <= monotone_f(zq, W) + 1e-12) and np.all(monotone_f(zq, W) <= fhi + 1e-12)


def test_bounds_are_attained_by_consistent_monotone_maps():
    # f_up(z) = inf{y_k - d_lo : z <= z_k} is monotone, consistent with the data, and
    # f_up + d_hi equals the upper bound of S_m: the upper bound cannot be tightened.
    rng = np.random.default_rng(1)
    _, Z, Y, d_lo, d_hi = random_problem(rng, K=40)
    S = MinimalSimulatingMap(Z, Y, d_lo, d_hi)

    def f_up(z):
        above = np.all(Z[None] >= z[:, None], axis=2)
        return np.where(above[..., None], (Y - d_lo)[None], np.inf).min(axis=1)

    def f_lo(z):
        below = np.all(Z[None] <= z[:, None], axis=2)
        return np.where(below[..., None], (Y - d_hi)[None], -np.inf).max(axis=1)

    for f in (f_up, f_lo):  # consistency with the data
        v = f(Z)
        assert np.all(Y - d_hi <= v + 1e-12) and np.all(v <= Y - d_lo + 1e-12)
    zq = rng.uniform(-1, 1, size=(300, 2))
    lo, hi = S.bounds(zq)
    np.testing.assert_allclose(f_up(zq) + d_hi, hi)
    np.testing.assert_allclose(f_lo(zq) + d_lo, lo)
    # monotonicity of f_up along random ordered pairs
    a = rng.uniform(-1, 1, size=(300, 2))
    b = a + rng.uniform(0, 0.5, size=a.shape)
    assert np.all(f_up(a) <= f_up(b) + 1e-12)


def test_merge_is_intersection():
    rng = np.random.default_rng(2)
    _, Z, Y, d_lo, d_hi = random_problem(rng, K=80)
    A = MinimalSimulatingMap(Z[:40], Y[:40], d_lo, d_hi)
    B = MinimalSimulatingMap(Z[40:], Y[40:], d_lo, d_hi)
    zq = rng.uniform(-1, 1, size=(200, 2))
    lo, hi = A.merge(B).bounds(zq)
    alo, ahi = A.bounds(zq)
    blo, bhi = B.bounds(zq)
    np.testing.assert_array_equal(lo, np.maximum(alo, blo))
    np.testing.assert_array_equal(hi, np.minimum(ahi, bhi))


def test_sign_patterns_are_sound():
    rng = np.random.default_rng(3)
    # f1 increasing in z1, decreasing in z2; f2 independent of z1, increasing in z2.
    def f(z):
        return np.stack([z[:, 0] ** 3 - np.tanh(2 * z[:, 1]), np.exp(z[:, 1])], axis=1)

    Z = rng.uniform(-1, 1, size=(200, 2))
    Y = f(Z) + rng.uniform(-0.05, 0.05, size=(200, 2))
    signs = [[1, -1], [0, 1]]
    S = MinimalSimulatingMap(Z, Y, -0.05, 0.05, signs=signs)
    assert S.is_consistent()
    zq = rng.uniform(-1, 1, size=(400, 2))
    lo, hi = S.bounds(zq)
    assert np.all(lo <= f(zq) + 1e-12) and np.all(f(zq) <= hi + 1e-12)
    # the wrong pattern is detected as inconsistent with these data
    assert not MinimalSimulatingMap(Z, Y, -0.05, 0.05).is_consistent()


def test_consistency_check_detects_non_monotone_data():
    S = MinimalSimulatingMap([[0.0], [1.0]], [1.0, 0.0], -0.1, 0.1)
    assert not S.is_consistent()
    assert MinimalSimulatingMap([[0.0], [1.0]], [1.0, 0.85], -0.1, 0.1).is_consistent()


def test_input_validation():
    with pytest.raises(ValueError):
        MinimalSimulatingMap([[0.0]], [1.0], 1.0, 0.0)
    with pytest.raises(ValueError):
        MinimalSimulatingMap([[0.0], [1.0]], [1.0], 0, 0)
    with pytest.raises(ValueError):
        MinimalSimulatingMap([[0.0, 1.0]], [1.0], 0, 0, signs=[2, 1])
