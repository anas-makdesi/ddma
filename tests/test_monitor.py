"""Assumption monitor: each kind of wrong assumption is detected and diagnosed exactly."""

import numpy as np
import pytest

import ddma
from ddma import (
    AssumptionMonitor,
    BoundedDerivativeModel,
    GrowthMap,
    MinimalSimulatingMap,
    PartitionedSimulatingMap,
    RectangularPartition,
)
from ddma.bounded_derivatives import METHODS


def noisy(rng, f, K, n, d, lo=0.0, hi=1.0):
    Z = rng.uniform(lo, hi, (K, n))
    return Z, f(Z) + rng.uniform(-d, d, K)


def brute_width(Z, Y, lo, hi):
    """Smallest disturbance width making every pair consistent (all pairs, brute force)."""
    dz = Z[:, None, :] - Z[None, :, :]
    with np.errstate(invalid="ignore"):
        c = np.where(dz > 0, hi * dz, np.where(dz < 0, lo * dz, 0.0)).sum(-1)
    return float(np.max(Y[:, None] - Y[None, :] - c))


# ---------------------------------------------------------------------- no false alarms
def test_correct_assumptions_never_alarm():
    rng = np.random.default_rng(0)
    f = lambda Z: Z[:, 0] ** 3 + np.sin(2 * Z[:, 1])  # noqa: E731  slopes in [0,3]x[-2,2]
    Z, y = noisy(rng, f, 3000, 2, 0.05, -1, 1)
    mon = AssumptionMonitor([[0.0, -2.0]], [[3.0, 2.0]], -0.05, 0.05)
    mon.add(Z[:500], y[:500])
    assert mon.check(Z[500:], y[500:]) == []
    assert mon.ok and mon.consistent()
    assert "0 inconsistent" in mon.report()


@pytest.mark.parametrize("method", METHODS)
def test_correct_bd_model_never_alarms(method):
    if method.startswith("lin_"):
        pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    f = lambda Z: 0.5 * np.sin(2 * Z[:, 0]) + 0.3 * Z[:, 1]  # noqa: E731
    P = RectangularPartition.uniform([-1, -1], [1, 1], [8, 8])
    Z, y = noisy(rng, f, 400, 2, 0.02, -1, 1)
    model = BoundedDerivativeModel.from_data(Z[:, :1], Z[:, 1:], y[:, None], P,
                                             [[-1.0]], [[1.0]], [[0.3]], [[0.3]],
                                             -0.02, 0.02, method=method)
    mon = AssumptionMonitor.for_model(model, Z, y[:, None])
    Zn, yn = noisy(rng, f, 300, 2, 0.02, -1, 1)
    assert mon.check(Zn, yn[:, None]) == []
    growth = method in ("growth", "lin_growth")  # others rely on [min(lo, 0), max(hi, 0)]
    assert np.allclose(mon.slope_lo, [[-1.0, 0.3 if growth else 0.0]])
    assert np.allclose(mon.slope_hi, [[1.0, 0.3]])


def test_correct_monotone_maps_never_alarm():
    rng = np.random.default_rng(2)
    f = lambda Z: np.c_[Z[:, 0] + Z[:, 1] ** 3, Z[:, 0] - Z[:, 1]]  # noqa: E731
    Z = rng.uniform(0, 1, (800, 2))
    Y = f(Z) + rng.uniform(-0.03, 0.03, (800, 2))
    signs = [[1, 1], [1, -1]]
    P = RectangularPartition.uniform([0, 0], [1, 1], [12, 12])
    psm = PartitionedSimulatingMap.from_data(Z, Y, P, -0.03, 0.03, signs)
    msm = MinimalSimulatingMap(Z, Y, -0.03, 0.03, signs)
    Zn = rng.uniform(0, 1, (300, 2))
    Yn = f(Zn) + rng.uniform(-0.03, 0.03, (300, 2))
    for model in (psm, msm):
        mon = AssumptionMonitor.for_model(model, Z, Y)
        assert mon.check(Zn, Yn) == []


# ---------------------------------------------------------------------- each wrong assumption
def test_wrong_sign_detected_and_measured():
    """f decreasing in z_1 but assumed non-decreasing (monotone map)."""
    rng = np.random.default_rng(3)
    f = lambda Z: Z[:, 0] - 2.0 * Z[:, 1]  # noqa: E731
    Z, y = noisy(rng, f, 600, 2, 0.01)
    P = RectangularPartition.uniform([0, 0], [1, 1], [10, 10])
    psm = PartitionedSimulatingMap.from_data(Z[:200], y[:200], P, -0.01, 0.01)
    mon = AssumptionMonitor.for_model(psm, Z[:200], y[:200])
    viol = mon.check(Z[200:], y[200:])
    assert viol and not mon.ok
    need = mon.needed()
    # the sign of z_1 is the culprit: its lower slope bound must drop to about -2
    assert -2.0 <= need["slope_lo"][0, 1] <= -1.9
    assert np.isnan(need["slope_hi"]).all()  # upper bounds cannot explain anything
    assert "contradicts 'non-decreasing'" in mon.report()
    # the exact claim: changing that one bound to the needed value fixes all data
    mon.widen(("slope_lo", 0, 1))
    assert mon.consistent() and mon.needs_refit


def test_lipschitz_too_small_detected_and_exact():
    rng = np.random.default_rng(4)
    f = lambda Z: 3.0 * Z[:, 0] + 0.5 * Z[:, 1]  # noqa: E731
    Z, y = noisy(rng, f, 500, 2, 0.02)
    lo, hi = np.array([[-1.0, -1.0]]), np.array([[1.0, 1.0]])
    mon = AssumptionMonitor(lo, hi, -0.02, 0.02)
    mon.add(Z[:100], y[:100])
    assert mon.check(Z[100:], y[100:])
    need = mon.needed()
    assert 2.8 <= need["slope_hi"][0, 0] <= 3.0
    assert np.isnan(need["slope_lo"][0, 0])
    # exact: the needed bound is the smallest that works (bisection with consistent())
    a, b = 1.0, need["slope_hi"][0, 0]
    for _ in range(40):
        mid = 0.5 * (a + b)
        mon.slope_hi[0, 0] = mid
        a, b = (a, mid) if mon.consistent() else (mid, b)
    assert abs(b - need["slope_hi"][0, 0]) < 1e-6
    # and the needed width equals the brute-force one
    Zs, Ys = mon.data()
    assert np.isclose(need["width"][0], brute_width(Zs, Ys[:, 0], lo[0], hi[0]))


def test_missing_dependence_detected():
    """Assumed: f does not depend on z_1 (slope 0 / not a feature). Wrong."""
    rng = np.random.default_rng(5)
    f = lambda Z: Z[:, 0] + 0.8 * Z[:, 1]  # noqa: E731
    Z, y = noisy(rng, f, 400, 2, 0.01)
    mon = AssumptionMonitor.monotone([1, 0], -0.01, 0.01, m=1, n=2)
    mon.add(Z[:100], y[:100])
    assert mon.check(Z[100:], y[100:])
    need = mon.needed()
    # an upper slope bound of about 0.8 on z_1 explains all violations
    assert 0.7 <= need["slope_hi"][0, 1] <= 0.8
    assert "does not depend on it" in mon.report()


def test_disturbance_too_large_detected_and_measured():
    rng = np.random.default_rng(6)
    f = lambda Z: 0.5 * Z[:, 0]  # noqa: E731
    Z, y = noisy(rng, f, 2000, 1, 0.2)  # true width 0.4
    mon = AssumptionMonitor([[-1.0]], [[1.0]], -0.05, 0.05)
    mon.add(Z[:200], y[:200])
    assert mon.check(Z[200:], y[200:])
    need = mon.needed()
    assert 0.35 <= need["width"][0] <= 0.4
    # slope bounds alone would need to be far outside [-1, 1]: a disturbance problem
    assert np.isnan(need["slope_hi"][0, 0]) or need["slope_hi"][0, 0] > 10
    mon.widen("disturbance")
    assert mon.consistent()
    assert np.isclose(mon.theta[0], need["width"][0])


def test_system_change_detected_through_model_bounds_only():
    """With a tiny buffer the witness may be gone; the model bounds still catch it."""
    rng = np.random.default_rng(7)
    g = GrowthMap.from_data(*noisy(rng, lambda Z: Z[:, 0], 500, 1, 0.01),
                            RectangularPartition.uniform([0], [1], [20]), [0.5], [1.5],
                            -0.01, 0.01)
    mon = AssumptionMonitor.for_model(g, max_points=1)
    viol = mon.check([[0.5]], [[0.5 + 0.5]])  # f jumped by 0.5
    assert viol and viol[0].kind == "model" and viol[0].gap > 0.3


# ---------------------------------------------------------------------- responses
def test_widen_policy_keeps_data_consistent_and_refit_contains_data():
    rng = np.random.default_rng(8)
    f = lambda Z: Z[:, 0] + Z[:, 1]  # noqa: E731
    Z, y = noisy(rng, f, 300, 2, 0.01)
    P = RectangularPartition.uniform([0, 0], [1, 1], [8, 8])
    psm = PartitionedSimulatingMap.from_data(Z, y, P, -0.01, 0.01)
    mon = AssumptionMonitor.for_model(psm, Z, y, on_violation="widen", margin=0.1)
    Zn, yn = noisy(rng, f, 300, 2, 0.1)  # the disturbance grew
    mon.check(Zn, yn)
    assert mon.changes and mon.needs_refit and mon.model_bounds is None
    assert mon.consistent()
    Zs, Ys = mon.data()
    new = PartitionedSimulatingMap.from_data(Zs, Ys, P, mon.d_lo, mon.d_hi)
    assert new.contains(Zs, Ys).all() and not new.is_empty().any()
    mon.set_model(new.bounds, P.contains)
    assert not mon.needs_refit


def test_fallback_latches_until_reset_and_callback():
    rng = np.random.default_rng(9)
    seen = []
    mon = AssumptionMonitor([[0.0]], [[1.0]], -0.01, 0.01, on_violation="fallback",
                            callback=seen.append)
    Z, y = noisy(rng, lambda Z: Z[:, 0], 50, 1, 0.01)
    mon.add(Z, y)
    assert mon.check([[0.5]], [0.5]) == [] and not mon.fallback_active
    mon.check([[0.5]], [2.0])
    assert mon.fallback_active and len(seen) == 1
    mon.check([[0.6]], [0.6])
    assert mon.fallback_active  # latched
    assert "Fallback active since #1" in mon.report()
    mon.reset()
    assert mon.ok and not mon.fallback_active


def test_observe_uses_g_and_features():
    rng = np.random.default_rng(10)
    X = rng.uniform(0, 1, (200, 2))
    U = rng.uniform(-1, 1, (200, 1))
    Xn = np.c_[X[:, 0] + X[:, 1], X[:, 1] + 0.5 * U[:, 0]]  # f(v, u) for output 2
    g = lambda X, U: np.c_[X[:, 0] + X[:, 1], X[:, 1]]  # noqa: E731
    mon = AssumptionMonitor.monotone(None, 0.0, 1e-9, m=2, n=2, g=g, features=[1, 2])
    Z, Y = ddma.transitions_to_data(X, U, Xn, g)
    mon.add(Z[:, [1, 2]], Y)
    assert mon.observe(X[:5], U[:5], Xn[:5]) == []
    bad = mon.observe(X[:1], U[:1], Xn[:1] - [0.0, 0.3])  # slower than ever seen
    assert bad and bad[0].output == 1


def test_max_points_and_reported_gaps_are_real():
    rng = np.random.default_rng(11)
    Z, y = noisy(rng, lambda Z: -Z[:, 0], 600, 1, 0.01)
    mon = AssumptionMonitor.monotone(None, -0.01, 0.01, m=1, n=1, max_points=50)
    viol = mon.check(Z, y)
    assert mon.n_points == 50 and viol
    for v in viol:  # each gap is witnessed by an actual pair of the data
        w = v.witness
        a, b = y[v.index], y[w]
        dz = Z[v.index, 0] - Z[w, 0]
        gap = (a - b if dz <= 0 else b - a) - 0.02
        assert np.isclose(gap, v.gap)


def test_input_validation():
    with pytest.raises(ValueError):
        AssumptionMonitor([[1.0]], [[0.0]])
    with pytest.raises(ValueError):
        AssumptionMonitor([[0.0]], [[1.0]], on_violation="panic")
    with pytest.raises(ValueError):
        AssumptionMonitor([[0.0]], [[1.0]], widen=("slope", 0, 0))
    mon = AssumptionMonitor([[0.0]], [[1.0]])
    with pytest.raises(ValueError):
        mon.check(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(TypeError):
        AssumptionMonitor.for_model(object())
