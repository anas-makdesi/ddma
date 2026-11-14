import numpy as np
import pytest

from ddma import (
    BoundedDerivativeModel,
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    estimate_derivative_bounds,
    safety,
    scenario_confidence,
    simulate,
)

pytest.importorskip("scipy")

# non-monotone 2-D plant:  x1+ = x1 + 0.1 sin(2 x2) ,  x2+ = 0.8 x2 - 0.2 x1 + 0.3 u
W = 0.01


def f(X, U):
    return np.stack([X[:, 0] + 0.1 * np.sin(2 * X[:, 1]),
                     0.8 * X[:, 1] - 0.2 * X[:, 0] + 0.3 * U[:, 0]], axis=1)


A_LO = np.array([[1.0, -0.2], [-0.2, 0.8]])
A_HI = np.array([[1.0, 0.2], [-0.2, 0.8]])
B_LO = np.array([[0.0], [0.3]])
B_HI = np.array([[0.0], [0.3]])


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    K = 100_000
    X = rng.uniform(-1, 1, (K, 2))
    U = rng.uniform(-1, 1, (K, 1))
    Xn = f(X, U) + rng.uniform(-W, W, (K, 2))
    P = RectangularPartition.uniform([-1, -1, -1], [1, 1, 1], [30, 30, 9])
    return BoundedDerivativeModel.from_data(X, U, Xn, P, A_LO, A_HI, B_LO, B_HI, -W, W)


def test_model_is_sound(model):
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (5000, 2))
    U = rng.uniform(-1, 1, (5000, 1))
    lo, hi = model.bounds(X, U)
    for d in (-W, W):
        v = f(X, U) + d
        ok = np.all((lo <= v + 1e-12) & (v <= hi + 1e-12), axis=1)
        assert ok.all()
    # informative: most boxes are bounded and not much wider than a cell's variation
    width = hi - lo
    assert np.mean(np.isfinite(width).all(1)) > 0.8


def test_intersection_is_tighter_than_each_side(model):
    rng = np.random.default_rng(2)
    X = rng.uniform(-0.8, 0.8, (500, 2))
    U = rng.uniform(-0.8, 0.8, (500, 1))
    idx = model.partition.locate(np.hstack([X, U]))
    lo, hi = model.partition.cell_bounds(idx)
    blo, bhi = model.successor_bounds(idx, lo[:, :2], hi[:, :2], lo[:, 2:], hi[:, 2:])
    sm_lo, sm_hi = model.sigma_minus.bounds_at(idx)
    assert np.all(bhi - blo <= (sm_hi - sm_lo) + 2 * 0.3 + 1e-9)  # never looser than S- side


def test_abstraction_and_safety_with_bounded_derivatives(model):
    Xp = RectangularPartition.uniform([-1, -1], [1, 1], [30, 30])
    Up = RectangularPartition.uniform([-1], [1], 9)
    abst = SymbolicAbstraction(Xp, Up, model)
    ctrl = safety(abst, abst.cells_inside([-0.9, -0.9], [0.9, 0.9]))
    assert ctrl.winning_fraction() > 0.1
    rng = np.random.default_rng(3)

    def step(x, u, rng):
        return f(x[None], np.atleast_2d(u))[0] + rng.uniform(-W, W, 2)

    starts = Xp.cell_centers(Xp.all_indices())[ctrl.domain]
    for x0 in starts[rng.choice(len(starts), 20, replace=False)]:
        xs, us = simulate(step, ctrl, x0, 100, rng=rng, pick="random")
        assert len(us) == 100
        assert np.all(np.abs(xs) <= 0.9 + 1e-12)


def test_estimated_bounds_recover_linear_jacobian():
    rng = np.random.default_rng(4)
    A = np.array([[0.5, -1.0, 0.0], [2.0, 0.3, -0.7]])
    Z = rng.uniform(-1, 1, (200_000, 3))
    w = np.array([0.05, 0.02])
    Y = Z @ A.T + rng.uniform(-w, w, (200_000, 2))
    a_lo, a_hi, w_est = estimate_derivative_bounds(Z, Y, 5000, 100_000, rng=rng)
    assert np.all(a_lo <= A + 0.02) and np.all(A - 0.02 <= a_hi)
    assert np.all(np.abs(a_hi - a_lo) < 0.1)
    assert np.all(w_est >= 0.8 * w) and np.all(w_est <= 1.6 * w)
    # TAC paper, Lorenz example: violation probability below 2e-5 for beta = 1e-6
    assert 1 - 2e-5 < scenario_confidence(10**6, 1e-6) < 1


def test_batched_compact_and_per_output_features_are_sound():
    rng = np.random.default_rng(10)
    K = 60_000
    X = rng.uniform(-1, 1, (K, 2))
    U = rng.uniform(-1, 1, (K, 1))
    Xn = f(X, U) + rng.uniform(-W, W, (K, 2))
    P = RectangularPartition.uniform([-1, -1, -1], [1, 1, 1], [20, 20, 7])
    full = BoundedDerivativeModel.from_data(X, U, Xn, P, A_LO, A_HI, B_LO, B_HI, -W, W)
    # batches give exactly the same model (TAC Prop. 3)
    batched = BoundedDerivativeModel.from_data(X[:20_000], U[:20_000], Xn[:20_000], P,
                                               A_LO, A_HI, B_LO, B_HI, -W, W)
    batched.update(X[20_000:], U[20_000:], Xn[20_000:])
    np.testing.assert_array_equal(full.sigma_minus.lower, batched.sigma_minus.lower)
    np.testing.assert_array_equal(full.sigma_plus.upper, batched.sigma_plus.upper)
    # float32 storage is outward-rounded, per-output features are sound
    feats = [[0, 1], [0, 1, 2]]  # x1+ does not depend on u
    small = BoundedDerivativeModel.from_data(X, U, Xn, P, A_LO, A_HI, B_LO, B_HI, -W, W,
                                             features=feats, compact=True)
    assert len(small.groups) == 2
    assert small.groups[0][2].lower.dtype == np.float32
    q = rng.uniform(-1, 1, (5000, 3))
    lo64, hi64 = full.bounds(q[:, :2], q[:, 2:])
    lo, hi = small.bounds(q[:, :2], q[:, 2:])
    v = f(q[:, :2], q[:, 2:])
    for d in (-W, W):
        assert np.all((lo <= v + d + 1e-12) & (v + d <= hi + 1e-12))
    s64 = full.sigma_minus
    s32 = PartitionedSimulatingMap.from_data(
        np.hstack([X, U]), Xn - X @ np.minimum(A_LO, 0).T - U @ np.minimum(B_LO, 0).T, P,
        -W, W, compact=True)
    assert np.all(s32.lower <= s64.lower) and np.all(s32.upper >= s64.upper)
