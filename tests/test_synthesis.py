import numpy as np
import pytest

from ddma import (
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    reach_avoid,
    safety,
    simulate,
    transitions_to_data,
)

# A 2-D monotone plant with an unknown part f and a known part g:
#   x1+ = x1 + 0.1 * x2                         (known, g)
#   x2+ = 0.9 * x2 + 0.3 * tanh(u) + w          (unknown, monotone in (x2, u))
A_G = np.array([[1.0, 0.1], [0.0, 0.0]])
W = 0.02


def f(x, u):
    return np.stack([np.zeros(len(x)), 0.9 * x[:, 1] + 0.3 * np.tanh(u[:, 0])], axis=1)


def g(x, u):
    return x @ A_G.T


def step(x, u, rng):
    x, u = np.atleast_2d(x), np.atleast_2d(u)
    return (f(x, u) + g(x, u) + np.array([0.0, rng.uniform(-W, W)]))[0]


def build(rng, nx=(40, 40), nu=11, K=50_000):
    Xp = RectangularPartition.uniform([-1, -1], [1, 1], nx)
    Up = RectangularPartition.uniform([-1], [1], nu)
    X = rng.uniform(-1, 1, size=(K, 2))
    U = rng.uniform(-1, 1, size=(K, 1))
    Xn = f(X, U) + g(X, U) + np.stack([np.zeros(K), rng.uniform(-W, W, K)], axis=1)
    Zd, Y = transitions_to_data(X, U, Xn, g)
    feats = [1, 2]  # f depends on (x2, u) only
    joint = RectangularPartition.product(Xp, Up)
    sigma = PartitionedSimulatingMap.from_data(
        Zd[:, feats], Y, joint.project(feats), [0, -W], [0, W]
    )
    return SymbolicAbstraction(Xp, Up, sigma, g_bounds=affine_bounds(A_G), features=feats)


@pytest.fixture(scope="module")
def abstraction():
    return build(np.random.default_rng(0))


def naive_safety(a, safe):
    W_ = safe.copy()
    while True:
        keep = W_.copy()
        for q in np.flatnonzero(W_):
            ok = any(
                a.enabled[q, p] and np.all(W_[a.successors(q, p)]) for p in range(a.n_inputs)
            )
            keep[q] = ok
        if np.array_equal(keep, W_):
            return W_
        W_ = keep


def naive_reach(a, target, avoid):
    R = target & ~avoid
    val = np.where(R, 0.0, np.inf)
    k = 0
    while True:
        k += 1
        new = np.zeros_like(R)
        for q in np.flatnonzero(~R & ~avoid):
            new[q] = any(
                a.enabled[q, p] and np.all(R[a.successors(q, p)]) for p in range(a.n_inputs)
            )
        if not new.any():
            return R, val
        val[new] = k
        R |= new


def test_affine_bounds_contain_samples():
    rng = np.random.default_rng(1)
    A, B, c = rng.normal(size=(3, 2)), rng.normal(size=(3, 1)), rng.normal(size=3)
    gb = affine_bounds(A, B, c)
    lo, hi = gb(np.array([[-1, 0]]), np.array([[0, 2]]), np.array([[0.5]]), np.array([[1.0]]))
    x = rng.uniform([-1, 0], [0, 2], size=(1000, 2))
    u = rng.uniform(0.5, 1.0, size=(1000, 1))
    v = x @ A.T + u @ B.T + c
    assert np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12)


def test_successors_contain_true_next_states(abstraction):
    rng = np.random.default_rng(2)
    a = abstraction
    for _ in range(2000):
        x = rng.uniform(-1, 1, 2)
        u = rng.uniform(-1, 1, 1)
        q = np.ravel_multi_index(tuple(a.X.locate(x[None])[0]), a.X.shape)
        p = np.ravel_multi_index(tuple(a.U.locate(u[None])[0]), a.U.shape)
        if not a.enabled[q, p]:
            continue
        xn = step(x, u, rng)
        assert a.X.contains(xn[None])[0]
        qn = np.ravel_multi_index(tuple(a.X.locate(xn[None])[0]), a.X.shape)
        assert qn in a.successors(q, p)


def test_safety_matches_naive_fixed_point(abstraction):
    safe = abstraction.cells_inside([-0.8, -0.8], [0.8, 0.8])
    ctrl = safety(abstraction, safe)
    np.testing.assert_array_equal(ctrl.domain, naive_safety(abstraction, safe))
    assert 0 < ctrl.winning_fraction() < 1


def test_reach_avoid_matches_naive_fixed_point(abstraction):
    target = abstraction.cells_inside([0.4, -0.3], [0.8, 0.3])
    avoid = abstraction.cells_meeting([-0.2, -1.0], [0.0, 0.2])
    ctrl = reach_avoid(abstraction, target, avoid)
    R, val = naive_reach(abstraction, target, avoid)
    np.testing.assert_array_equal(ctrl.domain, R)
    np.testing.assert_array_equal(ctrl.value, val)


def test_closed_loop_never_leaves_safe_set(abstraction):
    rng = np.random.default_rng(3)
    lo, hi = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    ctrl = safety(abstraction, abstraction.cells_inside(lo, hi))
    starts = abstraction.X.cell_centers(abstraction.X.all_indices())[ctrl.domain]
    for x0 in starts[rng.choice(len(starts), size=30, replace=False)]:
        xs, us = simulate(step, ctrl, x0, 200, rng=rng, pick="random")
        assert len(us) == 200  # never left the controller domain
        assert np.all(xs >= lo) and np.all(xs <= hi)


def test_closed_loop_reaches_target(abstraction):
    rng = np.random.default_rng(4)
    target = abstraction.cells_inside([0.4, -0.3], [0.8, 0.3])
    avoid = abstraction.cells_meeting([-0.2, -1.0], [0.0, 0.2])
    ctrl = reach_avoid(abstraction, target, avoid)
    starts = abstraction.X.cell_centers(abstraction.X.all_indices())[ctrl.domain & ~target]
    A = abstraction.X
    for x0 in starts[rng.choice(len(starts), size=min(20, len(starts)), replace=False)]:
        q0 = np.ravel_multi_index(tuple(A.locate(x0[None])[0]), A.shape)
        xs, _ = simulate(step, ctrl, x0, 500, rng=rng, stop_in=target)
        q = np.ravel_multi_index(tuple(A.locate(xs[-1][None])[0]), A.shape)
        assert target[q]
        assert len(xs) - 1 <= ctrl.value[q0]
        for x in xs:
            qq = np.ravel_multi_index(tuple(A.locate(x[None])[0]), A.shape)
            assert not avoid[qq]


def test_feature_partition_must_match():
    Xp = RectangularPartition.uniform([0], [1], 4)
    Up = RectangularPartition.uniform([0], [1], 3)
    wrong = PartitionedSimulatingMap.from_data(
        [[0.5, 0.5]], [[0.5]], RectangularPartition.uniform([0, 0], [1, 1], 5)
    )
    with pytest.raises(ValueError):
        SymbolicAbstraction(Xp, Up, wrong)


def test_safety_filter(abstraction):
    rng = np.random.default_rng(5)
    lo, hi = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    ctrl = safety(abstraction, abstraction.cells_inside(lo, hi))
    U = abstraction.U
    starts = abstraction.X.cell_centers(abstraction.X.all_indices())[ctrl.domain]
    for x in starts[rng.choice(len(starts), size=200, replace=False)]:
        allowed = ctrl.input_cells(x)
        u_des = rng.uniform(-1.5, 1.5, size=1)
        u = ctrl.filter(x, u_des)
        p = np.ravel_multi_index(tuple(U.locate(u[None])[0]), U.shape)
        assert p in allowed
        if U.contains(u_des[None])[0]:
            p_des = np.ravel_multi_index(tuple(U.locate(u_des[None])[0]), U.shape)
            if p_des in allowed:
                np.testing.assert_array_equal(u, u_des)
    # an aggressive nominal controller stays safe under the filter
    for x0 in starts[rng.choice(len(starts), size=20, replace=False)]:
        xs, us = simulate(step, ctrl, x0, 200, rng=rng, nominal=lambda x: np.array([1.0]))
        assert len(us) == 200
        assert np.all(xs >= lo) and np.all(xs <= hi)
    assert ctrl.filter(np.array([0.95, 0.95]), [0.0]) is None


def test_lazy_abstraction_gives_the_same_controller(abstraction):
    a = abstraction
    lazy = SymbolicAbstraction(a.X, a.U, a.sim_map, g_bounds=affine_bounds(A_G),
                               features=a.features, lazy=True)
    assert lazy.first is None and lazy.enabled is None
    q, p = 123, 4
    np.testing.assert_array_equal(lazy.successors(q, p), a.successors(q, p))
    safe = a.cells_inside([-0.9, -0.9], [0.9, 0.9])
    eager = safety(a, safe)
    lazy.n_evaluated = 0
    c = safety(lazy, safe)
    np.testing.assert_array_equal(c.domain, eager.domain)
    assert c.allowed is None
    assert 0 < lazy.n_evaluated < a.n_states * a.n_inputs
    for qq in np.flatnonzero(eager.domain)[::97]:
        np.testing.assert_array_equal(c.allowed_row(qq), eager.allowed[qq])
    x = a.X.cell_centers(a.X.all_indices()[[np.flatnonzero(eager.domain)[5]]])[0]
    assert c(x) is not None and c.filter(x, [0.0]) is not None
    with pytest.raises(ValueError, match="eager"):
        reach_avoid(lazy, safe)
