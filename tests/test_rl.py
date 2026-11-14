"""ddma.rl: certified safety filter around Gymnasium environments."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from ddma import (  # noqa: E402
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    safety,
    transitions_to_data,
)
from ddma.rl import CruiseControlEnv, SafetyFilterWrapper  # noqa: E402


@pytest.fixture(scope="module")
def ctrl():
    """Cruise-control safety controller learned from data (coarse grid, fast)."""
    E = CruiseControlEnv
    rng = np.random.default_rng(0)
    lo, hi = np.array([5.0, 0.0]), np.array([60.0, 30.0])
    K = 20_000
    X = rng.uniform(lo, hi, (K, 2))
    U = rng.uniform(-3, 3, (K, 1))
    vl = rng.uniform(*E.VL, K)
    Xn = np.stack([X[:, 0] - E.T * X[:, 1] + E.T * vl,
                   E.speed_update(X[:, 1], U[:, 0]) + rng.uniform(-E.W, E.W, K)], 1)
    A_G = np.array([[1.0, -E.T], [0.0, 0.0]])
    Z, Y = transitions_to_data(X, U, Xn, lambda X, U: X @ A_G.T)
    Xp = RectangularPartition.uniform(lo, hi, [110, 60])
    Up = RectangularPartition.uniform([-3], [3], 24)
    feats = [1, 2]
    part = RectangularPartition.product(Xp, Up).project(feats)
    sigma = PartitionedSimulatingMap.from_data(Z[:, feats], Y, part, [E.T * E.VL[0], -E.W],
                                               [E.T * E.VL[1], E.W])
    a = SymbolicAbstraction(Xp, Up, sigma, g_bounds=affine_bounds(A_G), features=feats)
    c = safety(a, a.cells_inside(E.SAFE_LO, E.SAFE_HI))
    assert c.domain.any()
    return c


def test_random_and_aggressive_agents_never_violate(ctrl):
    env = SafetyFilterWrapper(CruiseControlEnv(horizon=200), ctrl)
    env.action_space.seed(0)
    viol = 0
    for ep in range(12):
        env.reset(seed=ep)
        done = False
        while not done:
            a = env.action_space.sample() if ep % 3 else np.array([3.0 if ep % 2 else -3.0],
                                                                   np.float32)
            _, _, term, trunc, info = env.step(a)
            viol += info["violation"]
            assert not info["outside_domain"]
            done = term or trunc
    assert viol == 0
    assert env.n_filtered > 0


def test_without_filter_the_same_agent_violates():
    env = CruiseControlEnv(horizon=200)
    viol = 0
    for ep in range(6):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.array([3.0], np.float32))
            viol += info["violation"]
            done = term or trunc
    assert viol == 6


def test_safe_actions_pass_unchanged_and_filter_equals_controller(ctrl):
    env = SafetyFilterWrapper(CruiseControlEnv(), ctrl)
    env.reset(seed=3)
    x = env.unwrapped.state.copy()
    ps = ctrl.input_cells(x)
    U = ctrl.abstraction.U
    lo, hi = U.cell_bounds(np.stack(np.unravel_index(ps, U.shape), axis=1))
    u_ok = 0.5 * (lo[0] + hi[0])
    _, _, _, _, info = env.step(u_ok.astype(np.float32))
    assert not info["filtered"] and info["correction"] == 0.0
    x = env.unwrapped.state.copy()
    s, inside = env.safe_action(np.array([3.0]))
    assert inside and np.allclose(s, ctrl.filter(x, np.array([3.0])))


def test_reset_in_domain_penalty_and_outside(ctrl):
    env = CruiseControlEnv(init_lo=(10.0, 2.0), init_hi=(50.0, 28.0))  # partly outside
    w = SafetyFilterWrapper(env, ctrl)
    w.reset(seed=0)
    assert w.in_domain()
    w2 = SafetyFilterWrapper(env, ctrl, reset_in_domain=False, outside="raise")
    for seed in range(50):
        w2.reset(seed=seed)
        if not w2.in_domain():
            with pytest.raises(RuntimeError):
                w2.step(np.array([0.0], np.float32))
            break
    w3 = SafetyFilterWrapper(CruiseControlEnv(), ctrl, penalty=2.0)
    w3.reset(seed=1)
    _, r, _, _, info = w3.step(np.array([3.0], np.float32))
    base = 1.0 - ((w3.unwrapped.state[1] - 22.0) / 10.0) ** 2
    assert np.isclose(r, base - 2.0 * info["correction"])
    with pytest.raises(ValueError):
        SafetyFilterWrapper(CruiseControlEnv(), ctrl, outside="ignore")
