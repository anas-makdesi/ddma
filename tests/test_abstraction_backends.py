"""The compiled abstraction kernel gives exactly the NumPy abstraction, on any number of threads."""

import numpy as np
import pytest

import ddma
from ddma import (
    BoundedDerivativeModel,
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    compiled_backend_available,
    safety,
)

pytestmark = pytest.mark.skipif(not compiled_backend_available(), reason="no compiled kernel")


@pytest.fixture(autouse=True)
def _restore_threads():
    yield
    ddma.set_num_threads(0)


def _same(A, B):
    assert np.array_equal(A.enabled, B.enabled)
    assert A.first.dtype == B.first.dtype
    assert np.array_equal(A.first, B.first)
    assert np.array_equal(A.last, B.last)


def _check(Xp, Up, sm, threads=(1, 2, 3), **kw):
    """Abstractions of every backend and thread count are equal; returns the NumPy one."""
    ref = SymbolicAbstraction(Xp, Up, sm, backend="numpy", **kw)
    for t in threads:
        ddma.set_num_threads(t)
        for chunk in (10**6, 37):  # one call, and many small chunks
            A = SymbolicAbstraction(Xp, Up, sm, backend="cpp", chunk=chunk, **kw)
            _same(ref, A)
            assert A.n_evaluated == ref.n_evaluated
    STATS.append((ref.enabled.mean(), np.mean(ref.last > ref.first)))
    return ref


STATS = []


def _uneven(rng, lo, hi, k):
    """Partition with random (non-uniform) cells."""
    edges = []
    for a, b, kk in zip(lo, hi, k):
        inner = np.sort(rng.uniform(a, b, kk - 1))
        edges.append(np.r_[a, inner, b])
    return RectangularPartition(edges)


def _system(seed, n=2, nu=1, uniform=True, k=(9, 7, 5)):
    rng = np.random.default_rng(seed)
    X_lo, X_hi = -np.ones(n), np.ones(n)
    U_lo, U_hi = -np.ones(nu), np.ones(nu)
    if uniform:
        Xp = RectangularPartition.uniform(X_lo, X_hi, k[:n])
        Up = RectangularPartition.uniform(U_lo, U_hi, (k[-1],) * nu)
    else:
        Xp = _uneven(rng, X_lo, X_hi, k[:n])
        Up = _uneven(rng, U_lo, U_hi, (k[-1],) * nu)
    A = 0.6 * np.eye(n) + 0.15 * rng.random((n, n))
    B = 0.3 * rng.random((n, nu))

    def plant(X, U):
        return X @ A.T + U @ B.T + 0.05 * np.tanh(X)

    N = 400
    X = rng.uniform(X_lo, X_hi, (N, n))
    U = rng.uniform(U_lo, U_hi, (N, nu))
    return rng, Xp, Up, X, U, plant(X, U), A, B


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("compact", [False, True])
def test_map(seed, uniform, compact):
    rng, Xp, Up, X, U, Y, A, B = _system(seed, uniform=uniform)
    joint = RectangularPartition.product(Xp, Up)
    sm = PartitionedSimulatingMap.from_data(np.hstack([X, U]), Y, joint, -0.02, 0.02)
    if compact:
        sm.compact()
    ref = _check(Xp, Up, sm)
    assert 0 < ref.enabled.sum() < ref.enabled.size  # some pairs enabled, some not


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("with_B", [True, False])
def test_map_with_affine_g(seed, with_B):
    """f = small nonlinearity, g = affine part, learned on a subset of features."""
    rng, Xp, Up, X, U, Y, A, B = _system(seed, n=3, uniform=bool(seed % 2))
    g = affine_bounds(A, B if with_B else None, c=0.01 * np.arange(3))
    gx = X @ A.T + (U @ B.T if with_B else 0) + 0.01 * np.arange(3)
    feats = [0, 1, 2] if with_B else [0, 1, 2, 3]
    joint = RectangularPartition.product(Xp, Up).project(feats)
    Z = np.hstack([X, U])[:, feats]
    sm = PartitionedSimulatingMap.from_data(Z, Y - gx, joint, -0.05, 0.05)
    _check(Xp, Up, sm, g_bounds=g, features=feats)


def test_map_with_other_g_and_assumption():
    """A g that is not affine_bounds is evaluated in Python per pair; assumption clipping."""
    rng, Xp, Up, X, U, Y, A, B = _system(5)
    affine = affine_bounds(A, B)

    def g(x_lo, x_hi, u_lo, u_hi):  # any interval function
        lo, hi = affine(x_lo, x_hi, u_lo, u_hi)
        return lo - 0.01, hi + 0.01

    joint = RectangularPartition.product(Xp, Up)
    sm = PartitionedSimulatingMap.from_data(np.hstack([X, U]), Y - X @ A.T - U @ B.T, joint)
    _check(Xp, Up, sm, g_bounds=g)
    e = Xp.edges[1]
    _check(Xp, Up, sm, g_bounds=g, assumption=([-np.inf, e[2]], [np.inf, e[-3]]))
    _check(Xp, Up, sm, assumption=([e[1], -np.inf], [e[-2], np.inf]))


def _model(seed, method, compact=False, g_bounds=None, features=None):
    rng, Xp, Up, X, U, Y, A, B = _system(seed, n=2, uniform=bool(seed % 2), k=(8, 6, 5))
    joint = RectangularPartition.product(Xp, Up)
    J = np.hstack([A, B])
    kw = dict(method=method, compact=compact, features=features)
    if g_bounds is not None:
        kw.update(g=lambda X, U: X @ A.T, g_bounds=g_bounds(A))
        J = np.hstack([np.zeros_like(A), B])
    m = BoundedDerivativeModel.from_data(X, U, Y, joint, J[:, :2] - 0.2, J[:, :2] + 0.2,
                                         J[:, 2:] - 0.01, J[:, 2:] + 0.01, -0.02, 0.02, **kw)
    return Xp, Up, m


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("compact", [False, True])
def test_two_matrix_model(seed, compact):
    Xp, Up, m = _model(seed, "two_matrix", compact)
    ref = _check(Xp, Up, m)
    assert SymbolicAbstraction(Xp, Up, m, backend="cpp")._spec["mode"] == 1
    assert ref.enabled.any()


def test_two_matrix_model_features_and_g():
    Xp, Up, m = _model(1, "two_matrix", features=[[0, 2], [0, 1, 2]])
    _check(Xp, Up, m)
    Xp, Up, m = _model(2, "two_matrix", g_bounds=lambda A: affine_bounds(A))
    _check(Xp, Up, m)

    def other(A):
        a = affine_bounds(A)
        return lambda *b: tuple(v + s for v, s in zip(a(*b), (-1e-3, 1e-3)))

    Xp, Up, m = _model(3, "two_matrix", g_bounds=other)
    assert SymbolicAbstraction(Xp, Up, m)._g_user is not None
    _check(Xp, Up, m)


@pytest.mark.parametrize("method", ["all_matrix", "growth"])
def test_other_models_boxes_from_python(method):
    """Models without the two-matrix structure: boxes from Python, rest in the kernel."""
    Xp, Up, m = _model(0, method)
    assert SymbolicAbstraction(Xp, Up, m, backend="cpp")._spec["mode"] == 2
    _check(Xp, Up, m)


def test_index_dtypes():
    """uint8, uint16 and int32 cell indices (axes with > 256 and > 65536 cells)."""
    for k, dtype in [(300, np.uint16), (70_000, np.int32)]:
        Xp = RectangularPartition.uniform([-1.0], [1.0], [k])
        Up = RectangularPartition.uniform([-1.0], [1.0], [3])
        rng = np.random.default_rng(0)
        X, U = rng.uniform(-1, 1, (2000, 1)), rng.uniform(-1, 1, (2000, 1))
        sm = PartitionedSimulatingMap.from_data(np.hstack([X, U]), 0.9 * X + 0.05 * U,
                                                RectangularPartition.product(Xp, Up), -1e-3, 1e-3)
        ref = _check(Xp, Up, sm, threads=(2,))
        assert ref.first.dtype == dtype


def test_lazy_pairs_and_refresh():
    rng, Xp, Up, X, U, Y, A, B = _system(7, uniform=False)
    joint = RectangularPartition.product(Xp, Up)
    sm = PartitionedSimulatingMap.from_data(np.hstack([X, U]), Y, joint, -0.02, 0.02)
    ref = SymbolicAbstraction(Xp, Up, sm, backend="numpy")
    lz = SymbolicAbstraction(Xp, Up, sm, lazy=True, backend="cpp")
    Q = rng.integers(0, ref.n_states, 500)
    P = rng.integers(0, ref.n_inputs, 500)
    ok, first, last = lz.pair_boxes(Q, P)
    assert np.array_equal(ok, ref.enabled[Q, P])
    assert np.array_equal(first, ref.first[Q, P]) and np.array_equal(last, ref.last[Q, P])
    assert first.dtype == ref.first.dtype and lz.n_evaluated == 500
    assert np.array_equal(safety(lz, np.ones(lz.n_states, bool)).domain,
                          safety(ref, np.ones(ref.n_states, bool)).domain)
    # refresh after a refinement: recomputed rows equal a fresh abstraction
    A = SymbolicAbstraction(Xp, Up, sm, backend="cpp")
    rows = rng.random(A.n_states) < 0.3
    A.refresh(rows)
    _same(ref, A)
    A.first[rows] = 0
    A.enabled[rows] = False
    A.refresh(np.flatnonzero(rows))
    _same(ref, A)


def test_backend_argument():
    rng, Xp, Up, X, U, Y, A, B = _system(0)
    sm = PartitionedSimulatingMap.from_data(np.hstack([X, U]), Y,
                                            RectangularPartition.product(Xp, Up))
    with pytest.raises(ValueError):
        SymbolicAbstraction(Xp, Up, sm, backend="gpu")
    assert SymbolicAbstraction(Xp, Up, sm, backend="numpy")._spec is None
    assert SymbolicAbstraction(Xp, Up, sm)._spec is not None  # auto uses the kernel


def test_cases_are_not_trivial():
    """(Runs last.) The cases above have enabled and disabled pairs and wide successor boxes."""
    if not STATS:
        pytest.skip("run with the other tests")
    en = np.array([s[0] for s in STATS])
    wide = np.array([s[1] for s in STATS])
    assert np.mean((en > 0) & (en < 1)) > 0.8
    assert wide.max() > 0.2


class _BoxModel:
    """Model returning given successor boxes (tests the clipping and cell ranges)."""

    def __init__(self, partition, n, boxes):
        self.partition, self.m, self.boxes = partition, n, boxes

    def successor_bounds(self, joint_idx, x_lo, x_hi, u_lo, u_hi):
        k = len(joint_idx)
        lo, hi = self.boxes
        return lo[:k].copy(), hi[:k].copy()


@pytest.mark.parametrize("seed", range(3))
def test_cells_exact_on_edges(seed):
    """Successor boxes exactly on edges, one ulp away, at the ends of X, NaN and inf."""
    rng = np.random.default_rng(seed)
    clustered = np.r_[-1.0, -1 + 1e-9, -1 + 2e-9, np.sort(rng.uniform(-1, 1, 40)), 1 - 1e-12, 1.0]
    Xp = RectangularPartition([np.unique(clustered), np.linspace(0.0, 3.0, 7)])
    Up = RectangularPartition.uniform([0.0], [1.0], [1])
    joint = RectangularPartition.product(Xp, Up)
    nq = Xp.n_cells
    vals = []
    for e in Xp.edges:
        v = np.r_[e, np.nextafter(e, -np.inf), np.nextafter(e, np.inf)]
        vals.append(rng.choice(v, (nq, 2)))
    lo = np.stack([np.minimum(v[:, 0], v[:, 1]) for v in vals], 1)
    hi = np.stack([np.maximum(v[:, 0], v[:, 1]) for v in vals], 1)
    hi[::7, 0] = lo[::7, 0]  # degenerate boxes
    lo[::11, 1] = np.nan
    hi[::13, 0] = np.inf
    lo[::17] = hi[::17] + 0.1  # empty boxes
    model = _BoxModel(joint, 2, (lo, hi))
    ref = SymbolicAbstraction(Xp, Up, model, backend="numpy")
    A = SymbolicAbstraction(Xp, Up, model, backend="cpp")
    _same(ref, A)
    assert 0.3 < ref.enabled.mean() < 0.95
