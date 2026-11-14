"""The five bounded-derivative approaches of the thesis (Chapter 3)."""

import itertools

import numpy as np
import pytest

import ddma
from ddma import BoundedDerivativeModel, RectangularPartition, SymbolicAbstraction
from ddma._bd_components import GrowthComponent, SignMatrixComponent, sign_patterns
from ddma._box_lp import max_min_affine
from ddma.bounded_derivatives import METHODS

TOL = 1e-9


def make_function(rng, n):
    """Random smooth f: R^n -> R with known (conservative) derivative bounds."""
    c = rng.uniform(0.2, 1.0, n)
    w = rng.uniform(0.5, 2.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    a = rng.uniform(-0.6, 0.6, n)
    k = 0.3

    def f(Z):
        return (c * np.sin(w * Z + ph)).sum(1) + Z @ a + k * np.sin(Z.sum(1))

    rad = np.abs(c * w) + k
    return f, a - rad, a + rad


def data(rng, f, n, K, lo=-2.0, hi=2.0, d=0.1, margin=0.0):
    Z = rng.uniform(lo - margin, hi + margin, (K, n))
    return Z, f(Z) + rng.uniform(-d, d, K)


def check_sound(comp, f, rng, d=0.1, n_test=20000):
    P = comp.partition
    Zt = rng.uniform(P.lo, P.hi, (n_test, P.ndim))
    idx = tuple(P.locate(Zt).T)
    y = f(Zt)
    assert np.all(comp.lower[idx] <= y - d + TOL)
    assert np.all(comp.upper[idx] >= y + d - TOL)


# ---------------------------------------------------------------------------- patterns
def test_sign_patterns():
    a, s = sign_patterns([-1, 0.5, 0], [2, 3, 0], "all")
    assert a.shape == (4, 3)  # third coordinate: one choice only
    assert set(map(tuple, a[:, :2])) == {(-1, 0), (-1, 3), (2, 0), (2, 3)}
    assert np.all(s[:, 2] == 0)
    a2, _ = sign_patterns([-1, 0.5], [2, 3], "two")
    np.testing.assert_array_equal(a2, [[-1, 0], [2, 3]])


# ---------------------------------------------------------------------------- soundness
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("which", ["box", "lp"])
def test_components_sound(n, which):
    rng = np.random.default_rng(n + 10 * (which == "lp"))
    f, glo, ghi = make_function(rng, n)
    Z, y = data(rng, f, n, 3000, margin=0.3)  # some points outside the box
    P = RectangularPartition.uniform([-2] * n, [2] * n, [12, 9, 6][:n] if n > 1 else 15)
    for comp in (SignMatrixComponent(P, glo, ghi, -0.1, 0.1, final=which),
                 GrowthComponent(P, glo, ghi, -0.1, 0.1, final=which)):
        comp.update(Z, y)
        check_sound(comp, f, rng)
        assert np.all(comp.lower <= comp.upper + TOL)


def test_lp_tighter_than_box():
    rng = np.random.default_rng(3)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 2000)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [10, 10])
    for cls in (SignMatrixComponent, GrowthComponent):
        b = cls(P, glo, ghi, -0.1, 0.1, final="box")
        lp = cls(P, glo, ghi, -0.1, 0.1, final="lp")
        b.update(Z, y)
        lp.update(Z, y)
        fin = np.isfinite(b.upper)
        assert np.all(lp.upper[fin] <= b.upper[fin] + TOL)
        assert np.all(lp.lower[fin] >= b.lower[fin] - TOL)
        assert (b.upper - b.lower)[fin].mean() > (lp.upper - lp.lower)[fin].mean()


# ---------------------------------------------------------------------------- growth cones
def brute_vertex_values(P, Z, y, glo, ghi, theta):
    V = np.stack(np.meshgrid(*P.edges, indexing="ij"), axis=-1)
    dz = V[..., None, :] - Z  # (*vshape, K, n)
    up = (y + theta + np.maximum(ghi * dz, glo * dz).sum(-1)).min(-1)
    lo = (y - theta + np.minimum(ghi * dz, glo * dz).sum(-1)).max(-1)
    return lo, up


@pytest.mark.parametrize("n", [1, 2, 3])
def test_growth_vertex_values_equal_brute_force(n):
    """Eqs. 3.20-3.21: vertex values are the set-membership bounds min_k / max_k."""
    rng = np.random.default_rng(20 + n)
    f, glo, ghi = make_function(rng, n)
    glo[0] = 0.2  # also a coordinate with derivative bounds of one sign
    ghi[0] = max(ghi[0], 0.5)
    Z, y = data(rng, f, n, 150, margin=0.5)
    edges = [np.sort(np.concatenate(([-2, 2], rng.uniform(-2, 2, k)))) for k in (7, 5, 4)[:n]]
    P = RectangularPartition(edges)
    comp = GrowthComponent(P, glo, ghi, -0.1, 0.1)
    lo, up = comp.vertex_values(Z, y)
    blo, bup = brute_vertex_values(P, Z, y, glo, ghi, 0.2)
    np.testing.assert_allclose(up, bup, rtol=0, atol=1e-12)
    np.testing.assert_allclose(lo, blo, rtol=0, atol=1e-12)


def test_growth_box_step_equals_definition():
    """Eqs. 3.26-3.27: min over cell vertices v of max over cell vertices v* of the cone."""
    rng = np.random.default_rng(5)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 300)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [6, 5])
    comp = GrowthComponent(P, glo, ghi, -0.1, 0.1)
    comp.update(Z, y)
    blo, bup = brute_vertex_values(P, Z, y, glo, ghi, 0.2)
    for q in itertools.product(*map(range, P.shape)):
        verts = [tuple(np.array(q) + b) for b in itertools.product((0, 1), repeat=2)]
        coords = {v: np.array([P.edges[j][v[j]] for j in range(2)]) for v in verts}
        U = min(max(bup[v] + np.maximum(ghi * (coords[w] - coords[v]),
                                        glo * (coords[w] - coords[v])).sum() for w in verts)
                for v in verts)
        L = max(min(blo[v] + np.minimum(ghi * (coords[w] - coords[v]),
                                        glo * (coords[w] - coords[v])).sum() for w in verts)
                for v in verts)
        assert comp.upper[q] == pytest.approx(U, abs=1e-12)
        assert comp.lower[q] == pytest.approx(L, abs=1e-12)


def test_bounds_on_cells_without_data():
    """With data in one corner, growth cones and all sign matrices bound every cell;
    the two matrices of approach 1 do not."""
    rng = np.random.default_rng(6)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 200, lo=-2, hi=-1)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [8, 8])
    g = GrowthComponent(P, glo, ghi, -0.1, 0.1)
    s = SignMatrixComponent(P, glo, ghi, -0.1, 0.1)
    two = SignMatrixComponent(P, glo, ghi, -0.1, 0.1, which="two")
    for c in (g, s, two):
        c.update(Z, y)
    for c in (g, s):
        assert np.all(np.isfinite(c.upper)) and np.all(np.isfinite(c.lower))
        check_sound(c, f, rng)
    assert not np.all(np.isfinite(two.upper))


# ---------------------------------------------------------------------------- updates
@pytest.mark.parametrize("cls,final", [(SignMatrixComponent, "box"), (SignMatrixComponent, "lp"),
                                       (GrowthComponent, "box"), (GrowthComponent, "lp")])
def test_update_equals_refit(cls, final):
    rng = np.random.default_rng(7)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 2000)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [9, 7])
    a = cls(P, glo, ghi, -0.1, 0.1, final=final)
    a.update(Z[:700], y[:700])
    a.update(Z[700:], y[700:])
    b = cls(P, glo, ghi, -0.1, 0.1, final=final)
    b.update(Z, y)
    tol = 1e-12 if final == "box" else 1e-6  # LP: certified bounds, solver-dependent
    np.testing.assert_allclose(a.upper, b.upper, atol=tol, rtol=0)
    np.testing.assert_allclose(a.lower, b.lower, atol=tol, rtol=0)


@pytest.mark.parametrize("final", ["box", "lp"])
def test_local_update(final):
    """Prop. 13: windowed update tightens vertex values on the window only; sound."""
    rng = np.random.default_rng(8)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 400)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [10, 10])
    comp = GrowthComponent(P, glo, ghi, -0.1, 0.1, final=final)
    comp.update(Z[:200], y[:200])
    old_up = comp.v_upper.copy()
    Zn, yn = Z[200:220], y[200:220]
    comp.update_local(Zn, yn, window=2)
    check_sound(comp, f, rng)
    # expected vertex values: old, tightened by each new point on its window
    exp = old_up.copy()
    cells = P.locate(Zn)
    _, bup = brute_vertex_values(P, Zn, yn, glo, ghi, 0.2)
    for k, c in enumerate(cells):
        a, b = np.maximum(c - 2, 0), np.minimum(c + 2, 9) + 2
        _, bk = brute_vertex_values(P, Zn[k:k + 1], yn[k:k + 1], glo, ghi, 0.2)
        exp[a[0]:b[0], a[1]:b[1]] = np.minimum(exp[a[0]:b[0], a[1]:b[1]], bk[a[0]:b[0], a[1]:b[1]])
    np.testing.assert_allclose(comp.v_upper, exp, atol=1e-12, rtol=0)
    # not tighter than the global update
    full = GrowthComponent(P, glo, ghi, -0.1, 0.1, final=final)
    full.update(np.vstack([Z[:200], Zn]), np.concatenate([y[:200], yn]))
    assert np.all(comp.upper >= full.upper - 1e-6)
    # a window covering the whole grid gives the global update
    big = GrowthComponent(P, glo, ghi, -0.1, 0.1, final=final)
    big.update(Z[:200], y[:200])
    big.update_local(Zn, yn, window=20)
    np.testing.assert_allclose(big.upper, full.upper, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------- LP helper
def test_max_min_affine_matches_reference_lp():
    from scipy.optimize import linprog

    rng = np.random.default_rng(9)
    N, L, n = 60, 5, 3
    off = rng.normal(size=(N, L))
    off[rng.random((N, L)) < 0.2] = np.inf
    off[0] = np.inf
    slopes = rng.normal(size=(L, n))
    lo = rng.uniform(-1, 0, (N, n))
    hi = lo + rng.uniform(0.1, 1, (N, n))
    got = max_min_affine(off, slopes, lo, hi, batch=17)
    assert got[0] == np.inf
    for k in range(1, N):
        fin = np.isfinite(off[k])
        A = np.hstack([-slopes[fin], np.ones((fin.sum(), 1))])
        res = linprog(np.r_[np.zeros(n), -1], A_ub=A, b_ub=off[k, fin],
                      bounds=list(zip(lo[k], hi[k])) + [(None, None)], method="highs")
        assert got[k] >= -res.fun - 1e-9  # sound
        assert got[k] <= -res.fun + 1e-6  # tight


# ---------------------------------------------------------------------------- model level
def test_two_patterns_equal_approach_one():
    """The sign-matrix component restricted to A-, A+ is approach 1 ("two_matrix")."""
    rng = np.random.default_rng(11)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 3000)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [12, 10])
    comp = SignMatrixComponent(P, glo, ghi, -0.1, 0.1, which="two")
    comp.update(Z, y)
    M = BoundedDerivativeModel.from_data(Z[:, :1], Z[:, 1:], y[:, None], P, glo[None, :1],
                                         ghi[None, :1], glo[None, 1:], ghi[None, 1:],
                                         -0.1, 0.1)
    idx = P.all_indices()
    lo, hi = P.cell_bounds(idx)
    mlo, mhi = M.successor_bounds(idx, lo[:, :1], hi[:, :1], lo[:, 1:], hi[:, 1:])
    fin = np.isfinite(mhi[:, 0])
    np.testing.assert_array_equal(fin, np.isfinite(comp.upper.ravel()))
    np.testing.assert_allclose(comp.upper.ravel()[fin], mhi[fin, 0], atol=1e-12)
    np.testing.assert_allclose(comp.lower.ravel()[fin], mlo[fin, 0], atol=1e-12)


def test_two_matrix_with_zero_bounds_equals_two_patterns():
    """A coordinate with derivative bounds [0, 0] is constant for approach 1 too (sign 0),
    so "two_matrix" equals the two-pattern component also then (tighter than treating it
    as increasing)."""
    rng = np.random.default_rng(12)
    Z = rng.uniform(-1, 1, (800, 2))
    y = np.sin(2 * Z[:, 0]) + rng.uniform(-0.05, 0.05, 800)
    glo, ghi = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    P = RectangularPartition.uniform([-1.5, -1], [1.5, 1], [10, 4])  # data not at the ends
    comp = SignMatrixComponent(P, glo, ghi, -0.05, 0.05, which="two")
    comp.update(Z, y)
    M = BoundedDerivativeModel.from_data(Z[:, :1], Z[:, 1:], y[:, None], P, glo[None, :1],
                                         ghi[None, :1], glo[None, 1:], ghi[None, 1:],
                                         -0.05, 0.05)
    idx = P.all_indices()
    lo, hi = P.cell_bounds(idx)
    mlo, mhi = M.successor_bounds(idx, lo[:, :1], hi[:, :1], lo[:, 1:], hi[:, 1:])
    assert np.all(np.isfinite(mhi)) and np.all(np.isfinite(mlo))
    np.testing.assert_allclose(comp.upper.ravel(), mhi[:, 0], atol=1e-12)
    np.testing.assert_allclose(comp.lower.ravel(), mlo[:, 0], atol=1e-12)


def lorenz_like(X, U):
    x, y = X[:, 0], X[:, 1]
    return np.stack([x + 0.1 * np.sin(y), y + 0.1 * (np.cos(x) * y) + 0.1 * U[:, 0]], axis=1)


def lorenz_like_bounds():
    # d/dx, d/dy of each output on [-1, 1]^2; d/du
    A_lo = np.array([[1.0, -0.1], [-0.1 * np.sin(1), 1 + 0.1 * np.cos(1)]])
    A_hi = np.array([[1.0, 0.1], [0.1 * np.sin(1), 1.1]])
    return A_lo, A_hi, np.array([[0.0], [0.1]]), np.array([[0.0], [0.1]])


@pytest.mark.parametrize("method", METHODS)
def test_model_methods_sound_with_known_part_and_features(method):
    rng = np.random.default_rng(12)
    K = 4000
    X = rng.uniform(-1, 1, (K, 2))
    U = rng.uniform(-1, 1, (K, 1))
    W = rng.uniform(-0.01, 0.01, (K, 2))
    g = lambda X, U: 0.05 * X  # noqa: E731  known part
    g_b = ddma.affine_bounds(0.05 * np.eye(2), np.zeros((2, 1)))
    Xn = lorenz_like(X, U) + g(X, U) + W
    A_lo, A_hi, B_lo, B_hi = lorenz_like_bounds()
    P = RectangularPartition.uniform([-1, -1, -1], [1, 1, 1], [8, 8, 4])
    M = BoundedDerivativeModel.from_data(X, U, Xn, P, A_lo, A_hi, B_lo, B_hi, -0.01, 0.01,
                                         g=g, g_bounds=g_b, method=method,
                                         features=[[0, 1], [0, 1, 2]])
    Xt = rng.uniform(-1, 1, (20000, 2))
    Ut = rng.uniform(-1, 1, (20000, 1))
    lo, hi = M.bounds(Xt, Ut)
    truth = lorenz_like(Xt, Ut) + g(Xt, Ut)
    assert np.all(lo <= truth - 0.01 + TOL) and np.all(hi >= truth + 0.01 - TOL)


@pytest.mark.parametrize("method", ["all_matrix", "growth"])
def test_model_plugs_into_abstraction(method):
    """x+ = 0.9 x + 0.3 sin(x) + u: safety on [-1, 1] with |u| <= 0.2."""
    rng = np.random.default_rng(13)
    X = rng.uniform(-1, 1, (5000, 1))
    U = rng.uniform(-0.2, 0.2, (5000, 1))
    Xn = 0.9 * X + 0.3 * np.sin(X) + U
    Px = RectangularPartition.uniform([-1], [1], [40])
    Pu = RectangularPartition.uniform([-0.2], [0.2], [5])
    P = RectangularPartition.product(Px, Pu)
    M = BoundedDerivativeModel.from_data(X, U, Xn, P, [[0.9 - 0.3]], [[1.2]], [[1.0]], [[1.0]],
                                         method=method)
    abst = SymbolicAbstraction(Px, Pu, M)
    ctrl = ddma.safety(abst, np.ones(Px.n_cells, bool))
    assert ctrl.domain.sum() > 0.5 * Px.n_cells
    lazy = SymbolicAbstraction(Px, Pu, M, lazy=True)
    np.testing.assert_array_equal(ddma.safety(lazy, np.ones(Px.n_cells, bool)).domain,
                                  ctrl.domain)


def test_model_errors():
    P = RectangularPartition.uniform([-1, -1], [1, 1], [4, 4])
    X, U = np.zeros((3, 1)), np.zeros((3, 1))
    with pytest.raises(ValueError):
        BoundedDerivativeModel.from_data(X, U, X, P, [[0]], [[1]], [[0]], [[1]], method="nope")
    M = BoundedDerivativeModel.from_data(X, U, X, P, [[0]], [[1]], [[0]], [[1]],
                                         method="all_matrix")
    with pytest.raises(ValueError):
        M.update_local(X, U, X, window=1)


# ---------------------------------------------------------------------------- local map (Prop. 7)
def eq11_bounds(P, Z, y, glo, ghi, theta):
    """min over data of the max of the cone over each cell's vertices (and the lower bound)."""
    up = np.full(P.shape, np.inf)
    lo = np.full(P.shape, -np.inf)
    for q in itertools.product(*map(range, P.shape)):
        a, b = P.cell_bounds(np.array([q]))
        V = np.array(list(itertools.product(*zip(a[0], b[0]))))
        dz = V[:, None, :] - Z[None]
        up[q] = (y + theta + np.maximum(ghi * dz, glo * dz).sum(-1)).max(0).min()
        lo[q] = (y - theta + np.minimum(ghi * dz, glo * dz).sum(-1)).min(0).max()
    return lo, up


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("final", ["box", "lp"])
def test_local_slab_never_looser_than_eq11(n, final):
    rng = np.random.default_rng(30 + n)
    for _ in range(6):
        f, glo, ghi = make_function(rng, n)
        Z, y = data(rng, f, n, int(rng.integers(3, 40)))
        P = RectangularPartition.uniform([-2] * n, [2] * n, [int(rng.integers(2, 6))] * n)
        lo11, up11 = eq11_bounds(P, Z, y, glo, ghi, 0.2)
        plain = GrowthComponent(P, glo, ghi, -0.1, 0.1, final=final)
        cell = GrowthComponent(P, glo, ghi, -0.1, 0.1, final=final, local="cell")
        slab = GrowthComponent(P, glo, ghi, -0.1, 0.1, final=final, local="slab")
        for c in (plain, cell, slab):
            c.update(Z, y)
            check_sound(c, f, rng, n_test=3000)
        tol = 1e-9 if final == "box" else 1e-6
        assert np.all(slab.upper <= up11 + tol) and np.all(slab.lower >= lo11 - tol)
        e = 1e-12
        assert np.all(slab.upper <= cell.upper + e) and np.all(cell.upper <= plain.upper + e)
        assert np.all(slab.lower >= cell.lower - e) and np.all(cell.lower >= plain.lower - e)


def test_prop7_counterexample_cell_local():
    """The cell-local intersection alone can be looser than eq. (11) (thesis/draft
    Prop. 7 as first stated); the slab version is not."""
    P = RectangularPartition.uniform([0, 0], [2, 2], [2, 2])
    Z, y = np.array([[0.375, 1.375]]), np.array([-0.75])
    g = np.ones(2)
    cell = GrowthComponent(P, -g, g, 0, 0, final="lp", local="cell")
    slab = GrowthComponent(P, -g, g, 0, 0, final="lp", local="slab")
    cell.update(Z, y)
    slab.update(Z, y)
    assert cell.upper[0, 0] == pytest.approx(1.625, abs=1e-6)
    assert slab.upper[0, 0] == pytest.approx(1.25, abs=1e-9)


@pytest.mark.parametrize("local", ["cell", "slab"])
def test_local_map_with_windowed_update(local):
    rng = np.random.default_rng(33)
    f, glo, ghi = make_function(rng, 2)
    Z, y = data(rng, f, 2, 300)
    P = RectangularPartition.uniform([-2, -2], [2, 2], [8, 8])
    a = GrowthComponent(P, glo, ghi, -0.1, 0.1, local=local)
    a.update(Z[:200], y[:200])
    a.update_local(Z[200:], y[200:], window=1)
    check_sound(a, f, rng)
    b = GrowthComponent(P, glo, ghi, -0.1, 0.1, local=local)
    b.update(Z[:200], y[:200])
    b.update_local(Z[200:], y[200:], window=10)  # covers the grid: same as update
    full = GrowthComponent(P, glo, ghi, -0.1, 0.1, local=local)
    full.update(Z, y)
    np.testing.assert_allclose(b.upper, full.upper, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b.lower, full.lower, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------- online loop
def online_setup(lazy=False, local=None):
    rng = np.random.default_rng(40)
    step = lambda X, U: np.stack([X[:, 0] + 0.1 * X[:, 1],  # noqa: E731
                                  X[:, 1] + 0.1 * (2 * np.sin(X[:, 0]) + U[:, 0])], 1)
    Px = RectangularPartition.uniform([-1, -1], [1, 1], [24, 24])
    Pu = RectangularPartition.uniform([-2], [2], [5])
    P = RectangularPartition.product(Px, Pu)
    A_lo = np.array([[1, 0.1], [0.2 * np.cos(1), 1]])
    A_hi = np.array([[1, 0.1], [0.2, 1]])
    B = np.array([[0.0], [0.1]])
    X = rng.uniform(-1, 1, (300, 2))
    U = rng.uniform(-2, 2, (300, 1))
    M = BoundedDerivativeModel.from_data(X, U, step(X, U), P, A_lo, A_hi, B, B,
                                         method="growth", local=local)
    abst = SymbolicAbstraction(Px, Pu, M, lazy=lazy)
    return rng, step, Px, Pu, P, M, abst


@pytest.mark.parametrize("backend", ["cpp", "numpy", "lazy"])
def test_incremental_safety_equals_recomputation(backend):
    lazy = backend == "lazy"
    rng, step, Px, Pu, P, M, abst = online_setup(lazy=lazy)
    be = "numpy" if lazy else backend
    if be == "cpp" and not ddma.compiled_backend_available():
        pytest.skip("compiled kernel not installed")
    safe = np.ones(Px.n_cells, bool)
    old = ddma.safety(abst, safe, backend=be).domain
    for _ in range(5):  # online: a few new points, local update, refresh, incremental safety
        X = rng.uniform(-1, 1, (20, 2))
        U = rng.uniform(-2, 2, (20, 1))
        M.update_local(X, U, step(X, U), window=3)
        changed = abst.refresh()
        assert changed.any() and not M.changed_boxes
        new = ddma.safety(abst, safe, backend=be, known_winning=old).domain
        ref_abst = SymbolicAbstraction(Px, Pu, M, lazy=lazy)
        ref = ddma.safety(ref_abst, safe, backend=be).domain
        np.testing.assert_array_equal(new, ref)
        assert np.all(new[old])  # the winning set only grows
        old = new
    assert old.sum() > 0


def test_refresh_equals_rebuild():
    rng, step, Px, Pu, P, M, abst = online_setup(local="slab")
    X = rng.uniform(-1, 1, (10, 2))
    U = rng.uniform(-2, 2, (10, 1))
    M.update_local(X, U, step(X, U), window=2)
    mask = abst.refresh()
    assert 0 < mask.sum() < Px.n_cells
    ref = SymbolicAbstraction(Px, Pu, M)
    np.testing.assert_array_equal(abst.enabled, ref.enabled)
    np.testing.assert_array_equal(abst.first[abst.enabled], ref.first[ref.enabled])
    np.testing.assert_array_equal(abst.last[abst.enabled], ref.last[ref.enabled])


def test_known_winning_is_checked():
    rng, step, Px, Pu, P, M, abst = online_setup()
    safe = np.ones(Px.n_cells, bool)
    bad = ~ddma.safety(abst, safe).domain  # losing cells are not controlled invariant
    with pytest.raises(ValueError):
        ddma.safety(abst, safe, known_winning=bad)
    with pytest.raises(ValueError):
        ddma.safety(abst, np.zeros(Px.n_cells, bool), known_winning=safe)
