"""Compiled growth-cone kernels (ddma._core.growth_vertices / growth_local) vs NumPy."""

import numpy as np
import pytest

from ddma import BoundedDerivativeModel, GrowthMap, RectangularPartition
from ddma._bd_components import GrowthComponent
from ddma.partitioned import compiled_backend_available

pytestmark = pytest.mark.skipif(not compiled_backend_available(),
                                reason="compiled kernel not installed")

ATTRS = ("v_upper", "v_lower", "lower", "upper")


def _assert_close(A, B):
    assert np.array_equal(A == np.inf, B == np.inf)
    assert np.array_equal(A == -np.inf, B == -np.inf)
    f = np.isfinite(A)
    np.testing.assert_allclose(A[f], B[f], rtol=0, atol=1e-12)


def _random_case(rng):
    n = int(rng.integers(1, 4))
    K = [int(rng.integers(1, 7)) for _ in range(n)]
    lo = rng.uniform(-1, 0, n)
    hi = lo + rng.uniform(0.5, 2, n)
    if rng.random() < 0.5:
        P = RectangularPartition.uniform(lo, hi, K)
    else:
        P = RectangularPartition([np.sort(np.r_[a, rng.uniform(a, b, k - 1), b])
                                  for a, b, k in zip(lo, hi, K)])
    gl = rng.uniform(-2, 1, n)
    gh = gl + rng.uniform(0, 2, n)
    if rng.random() < 0.3:
        gl[0] = gh[0] = 0.0
    N = int(rng.integers(40, 200))
    Z = rng.uniform(lo - 0.3, hi + 0.3, (N, n))  # some data outside the box
    y = rng.normal(size=N)
    return P, gl, gh, Z, y


@pytest.mark.parametrize("local", [None, "cell", "slab"])
@pytest.mark.parametrize("final", ["box", "lp"])
def test_cpp_equals_numpy(local, final):
    rng = np.random.default_rng(7)
    for _ in range(15):
        P, gl, gh, Z, y = _random_case(rng)
        a = GrowthComponent(P, gl, gh, -0.1, 0.05, final=final, local=local, backend="numpy")
        b = GrowthComponent(P, gl, gh, -0.1, 0.05, final=final, local=local, backend="cpp")
        a.update(Z, y)
        b.update(Z, y)
        for x in ATTRS + (("loc_upper", "loc_lower") if local else ()):
            _assert_close(getattr(a, x), getattr(b, x))


def test_local_cpp_is_sound_and_matches_brute_force():
    """Slab map = min over points sharing a cell coordinate of the cone's max over the
    cell's vertices; it contains f + d for the true f."""
    rng = np.random.default_rng(3)
    P = RectangularPartition.uniform([0, 0], [1, 1], [6, 5])
    gl, gh = np.array([-1.0, 0.5]), np.array([2.0, 1.0])

    def f(Z):
        return 0.5 * np.sin(2 * Z[:, 0]) + 0.75 * Z[:, 1]

    Z = rng.random((120, 2))
    y = f(Z) + rng.uniform(-0.02, 0.02, 120)
    c = GrowthComponent(P, gl, gh, -0.02, 0.02, local="slab", backend="cpp")
    c.update(Z, y)
    cells = c._cells(Z)
    up = np.full(P.shape, np.inf)
    for q in np.ndindex(*P.shape):
        V = np.array([[P.edges[j][q[j] + b[j]] for j in range(2)] for b in c._corners])
        for k in range(len(y)):
            if np.any(cells[k] == np.array(q)):
                _, phi = c._phi(V - Z[k])
                up[q] = min(up[q], y[k] + c.theta + phi.max())
    _assert_close(c.loc_upper, up)
    Zt = rng.random((5000, 2))
    idx = tuple(c._cells(Zt).T)
    assert np.all(f(Zt) - 0.02 >= c.lower[idx] - 1e-12)
    assert np.all(f(Zt) + 0.02 <= c.upper[idx] + 1e-12)


def test_growth_map_and_model_backends_agree():
    rng = np.random.default_rng(0)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [12, 9])
    Z = rng.uniform(-1, 1, (400, 2))
    y = np.sin(Z[:, 0]) + 0.5 * Z[:, 1]
    a = GrowthMap.from_data(Z, y, P, [-1, -0.5], [1, 0.5], -0.01, 0.01, backend="numpy")
    b = GrowthMap.from_data(Z, y, P, [-1, -0.5], [1, 0.5], -0.01, 0.01, backend="cpp")
    for x in ATTRS:
        _assert_close(getattr(a, x), getattr(b, x))
    assert b.copy().backend == "cpp"

    X = rng.uniform(-1, 1, (300, 1))
    U = rng.uniform(-1, 1, (300, 1))
    Xn = 0.9 * X + 0.2 * np.tanh(U)
    PP = RectangularPartition.uniform([-1, -1], [1, 1], [10, 6])
    kw = dict(A_lo=[[0.85]], A_hi=[[0.95]], B_lo=[[0.0]], B_hi=[[0.25]], method="growth",
              local="slab")
    ma = BoundedDerivativeModel.from_data(X, U, Xn, PP, backend="numpy", **kw)
    mb = BoundedDerivativeModel.from_data(X, U, Xn, PP, backend="cpp", **kw)
    for (_, ca), (_, cb) in zip(ma.components, mb.components):
        for x in ATTRS:
            _assert_close(getattr(ca, x), getattr(cb, x))


def test_backend_argument_checked():
    P = RectangularPartition.uniform([0], [1], [4])
    with pytest.raises(ValueError):
        GrowthComponent(P, [0], [1], 0, 0, backend="gpu")
