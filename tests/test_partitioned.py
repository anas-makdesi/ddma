import numpy as np
import pytest

from ddma import (
    MinimalSimulatingMap,
    PartitionedSimulatingMap,
    RectangularPartition,
    compiled_backend_available,
)


def random_partition(rng, n, lo=-1.0, hi=1.0, max_cells=6):
    edges = []
    for _ in range(n):
        k = rng.integers(1, max_cells + 1)
        inner = np.sort(rng.choice(np.linspace(lo, hi, 41)[1:-1], size=k - 1, replace=False))
        edges.append(np.concatenate(([lo], inner, [hi])))
    return RectangularPartition(edges)


def brute_force(S: MinimalSimulatingMap, P: RectangularPartition):
    """sigma(q) per the theorem, using the pointwise map at the order-corners of each cell."""
    idx = P.all_indices()
    clo, chi = P.cell_bounds(idx)
    lower = np.empty((len(idx), S.m))
    upper = np.empty((len(idx), S.m))
    for i in range(S.m):
        s = S.signs[i]
        # corners that are smallest/largest in the order of output i
        small = np.where(s >= 0, clo, chi)
        large = np.where(s >= 0, chi, clo)
        lower[:, i] = S.bounds(small)[0][:, i]
        upper[:, i] = S.bounds(large)[1][:, i]
    return lower.reshape(P.shape + (S.m,)), upper.reshape(P.shape + (S.m,))


BACKENDS = ["numpy"] + (["cpp"] if compiled_backend_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", range(12))
def test_sweep_matches_brute_force(seed, backend):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 4), rng.integers(1, 3)
    P = random_partition(rng, n)
    K = 50
    # include points on cell edges and outside the box
    Z = rng.choice(np.linspace(-1.2, 1.2, 25), size=(K, n))
    Y = rng.normal(size=(K, m))
    signs = rng.choice([-1, 0, 1], size=(m, n)) if seed % 2 else None
    S = MinimalSimulatingMap(Z, Y, -0.1, 0.2, signs=signs)
    sigma = PartitionedSimulatingMap.from_data(Z, Y, P, -0.1, 0.2, signs=signs, backend=backend)
    lo, hi = brute_force(S, P)
    np.testing.assert_array_equal(sigma.lower, lo)
    np.testing.assert_array_equal(sigma.upper, hi)


def test_data_induced_partition_recovers_minimal_map():
    rng = np.random.default_rng(5)
    Z = rng.uniform(0, 1, size=(30, 2))
    Y = (Z @ np.array([[1.0], [2.0]])) + rng.uniform(-0.1, 0.1, size=(30, 1))
    P = RectangularPartition.from_data(Z, [0, 0], [1, 1])
    sigma = PartitionedSimulatingMap.from_data(Z, Y, P, -0.1, 0.1)
    S = MinimalSimulatingMap(Z, Y, -0.1, 0.1)
    centers = P.cell_centers(P.all_indices())  # interior points, off the data grid
    lo, hi = sigma.bounds(centers)
    slo, shi = S.bounds(centers)
    np.testing.assert_array_equal(lo, slo)
    np.testing.assert_array_equal(hi, shi)


def test_partitioned_map_is_sound_and_contains_minimal_map():
    rng = np.random.default_rng(6)
    def f(z):
        return np.stack([z[:, 0] + z[:, 1] ** 3, np.tanh(3 * z[:, 0])], axis=1)

    Z = rng.uniform(-1, 1, size=(2000, 2))
    Y = f(Z) + rng.uniform(-0.05, 0.05, size=(2000, 2))
    P = RectangularPartition.uniform([-1, -1], [1, 1], [20, 15])
    sigma = PartitionedSimulatingMap.from_data(Z, Y, P, -0.05, 0.05)
    zq = rng.uniform(-1, 1, size=(1000, 2))
    lo, hi = sigma.bounds(zq)
    for d in (-0.05, 0.05):
        assert np.all(lo <= f(zq) + d + 1e-12) and np.all(f(zq) + d <= hi + 1e-12)
    slo, shi = MinimalSimulatingMap(Z, Y, -0.05, 0.05).bounds(zq)
    assert np.all(lo <= slo) and np.all(shi <= hi)
    assert not sigma.is_empty().any()


def test_update_equals_recomputation():
    rng = np.random.default_rng(7)
    Z = rng.uniform(-1, 1, size=(400, 3))
    Y = np.sum(Z, axis=1, keepdims=True) + rng.uniform(-0.1, 0.1, size=(400, 1))
    P = RectangularPartition.uniform(-np.ones(3), np.ones(3), 6)
    full = PartitionedSimulatingMap.from_data(Z, Y, P, -0.1, 0.1)
    inc = PartitionedSimulatingMap.from_data(Z[:150], Y[:150], P, -0.1, 0.1)
    inc = inc.update(Z[150:], Y[150:])
    np.testing.assert_array_equal(full.lower, inc.lower)
    np.testing.assert_array_equal(full.upper, inc.upper)


def test_partition_locate_and_errors():
    P = RectangularPartition([[0, 1, 2], [0, 0.5, 1]])
    np.testing.assert_array_equal(P.locate([[0, 0], [1, 0.5], [2, 1], [0.99, 0.49]]),
                                  [[0, 0], [1, 1], [1, 1], [0, 0]])
    with pytest.raises(ValueError):
        P.locate([[2.1, 0]])
    with pytest.raises(ValueError):
        RectangularPartition([[0, 0, 1]])
    first, last = P.box_to_cells([[0.2, 0.1]], [[1.0, 0.4]])
    np.testing.assert_array_equal(first, [[0, 0]])
    np.testing.assert_array_equal(last, [[1, 0]])


@pytest.mark.skipif(not compiled_backend_available(), reason="compiled kernel not built")
@pytest.mark.parametrize("seed", range(20))
def test_compiled_and_numpy_backends_agree(seed):
    rng = np.random.default_rng(100 + seed)
    n, m = int(rng.integers(1, 5)), int(rng.integers(1, 4))
    uniform = seed % 2 == 0
    if uniform:
        P = RectangularPartition.uniform(-np.ones(n), np.ones(n), rng.integers(1, 9, n))
    else:
        P = random_partition(rng, n)
    K = int(rng.integers(1, 3000))
    Z = rng.uniform(-1.3, 1.3, (K, n))
    Z[: K // 3] = rng.choice(np.linspace(-1, 1, 9), size=(K // 3, n))  # points on edges
    Y = rng.normal(size=(K, m))
    signs = rng.choice([-1, 0, 1], size=(m, n))
    a = PartitionedSimulatingMap.from_data(Z, Y, P, -0.05, 0.1, signs=signs, backend="cpp")
    b = PartitionedSimulatingMap.from_data(Z, Y, P, -0.05, 0.1, signs=signs, backend="numpy")
    np.testing.assert_array_equal(a.lower, b.lower)
    np.testing.assert_array_equal(a.upper, b.upper)


def test_uniform_search_matches_searchsorted():
    rng = np.random.default_rng(0)
    for k in (1, 3, 7, 100, 999):
        P = RectangularPartition.uniform([-np.pi], [np.e], k)
        e = P.edges[0]
        v = np.concatenate([rng.uniform(-5, 5, 5000), e, np.nextafter(e, -9), np.nextafter(e, 9)])
        for side in ("left", "right"):
            np.testing.assert_array_equal(P.search(0, v, side), np.searchsorted(e, v, side=side))


@pytest.mark.parametrize("seed", range(10))
def test_compact_float32_is_outward_and_backend_independent(seed):
    rng = np.random.default_rng(300 + seed)
    n = int(rng.integers(1, 4))
    P = RectangularPartition.uniform(-np.ones(n), np.ones(n), rng.integers(2, 9, n))
    Z = rng.uniform(-1.2, 1.2, (400, n))
    Y = rng.normal(size=(400, 2)) * 333.3
    sg = rng.choice([-1, 0, 1], (2, n))
    exact = PartitionedSimulatingMap.from_data(Z, Y, P, -0.1, 0.3, signs=sg, backend="numpy")
    for b in BACKENDS:
        c = PartitionedSimulatingMap.from_data(Z, Y, P, -0.1, 0.3, signs=sg, backend=b,
                                               compact=True)
        assert c.lower.dtype == np.float32 and c.upper.dtype == np.float32
        assert np.all(c.lower <= exact.lower) and np.all(c.upper >= exact.upper)
        fin = np.isfinite(exact.upper)  # rounding moves a bound by at most ~1 float32 ulp
        assert np.all(c.upper[fin] - exact.upper[fin] <= 1e-6 * np.abs(exact.upper[fin]) + 1e-30)
        fin = np.isfinite(exact.lower)
        assert np.all(exact.lower[fin] - c.lower[fin] <= 1e-6 * np.abs(exact.lower[fin]) + 1e-30)
