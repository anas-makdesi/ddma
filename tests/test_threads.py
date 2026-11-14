"""The learning kernels give bit-identical results for any number of threads and any way
of writing the data into the grid (direct, bucketed, private copies, owned parts)."""

import itertools
import os

import numpy as np
import pytest

from ddma import GrowthMap, PartitionedSimulatingMap, RectangularPartition
from ddma.partitioned import _core
from ddma.threads import get_num_threads, openmp_available, set_num_threads

pytestmark = pytest.mark.skipif(_core is None, reason="compiled kernel not installed")

MODES = {"auto": 0, "direct": 1, "bucketed": 2, "private": 3, "owned": 4}
THREADS = sorted({1, 2, os.cpu_count() or 1})


@pytest.fixture(autouse=True)
def restore_settings():
    yield
    set_num_threads(None)
    _core._set_scatter_mode(0)
    _core._set_chunk_entries(1 << 20)


def bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def all_settings(chunks=(1 << 20, 997)):
    """(threads, mode, values per round): small rounds exercise the multi-round paths."""
    return itertools.product(THREADS, MODES.values(), chunks)


def check_all_settings(compute):
    set_num_threads(1)
    _core._set_scatter_mode(MODES["direct"])
    ref = compute()
    for threads, mode, chunk in all_settings():
        set_num_threads(threads)
        _core._set_scatter_mode(mode)
        _core._set_chunk_entries(chunk)
        out = compute()
        for a, b in zip(out, ref):
            assert bitwise_equal(a, b), (threads, mode, chunk)
    return ref


def grid(rng, n, k, uniform):
    if uniform:
        return [np.linspace(-1, 1, k + 1) for _ in range(n)]
    return [np.sort(rng.uniform(-1, 1, k + 1)) + 1e-6 * np.arange(k + 1) for _ in range(n)]


def test_thread_setting():
    set_num_threads(1)
    assert get_num_threads() == 1
    set_num_threads(None)
    assert get_num_threads() >= 1
    if openmp_available():
        set_num_threads(3)
        assert get_num_threads() == 3
    with pytest.raises(ValueError):
        set_num_threads(-1)


# Grids big enough to be cut into several blocks / parts (> 256 KB).
@pytest.mark.parametrize("n,k,K,m", [(1, 40, 3000, 1), (2, 150, 60_000, 2), (3, 40, 80_000, 3),
                                     (4, 14, 50_000, 1)])
@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("single", [False, True])
def test_fit_sigma_same_bits(n, k, K, m, uniform, single):
    rng = np.random.default_rng(n * 100 + k)
    edges = grid(rng, n, k, uniform)
    Z = rng.uniform(-1.1, 1.1, (K, n))  # some points outside the box
    Y = rng.normal(size=(K, m))
    signs = rng.integers(-1, 2, (m, n)).astype(np.int8)
    theta = rng.uniform(0, 0.1, m)
    check_all_settings(lambda: _core.fit_sigma(Z, Y, edges, theta, signs, single))


@pytest.mark.parametrize("n,k,K", [(1, 30, 2000), (2, 200, 40_000), (3, 25, 30_000),
                                   (4, 9, 20_000)])
@pytest.mark.parametrize("uniform", [True, False])
def test_growth_kernels_same_bits(n, k, K, uniform):
    rng = np.random.default_rng(n * 10 + k)
    edges = grid(rng, n, k, uniform)
    Z = rng.uniform(-1.2, 1.2, (K, n))
    y = np.sin(Z.sum(1)) + rng.normal(scale=0.01, size=K)
    gl = rng.uniform(-2, 0.5, n)
    gh = gl + rng.uniform(0, 2, n)
    check_all_settings(lambda: _core.growth_vertices(Z, y, edges, gl, gh, 0.01))
    for slab in (False, True):
        check_all_settings(lambda s=slab: _core.growth_local(Z, y, edges, gl, gh, 0.01, s))


def test_points_in_one_cell_and_nan_values():
    """All points in one cell (one thread gets all the work) and NaN outputs (ignored)."""
    rng = np.random.default_rng(3)
    edges = [np.linspace(0, 1, 301)] * 2
    Z = np.full((20_000, 2), 0.5) + rng.uniform(0, 1e-4, (20_000, 2))
    Y = rng.normal(size=(20_000, 1))
    Y[::7] = np.nan
    check_all_settings(lambda: _core.fit_sigma(Z, Y, edges, np.array([0.1]),
                                               np.array([[1, -1]], np.int8), False))
    check_all_settings(lambda: _core.growth_vertices(Z, Y[:, 0], edges, np.array([-1.0, 0]),
                                                     np.array([1.0, 2.0]), 0.0))


def test_public_classes_same_bits():
    """Through the public classes: monotone map (float64 and compact float32), GrowthMap."""
    rng = np.random.default_rng(7)
    P = RectangularPartition.uniform([0, 0, 0], [1, 1, 1], [60, 50, 40])
    Z = rng.uniform(0, 1, (100_000, 3))
    Y = np.stack([Z[:, 0] + Z[:, 1] ** 2, np.sin(3 * Z[:, 2]) - Z[:, 0]], 1)

    def monotone(compact):
        mp = PartitionedSimulatingMap.from_data(Z, Y, P, -0.01, 0.01,
                                                signs=[[1, 1, 0], [-1, 0, 1]], compact=compact)
        return mp.lower, mp.upper

    check_all_settings(lambda: monotone(False))
    check_all_settings(lambda: monotone(True))

    def growth(local):
        g = GrowthMap.from_data(Z, Y[:, 1], P, [-1.5, -0.1, -3.1], [0.1, 0.1, 3.1],
                                -0.01, 0.01, local=local)
        return g.lower, g.upper, g.v_lower, g.v_upper

    for local in (None, "cell", "slab"):
        check_all_settings(lambda loc=local: growth(loc))


def test_same_as_numpy_backend_with_threads():
    """With all threads, the compiled kernel still equals the NumPy implementation."""
    rng = np.random.default_rng(11)
    P = RectangularPartition.uniform([0, 0], [1, 1], [120, 90])
    Z = rng.uniform(-0.1, 1.1, (30_000, 2))
    Y = (Z[:, :1] - 0.5 * Z[:, 1:]) + rng.uniform(-0.01, 0.01, (30_000, 1))
    set_num_threads(None)
    a = PartitionedSimulatingMap.from_data(Z, Y, P, -0.01, 0.01, signs=[[1, -1]], backend="cpp")
    b = PartitionedSimulatingMap.from_data(Z, Y, P, -0.01, 0.01, signs=[[1, -1]],
                                           backend="numpy")
    assert bitwise_equal(a.lower, b.lower) and bitwise_equal(a.upper, b.upper)
