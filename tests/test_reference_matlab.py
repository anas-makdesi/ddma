"""Agreement with the original MATLAB implementation (Data2Sigma0 +
ApproximationMonotoneSystemData3, A. Makdesi), ported line by line.

The MATLAB code writes each data point into the cell that contains it, takes running
min/max along every axis, then shifts by one cell. For data points that do not lie on
a cell edge this is exactly the fixed-partition map. (For points on an edge, ddma also
uses them for the cell whose corner they touch, which is still sound and slightly tighter.)
"""

import numpy as np
import pytest

from ddma import PartitionedSimulatingMap, RectangularPartition


def matlab_reference(Z, Y, edges, theta):
    shape = tuple(len(e) - 1 for e in edges)
    up = np.full(shape + (Y.shape[1],), np.inf)
    lo = np.full(shape + (Y.shape[1],), -np.inf)
    for z, y in zip(Z, Y):  # Data2Sigma0
        c = tuple(np.searchsorted(e, zi, side="right") - 1 for e, zi in zip(edges, z))
        up[c] = np.minimum(up[c], y + theta)
        lo[c] = np.maximum(lo[c], y - theta)
    for ax in range(len(shape)):  # ApproximationMonotoneSystemData3
        up = np.flip(np.minimum.accumulate(np.flip(up, ax), axis=ax), ax)
        lo = np.maximum.accumulate(lo, axis=ax)
        idx = [slice(None)] * up.ndim
        up = np.roll(up, -1, axis=ax)
        idx[ax] = -1
        up[tuple(idx)] = np.inf
        lo = np.roll(lo, 1, axis=ax)
        idx[ax] = 0
        lo[tuple(idx)] = -np.inf
    return lo, up


@pytest.mark.parametrize("seed", range(40))
def test_matches_original_matlab_algorithm(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 5), rng.integers(1, 3)
    edges = [
        np.unique(np.concatenate(([0.0, 1.0], rng.uniform(0, 1, rng.integers(1, 8)))))
        for _ in range(n)
    ]
    Z = rng.uniform(0, 1, (rng.integers(1, 300), n))
    Y = rng.normal(size=(len(Z), m))
    d = rng.uniform(0, 0.3, m)
    lo, up = matlab_reference(Z, Y, edges, 2 * d)
    s = PartitionedSimulatingMap.from_data(Z, Y, RectangularPartition(edges), -d, d)
    np.testing.assert_array_equal(s.lower, lo)
    np.testing.assert_array_equal(s.upper, up)
