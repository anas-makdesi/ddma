"""Write reference.mat: random problems solved by the Python package, for the MATLAB tests.

Run from the repository root:  python matlab/tests/make_reference.py
"""
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ddma import (
    MinimalSimulatingMap,
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    reach_avoid,
    safety,
)


def cell(arrs):
    c = np.empty((1, len(arrs)), dtype=object)
    for i, a in enumerate(arrs):
        c[0, i] = np.asarray(a, float).reshape(-1, 1)
    return c


rng = np.random.default_rng(7)
out = {}
fits = []
for k in range(12):
    n, m = int(rng.integers(1, 4)), int(rng.integers(1, 3))
    if k % 2:
        P = RectangularPartition.uniform(-np.ones(n), np.ones(n), rng.integers(2, 7, n))
    else:
        P = RectangularPartition([np.unique(np.r_[-1, 1, rng.uniform(-1, 1, rng.integers(1, 6))])
                                  for _ in range(n)])
    K = int(rng.integers(5, 400))
    Z = rng.uniform(-1.2, 1.2, (K, n))
    Z[: K // 4] = rng.choice(np.linspace(-1, 1, 5), (K // 4, n))
    Y = rng.normal(size=(K, m))
    signs = rng.choice([-1, 0, 1], size=(m, n)) if k % 3 else np.ones((m, n), int)
    d_lo, d_hi = -0.1 * np.ones(m), 0.2 * np.ones(m)
    s = PartitionedSimulatingMap.from_data(Z, Y, P, d_lo, d_hi, signs=signs)
    zq = rng.uniform(-1, 1, (50, n))
    lo, hi = MinimalSimulatingMap(Z, Y, d_lo, d_hi, signs=signs).bounds(zq)
    fits.append({"Z": Z, "Y": Y, "edges": cell(P.edges), "d_lo": d_lo, "d_hi": d_hi,
                 "signs": signs.astype(float), "lower": s.lower, "upper": s.upper,
                 "zq": zq, "msm_lo": lo, "msm_hi": hi})
fa = np.empty((1, len(fits)), dtype=object)
for i, f in enumerate(fits):
    fa[0, i] = f
out["fits"] = fa

# abstraction + synthesis: x1+ = x1 + 0.1 x2 (known), x2+ = 0.9 x2 + 0.3 tanh(u) + w
A_G = np.array([[1.0, 0.1], [0.0, 0.0]])
Xp = RectangularPartition.uniform([-1, -1], [1, 1], [12, 10])
Up = RectangularPartition.uniform([-1], [1], 7)
K = 20000
X = rng.uniform(-1, 1, (K, 2))
U = rng.uniform(-1, 1, (K, 1))
Y = np.stack([np.zeros(K), 0.9 * X[:, 1] + 0.3 * np.tanh(U[:, 0]) + rng.uniform(-.02, .02, K)], 1)
feats = [1, 2]
joint = RectangularPartition.product(Xp, Up)
s = PartitionedSimulatingMap.from_data(np.hstack([X, U])[:, feats], Y, joint.project(feats),
                                       [0, -.02], [0, .02])
assumption = (np.array([-np.inf, -0.8]), np.array([np.inf, 0.8]))
a = SymbolicAbstraction(Xp, Up, s, g_bounds=affine_bounds(A_G), features=feats,
                        assumption=assumption)
sh = Xp.shape + Up.shape
ctrl = safety(a, a.cells_inside([-0.6, -1], [0.6, 1]))
ra = reach_avoid(a, a.cells_inside([0.3, -0.4], [0.6, 0.4]), a.cells_meeting([-0.2, -1], [0, 0]))
out["abs"] = {
    "Xedges": cell(Xp.edges), "Uedges": cell(Up.edges), "fedges": cell(s.partition.edges),
    "lower": s.lower, "upper": s.upper, "d_lo": [0, -.02], "d_hi": [0, .02],
    "A_G": A_G, "features": np.array(feats, float) + 1, "assumption": np.vstack(assumption),
    "first": a.first.reshape(sh + (2,)), "last": a.last.reshape(sh + (2,)),
    "enabled": a.enabled.reshape(sh),
    "safety_domain": ctrl.domain_grid(),
    "reach_domain": ra.domain_grid(), "reach_value": ra.value.reshape(Xp.shape),
    "X": X, "U": U, "Y": Y,
}
savemat(Path(__file__).with_name("reference.mat"), out, do_compression=True)
print("wrote reference.mat")
