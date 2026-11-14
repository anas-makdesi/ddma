"""Write reference.mat: random problems solved by the Python package, for the MATLAB tests.

Run from the repository root:  python matlab/tests/make_reference.py
"""
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ddma import (
    BoundedDerivativeModel,
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
# bounded-derivative models (thesis Ch. 3), one entry per (case, method, local)
from ddma._bd_components import GrowthComponent, SignMatrixComponent  # noqa: E402

bd = []
for k in range(6):
    n, m = int(rng.integers(1, 4)), int(rng.integers(1, 3))
    if k % 2:
        P = RectangularPartition.uniform(-np.ones(n), np.ones(n), rng.integers(2, 6, n))
    else:
        P = RectangularPartition([np.unique(np.r_[-1, 1, rng.uniform(-1, 1, rng.integers(1, 5))])
                                  for _ in range(n)])
    K = int(rng.integers(30, 200))
    Z = rng.uniform(-1.2, 1.2, (K, n))
    g_lo = rng.uniform(-2, 1, (m, n))
    g_hi = g_lo + rng.uniform(0, 2, (m, n))
    if k == 2:
        g_lo[0, 0] = g_hi[0, 0] = 0.0
    Y = rng.normal(size=(K, m))
    d_lo, d_hi = -0.05 * np.ones(m), 0.1 * np.ones(m)
    for method in ("two_matrix", "all_matrix", "lin_mono", "growth", "lin_growth"):
        locals_ = ("", "cell", "slab") if "growth" in method else ("",)
        for loc in locals_:
            lo, hi = [], []
            for i in range(m):
                final = "lp" if method.startswith("lin_") else "box"
                if "growth" in method:
                    c = GrowthComponent(P, g_lo[i], g_hi[i], d_lo[i], d_hi[i], final=final,
                                        local=loc or None)
                else:
                    c = SignMatrixComponent(P, g_lo[i], g_hi[i], d_lo[i], d_hi[i],
                                            final=final,
                                            which="two" if method == "two_matrix" else "all")
                c.update(Z, Y[:, i])
                lo.append(c.lower)
                hi.append(c.upper)
            bd.append({"Z": Z, "Y": Y, "edges": cell(P.edges), "g_lo": g_lo, "g_hi": g_hi,
                       "d_lo": d_lo, "d_hi": d_hi, "method": method, "local": loc,
                       "lower": np.stack(lo, -1), "upper": np.stack(hi, -1)})
ba = np.empty((1, len(bd)), dtype=object)
for i, f in enumerate(bd):
    ba[0, i] = f
out["bd"] = ba

# bounded-derivative model + abstraction + safety: a damped pendulum
#   th+ = th + 0.1 om,  om+ = om + 0.1 (-sin th - 0.2 om + u)  (non-monotone in th)
Xp = RectangularPartition.uniform([-1, -1.5], [1, 1.5], [20, 20])
Up = RectangularPartition.uniform([-2], [2], 7)
K = 20000
X = np.column_stack([rng.uniform(-1, 1, K), rng.uniform(-1.5, 1.5, K)])
U = rng.uniform(-2, 2, (K, 1))
Xn = np.column_stack([X[:, 0] + 0.1 * X[:, 1],
                      X[:, 1] + 0.1 * (-np.sin(X[:, 0]) - 0.2 * X[:, 1] + U[:, 0])])
Xn = Xn + rng.uniform(-0.005, 0.005, Xn.shape)
A_lo = np.array([[1.0, 0.1], [-0.1, 0.98]])
A_hi = np.array([[1.0, 0.1], [-0.05, 0.98]])
B_lo = np.array([[0.0], [0.1]])
B_hi = B_lo.copy()
joint = RectangularPartition.product(Xp, Up)
pend = {"Xedges": cell(Xp.edges), "Uedges": cell(Up.edges), "X": X, "U": U, "Xn": Xn,
        "A_lo": A_lo, "A_hi": A_hi, "B_lo": B_lo, "B_hi": B_hi}
for method in ("all_matrix", "growth"):
    mdl = BoundedDerivativeModel.from_data(X, U, Xn, joint, A_lo, A_hi, B_lo, B_hi,
                                           d_lo=-0.005, d_hi=0.005, method=method,
                                           local="slab" if method == "growth" else None)
    ab = SymbolicAbstraction(Xp, Up, mdl)
    ctrl = safety(ab, ab.cells_inside([-0.8, -1.2], [0.8, 1.2]))
    pend[method + "_domain"] = ctrl.domain_grid()
out["pend"] = pend

# online refinement: fit, exact update, windowed update, refresh, incremental safety
K = 20330
X = np.column_stack([rng.uniform(-1, 1, K), rng.uniform(-1.5, 1.5, K)])
U = rng.uniform(-2, 2, (K, 1))
Xn = np.column_stack([X[:, 0] + 0.1 * X[:, 1],
                      X[:, 1] + 0.1 * (-np.sin(X[:, 0]) - 0.2 * X[:, 1] + U[:, 0])])
Xn = Xn + rng.uniform(-0.005, 0.005, Xn.shape)
A_lo = np.array([[1.0, 0.1], [-0.3, 0.9]])
A_hi = np.array([[1.0, 0.1], [0.1, 1.05]])
B_lo = np.array([[0.0], [0.05]])
B_hi = np.array([[0.0], [0.15]])
online = {"X": X, "U": U, "Xn": Xn, "A_lo": A_lo, "A_hi": A_hi, "B_lo": B_lo, "B_hi": B_hi,
          "Xedges": cell(Xp.edges), "Uedges": cell(Up.edges), "window": 1}


def comp_bounds(mdl):
    return (np.stack([c.lower for _, c in mdl.components], -1),
            np.stack([c.upper for _, c in mdl.components], -1))


mdl = BoundedDerivativeModel.from_data(X[:300], U[:300], Xn[:300], joint, A_lo, A_hi, B_lo,
                                       B_hi, d_lo=-0.005, d_hi=0.005, method="growth",
                                       local="slab")
ab = SymbolicAbstraction(Xp, Up, mdl)
safe = ab.cells_inside([-0.8, -1.2], [0.8, 1.2])
online["safe"] = safe.reshape(Xp.shape)
online["dom0"] = safety(ab, safe).domain_grid()
mdl.update(X[300:20300], U[300:20300], Xn[300:20300])
online["lo1"], online["hi1"] = comp_bounds(mdl)
ab.refresh(np.ones(ab.n_states, bool))
dom1 = safety(ab, safe, known_winning=online["dom0"].ravel()).domain
online["dom1"] = dom1.reshape(Xp.shape)
mdl.update_local(X[20300:20330], U[20300:20330], Xn[20300:20330], 1)
online["lo2"], online["hi2"] = comp_bounds(mdl)
online["refreshed"] = ab.refresh().reshape(Xp.shape)
online["first2"] = ab.first.reshape(Xp.shape + Up.shape + (2,))
online["last2"] = ab.last.reshape(Xp.shape + Up.shape + (2,))
online["enabled2"] = ab.enabled.reshape(Xp.shape + Up.shape)
online["dom2"] = safety(ab, safe, known_winning=dom1).domain_grid()
out["online"] = online
print("online:", online["dom0"].mean(), dom1.mean(), online["dom2"].mean(),
      online["refreshed"].mean())

savemat(Path(__file__).with_name("reference.mat"), out, do_compression=True)
print("wrote reference.mat")
