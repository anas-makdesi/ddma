"""Write reference.mat: random problems solved by the Python package, for the MATLAB tests.

Run from the repository root:  python matlab/tests/make_reference.py
"""
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ddma import (
    BoundedDerivativeModel,
    GrowthMap,
    MinimalSimulatingMap,
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    estimate_derivative_bounds,
    reach_avoid,
    safety,
    scenario_confidence,
    simulate,
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

# parity (v0.7): lazy safety, derivative-bound estimation, per-output features, GrowthMap,
# compact maps, map updates, minimal-map extras, partitions, cells_meeting, simulate
par = {}
p_ = out["pend"]
Xpend, Upend, Xnpend = p_["X"], p_["U"], p_["Xn"]
Xp = RectangularPartition.uniform([-1, -1.5], [1, 1.5], [20, 20])
Up = RectangularPartition.uniform([-2], [2], 7)
joint = RectangularPartition.product(Xp, Up)
mdl = BoundedDerivativeModel.from_data(Xpend, Upend, Xnpend, joint, p_["A_lo"], p_["A_hi"],
                                       p_["B_lo"], p_["B_hi"], d_lo=-0.005, d_hi=0.005,
                                       method="growth", local="slab")
lazy = SymbolicAbstraction(Xp, Up, mdl, lazy=True)
safe = lazy.cells_inside([-0.8, -1.2], [0.8, 1.2])
par["lazy_domain"] = safety(lazy, safe).domain_grid()
# per-output features: theta+ depends on (theta, omega) only, omega+ on all
feats = [[0, 1], [0, 1, 2]]
mf = BoundedDerivativeModel.from_data(Xpend, Upend, Xnpend, joint, p_["A_lo"], p_["A_hi"],
                                      p_["B_lo"], p_["B_hi"], d_lo=-0.005, d_hi=0.005,
                                      method="all_matrix", features=feats)
par["feat_lo0"], par["feat_hi0"] = mf.components[0][1].lower, mf.components[0][1].upper
par["feat_lo1"], par["feat_hi1"] = mf.components[1][1].lower, mf.components[1][1].upper
abf = SymbolicAbstraction(Xp, Up, mf)
par["feat_domain"] = safety(abf, safe).domain_grid()
# simulate: deterministic plant, centre inputs
ctrl = safety(SymbolicAbstraction(Xp, Up, mdl), safe)


def plant(x, u, rng=None):
    x = np.asarray(x, float)
    return np.array([x[0] + 0.1 * x[1], x[1] + 0.1 * (-np.sin(x[0]) - 0.2 * x[1] + u[0])])


x0 = Xp.cell_centers(Xp.all_indices()[np.flatnonzero(ctrl.domain)[:1]])[0]
xs, us = simulate(plant, ctrl, x0, 40, pick="center")
par["sim_x0"], par["sim_xs"], par["sim_us"] = x0, xs, us
par["meeting"] = lazy.cells_meeting([-0.2, -1.0], [0.0, 0.0]).reshape(Xp.shape)
# derivative-bound estimation with fixed pairs (indices drawn as the Python function does)
Kd = 4000
Zd = rng.uniform(-1, 1, (Kd, 2))
Yd = np.stack([np.sin(2 * Zd[:, 0]) + 0.5 * Zd[:, 1], Zd[:, 0] * Zd[:, 1]], 1)
Yd = Yd + rng.uniform(-0.01, 0.01, Yd.shape)
r1 = np.random.default_rng(11)
a_lo, a_hi, w = estimate_derivative_bounds(Zd, Yd, 3000, 20000, rng=r1, w_weight=1.0)
r2 = np.random.default_rng(11)
pa1, pb1 = r2.integers(0, Kd, 3000), r2.integers(0, Kd, 3000)
pa2, pb2 = r2.integers(0, Kd, 20000), r2.integers(0, Kd, 20000)
par.update(est_Z=Zd, est_Y=Yd, est_a1=pa1 + 1.0, est_b1=pb1 + 1.0, est_a2=pa2 + 1.0,
           est_b2=pb2 + 1.0, est_a_lo=a_lo, est_a_hi=a_hi, est_w=w,
           conf=scenario_confidence(20000, 1e-6))
# GrowthMap, compact maps, map update, minimal-map extras, data partition
P2 = RectangularPartition.uniform([-1, -1], [1, 1], [9, 7])
gm = GrowthMap.from_data(Zd[:500], Yd[:500, 0], P2, [-2, 0.5], [2, 0.5], -0.01, 0.01)
zq = rng.uniform(-1, 1, (60, 2))
par["gm_lo"], par["gm_hi"] = gm.bounds(zq)
par["gm_flo"], par["gm_fhi"] = gm.f_bounds()
par["gm_zq"], par["gm_edges"] = zq, cell(P2.edges)
Ym = np.stack([Zd[:, 0] + 0.5 * Zd[:, 1] ** 3, np.tanh(Zd[:, 0])], 1)
Ym = Ym + rng.uniform(-.02, .02, (Kd, 2))
s1 = PartitionedSimulatingMap.from_data(Zd[:300], Ym[:300], P2, [-.02, -.02], [.02, .02])
s2 = s1.update(Zd[300:900], Ym[300:900])
sc = PartitionedSimulatingMap.from_data(Zd[:300], Ym[:300], P2, [-.02, -.02], [.02, .02],
                                        compact=True)
par.update(m_Z=Zd[:900], m_Y=Ym[:900], upd_lo=s2.lower, upd_hi=s2.upper,
           cmp_lo=sc.lower.astype(np.float64), cmp_hi=sc.upper.astype(np.float64),
           upd_bfrac=s2.bounded_fraction())
msm = MinimalSimulatingMap(Zd[:200], Ym[:200], [-.02, -.02], [.02, .02])
par["msm_est"] = msm.estimate(zq)
par["msm_ok"] = msm.is_consistent()
bad = MinimalSimulatingMap(Zd[:200], -Ym[:200], [-.02, -.02], [.02, .02])
par["msm_bad_ok"] = bad.is_consistent()
dp = RectangularPartition.from_data(Zd[:7], [-1, -1], [1, 1])
par["dp_edges"] = cell(dp.edges)
# approach 1 as BoundedDerivativeModel computes it (TAC Prop. 8), incl. a zero-bound input
m2 = BoundedDerivativeModel.from_data(Xpend, Upend, Xnpend, joint, p_["A_lo"], p_["A_hi"],
                                      p_["B_lo"], p_["B_hi"], d_lo=-0.005, d_hi=0.005)
idx = joint.all_indices()
clo, chi = joint.cell_bounds(idx)
lo2, hi2 = m2.successor_bounds(idx, clo[:, :2], chi[:, :2], clo[:, 2:], chi[:, 2:])
par["two_lo"] = lo2.reshape(joint.shape + (2,))
par["two_hi"] = hi2.reshape(joint.shape + (2,))
out["par"] = par
print("online:", online["dom0"].mean(), dom1.mean(), online["dom2"].mean(),
      online["refreshed"].mean())

savemat(Path(__file__).with_name("reference.mat"), out, do_compression=True)
print("wrote reference.mat")
