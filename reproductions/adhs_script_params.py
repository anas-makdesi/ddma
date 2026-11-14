"""ADHS Fig. 3 with the parameters of the original experiment script: disturbance +-0.05,
data inputs in [-4, 4], inner partition points linspace(-80,-20,29), linspace(15,25,29),
linspace(-21.5,-18.5,19), linspace(-3,3,13), and the script's criterion R_F: state-input
pairs whose cell or successor touches an outer (unbounded) cell are skipped; successor
volume = span of the successor cells."""
import numpy as np
from common import save

from ddma import PartitionedSimulatingMap, RectangularPartition

T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
W = 0.05
XD_LO, XD_HI = np.array([-100.0, 10.0, -22.0]), np.array([0.0, 30.0, -18.0])


def plant(X, U):
    a = U[:, 0] - (F0 + F1 * X[:, 1] + F2 * X[:, 1] ** 2) / M
    return np.stack([X[:, 0] + (X[:, 1] + X[:, 2]) * T, X[:, 1] + a * T, X[:, 2]], 1)


def edges(lo, hi, pts):
    return np.concatenate(([lo], pts, [hi]))


Xe = [edges(-100, 0, np.linspace(-80, -20, 29)), edges(10, 30, np.linspace(15, 25, 29)),
      edges(-22, -18, np.linspace(-21.5, -18.5, 19))]
Ue = [edges(-4, 4, np.linspace(-3, 3, 13))]
joint = RectangularPartition(Xe + Ue)


def r_f(s):
    sh = joint.shape
    inner = [np.arange(1, k - 1) for k in sh]
    idx = np.stack(np.meshgrid(*inner, indexing="ij"), -1).reshape(-1, 4)
    lo, hi = s.bounds_at(idx)
    Xp = RectangularPartition(Xe)
    ok = np.all(np.isfinite(lo) & np.isfinite(hi), 1)
    lo = np.clip(lo, Xp.lo, Xp.hi)
    hi = np.clip(hi, Xp.lo, Xp.hi)
    first, last = Xp.box_to_cells(lo, hi)
    nx = np.array(Xp.shape)
    ok &= np.all((first > 0) & (first < nx - 1) & (last > 0) & (last < nx - 1), 1)
    span = np.ones(len(idx))
    for j, e in enumerate(Xe):
        span *= e[last[:, j] + 1] - e[first[:, j]]
    return float(span[ok].mean() / (2 * W) ** 3), float(ok.mean())


rng = np.random.default_rng(0)
res = {"n_data": [], "R_F": []}
for N in [k * 10**5 for k in range(1, 9)]:
    vals = []
    for _ in range(5):
        X = rng.uniform(XD_LO, XD_HI, (N, 3))
        U = rng.uniform(-4, 4, (N, 1))
        Y = plant(X, U) + rng.uniform(-W, W, (N, 3))
        s = PartitionedSimulatingMap.from_data(np.hstack([X, U]), Y, joint, -W, W)
        vals.append(r_f(s)[0])
    res["n_data"].append(N)
    res["R_F"].append(float(np.mean(vals)))
    print(f"N={N:>7,}  R_F = {np.mean(vals):.0f}")
save("adhs_script_params", res)
