"""Write reference_monitor.mat: assumption-monitor runs in Python, for the MATLAB tests.

Run from the repository root:  python matlab/tests/make_reference_monitor.py
Indices are stored 0-based as in Python; the MATLAB test adds 1 where needed.
"""
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ddma import (
    AssumptionMonitor,
    BoundedDerivativeModel,
    PartitionedSimulatingMap,
    RectangularPartition,
)


def viol_array(viols):
    """rows: index, output, kind (0 data, 1 model), gap, witness"""
    if not viols:
        return np.zeros((0, 5))
    return np.array([[v.index, v.output, v.kind == "model", v.gap, v.witness] for v in viols],
                    float)


def need(mon, prefix, out):
    nd = mon.needed()
    out[prefix + "need_w"] = nd["width"][None, :]
    out[prefix + "need_lo"] = nd["slope_lo"]
    out[prefix + "need_hi"] = nd["slope_hi"]


def main():
    rng = np.random.default_rng(2026)
    out = {}
    # A: Lipschitz bound too small (explicit slope intervals)
    Z = rng.uniform(0, 1, (300, 2))
    y = 3.0 * Z[:, 0] + 0.5 * Z[:, 1] + rng.uniform(-0.02, 0.02, 300)
    mon = AssumptionMonitor([[-1.0, -1.0]], [[1.0, 1.0]], -0.02, 0.02)
    mon.add(Z[:60], y[:60])
    v = mon.check(Z[60:], y[60:])
    out.update(a_Z=Z, a_y=y[:, None], a_viol=viol_array(v))
    need(mon, "a_", out)

    # B: wrong sign, monotone map on a grid (model bounds checked too)
    Z = rng.uniform(0, 1, (400, 2))
    y = Z[:, 0] - 2.0 * Z[:, 1] + rng.uniform(-0.01, 0.01, 400)
    P = RectangularPartition.uniform([0, 0], [1, 1], [10, 10])
    psm = PartitionedSimulatingMap.from_data(Z[:150], y[:150], P, -0.01, 0.01)
    mon = AssumptionMonitor.for_model(psm, Z[:150], y[:150])
    v = mon.check(Z[150:], y[150:])
    out.update(b_Z=Z, b_y=y[:, None], b_viol=viol_array(v))
    need(mon, "b_", out)

    # C: two outputs, disturbance grows mid-run, widen policy with margin
    Z = rng.uniform(0, 1, (300, 2))
    F = np.c_[Z[:, 0] + Z[:, 1], Z[:, 0] - Z[:, 1]]
    W = np.r_[np.full(150, 0.01), np.full(150, 0.08)][:, None]
    Y = F + rng.uniform(-1, 1, (300, 2)) * W
    mon = AssumptionMonitor.monotone([[1, 1], [1, -1]], -0.01, 0.01, m=2, n=2,
                                     on_violation="widen", margin=0.1)
    mon.add(Z[:100], Y[:100])
    v = mon.check(Z[100:], Y[100:])
    out.update(c_Z=Z, c_Y=Y, c_viol=viol_array(v), c_dlo=mon.d_lo[None, :],
               c_dhi=mon.d_hi[None, :], c_nchanges=float(len(mon.changes)))

    # D: missing dependence (sign 0), then widen that slope bound
    Z = rng.uniform(0, 1, (250, 2))
    y = Z[:, 0] + 0.8 * Z[:, 1] + rng.uniform(-0.01, 0.01, 250)
    mon = AssumptionMonitor.monotone([1, 0], -0.01, 0.01, m=1, n=2)
    mon.add(Z[:50], y[:50])
    v = mon.check(Z[50:], y[50:])
    out.update(d_Z=Z, d_y=y[:, None], d_viol=viol_array(v))
    need(mon, "d_", out)
    mon.widen(("slope_hi", 0, 1))
    out["d_slope_hi"] = mon.slope_hi

    # E: growth model, nothing stored (max_points=0): a jump is caught by the model bounds only
    Z = rng.uniform(-1, 1, (300, 2))
    y = 0.5 * np.sin(2 * Z[:, 0]) + 0.3 * Z[:, 1] + rng.uniform(-0.02, 0.02, 300)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [8, 8])
    bd = BoundedDerivativeModel.from_data(Z[:, :1], Z[:, 1:], y[:, None], P, [[-1.0]],
                                          [[1.0]], [[0.3]], [[0.3]], -0.02, 0.02,
                                          method="growth")
    mon = AssumptionMonitor.for_model(bd, max_points=0)
    Zq = rng.uniform(-1, 1, (40, 2))
    yq = 0.5 * np.sin(2 * Zq[:, 0]) + 0.3 * Zq[:, 1]
    yq[20:] += 0.6
    v = mon.check(Zq, yq[:, None])
    out.update(e_Z=Z, e_y=y[:, None], e_Zq=Zq, e_yq=yq[:, None], e_viol=viol_array(v),
               e_lo=mon.slope_lo, e_hi=mon.slope_hi)

    path = Path(__file__).with_name("reference_monitor.mat")
    savemat(path, out, do_compression=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
