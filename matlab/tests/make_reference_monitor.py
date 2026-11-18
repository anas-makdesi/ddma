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
    check_data,
)

LEVEL = {"cells": 0, "sample": 1, "full": 2}


def viol_array(viols):
    """rows: index, output, level (0 cells, 1 sample, 2 full), gap, witness (nan: none),
    found_at"""
    if not viols:
        return np.zeros((0, 6))
    return np.array([[v.index, v.output, LEVEL[v.level], v.gap,
                      np.nan if v.witness is None else v.witness, v.found_at] for v in viols],
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
    mon = AssumptionMonitor.monotone([1, 0], -0.01, 0.01, m=1, n=2, slope_limit=2.0)
    mon.add(Z[:50], y[:50])
    v = mon.check(Z[50:], y[50:])
    out.update(d_Z=Z, d_y=y[:, None], d_viol=viol_array(v))
    need(mon, "d_", out)
    mon.widen(("slope_hi", 0, 1))
    out["d_slope_hi"] = mon.slope_hi

    # E: growth model, nothing stored: a jump is caught by the model bounds (level 1) only
    Z = rng.uniform(-1, 1, (300, 2))
    y = 0.5 * np.sin(2 * Z[:, 0]) + 0.3 * Z[:, 1] + rng.uniform(-0.02, 0.02, 300)
    P = RectangularPartition.uniform([-1, -1], [1, 1], [8, 8])
    bd = BoundedDerivativeModel.from_data(Z[:, :1], Z[:, 1:], y[:, None], P, [[-1.0]],
                                          [[1.0]], [[0.3]], [[0.3]], -0.02, 0.02,
                                          method="growth")
    mon = AssumptionMonitor.for_model(bd)
    Zq = rng.uniform(-1, 1, (40, 2))
    yq = 0.5 * np.sin(2 * Zq[:, 0]) + 0.3 * Zq[:, 1]
    yq[20:] += 0.6
    v = mon.check(Zq, yq[:, None])
    out.update(e_Z=Z, e_y=y[:, None], e_Zq=Zq, e_yq=yq[:, None], e_viol=viol_array(v),
               e_lo=mon.slope_lo, e_hi=mon.slope_hi)

    # F: collect during a run (levels 1-2, random samples), full check at base (flush)
    def mono(K):
        Z = rng.uniform(0, 1, (K, 2))
        Y = np.c_[0.8 * Z[:, 0] + 0.3 * Z[:, 1] + 0.2 * Z[:, 0] ** 2
                  + rng.uniform(-0.05, 0.05, K), 3 + rng.uniform(-1, 1, K)]
        return Z, Y
    lo, hi = np.zeros((2, 2)), np.array([[np.inf, np.inf], [0.0, 0.0]])
    dlo, dhi = np.array([-0.05, -1.0]), np.array([0.05, 1.0])
    Z, Y = mono(700)
    Y[400 + rng.choice(300, 12, replace=False), 0] -= rng.uniform(0.05, 0.5, 12)
    Y[400 + rng.choice(300, 2, replace=False), 1] += 1.5
    mon = AssumptionMonitor(lo, hi, dlo, dhi, full="later", sample=150, seed=11)
    mon.add(Z[:400], Y[:400])
    v1 = mon.check(Z[400:], Y[400:])
    v2 = mon.flush()
    out.update(f_Z=Z, f_Y=Y, f_viol1=viol_array(v1), f_viol2=viol_array(v2),
               f_st=mon._st[: mon.n_points].astype(float)[:, None])
    need(mon, "f_", out)

    # G: check_data with corrupted points
    Z, Y = mono(500)
    Y[rng.choice(500, 5, replace=False), 0] -= 0.4
    r = check_data(Z, Y, lo, hi, dlo, dhi)
    out.update(g_Z=Z, g_Y=Y, g_count=r.violating[None, :].astype(float), g_worst=r.worst[None, :],
               g_pair=r.worst_pair.astype(float), g_conf=r.conflicts.astype(float)[:, None],
               g_need_w=r.needed["width"][None, :], g_need_lo=r.needed["slope_lo"],
               g_need_hi=r.needed["slope_hi"])

    # H, I: widen_then_fallback beyond and under the limit (2 x the original width)
    for key, extra in (("h_", 0.2), ("i_", 0.02)):
        Z = rng.uniform(0, 1, (800, 1))
        y = Z[:, 0] + rng.uniform(-0.05, 0.05, 800)
        y[400:] = Z[400:, 0] + rng.uniform(-0.05 - extra, 0.05 + extra, 400)
        mon = AssumptionMonitor([[0.0]], [[2.0]], -0.05, 0.05,
                                on_violation="widen_then_fallback")
        mon.add(Z[:400], y[:400])
        v = mon.check(Z[400:], y[400:])
        out.update({key + "Z": Z, key + "y": y[:, None], key + "viol": viol_array(v),
                    key + "dlo": mon.d_lo[None, :], key + "dhi": mon.d_hi[None, :],
                    key + "fallback": float(mon.fallback_active),
                    key + "since": float(-1 if mon._fallback_since is None
                                         else mon._fallback_since)})

    path = Path(__file__).with_name("reference_monitor.mat")
    savemat(path, out, do_compression=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
