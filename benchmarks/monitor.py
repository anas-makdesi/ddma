"""Timings of the assumption monitor and check_data (C++ kernel, NumPy for comparison).

    python benchmarks/monitor.py                 # K = 20k, 200k, 2M; threads 1 and all
    python benchmarks/monitor.py --sizes 20000 200000 --threads 1 2
    python benchmarks/monitor.py --json out.json

Data: the adaptive-cruise-control learning data of examples/assumption_monitor.py
(z = (v, u), y = (gap update, speed update); assumptions: gap update constant, speed
update monotone, disturbance widths 2 m and 0.1 m/s), uniform over the state-input box.
New points come from the same plant (no violation), which is the normal running case:
pruning has the most to skip there. Each time is the best of several repeats.

Measured per new point (full="step"):
  level 1 "cells": the model's bounds at the point's cell (PartitionedSimulatingMap.bounds);
  level 2 "sample": the point against 2,000 random stored points (C++);
  level 3 "full": the point against every stored point, C++ with pruning, C++ without
          pruning, and NumPy (round 1's cost);
  whole step: AssumptionMonitor.check of one point (levels 1 + 3, Python overhead
          included), median and largest over 200 steps.
Back at base (full="later"): flush() of 500 pending points.
check_data: every pair of the K points, C++ with pruning (and without, for small K).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ddma
from ddma import AssumptionMonitor, PartitionedSimulatingMap, RectangularPartition, check_data
from ddma import _monitor_kernels as npk
from ddma.monitor import slopes_from_signs
from ddma.partitioned import _core

T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
W = 0.05
SIGNS = [[0, 0], [1, 1]]
D_LO, D_HI = np.array([T * 13.0, -W]), np.array([T * 17.0, W])


def data(K, seed):
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, 30, K)
    u = rng.uniform(-3, 3, K)
    drag = (F0 + F1 * v + F2 * v**2) / M
    Y = np.c_[T * rng.uniform(13, 17, K), v + T * (u - drag) + rng.uniform(-W, W, K)]
    return np.ascontiguousarray(np.c_[v, u]), np.ascontiguousarray(Y)


def best(f, reps=5):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def per_point(f, Zq, Yq, reps=3):
    """Best over repeats of the mean time per query point."""
    def run():
        for z, y in zip(Zq, Yq):
            f(z, y)
    return best(run, reps) / len(Zq)


def measure(K, nt, Z, Y, Zq, Yq, sigma, lo, hi, theta):
    """One row of the table: K stored points, nt threads."""
    ddma.set_num_threads(nt)
    st = np.zeros(K + 1, np.uint8)
    r = {"K": K, "threads": nt, "cells": int(_core.monitor_grid(Z, Y)[7].size)}
    r["cells_ms"] = 1e3 * per_point(lambda z, y: sigma.bounds(z[None]), Zq, Yq)
    sidx = np.random.default_rng(2).integers(0, K, 2000)
    r["sample_ms"] = 1e3 * per_point(
        lambda z, y: _core.monitor_list(lo, hi, theta, 1e-9, Z, Y, z, y, sidx, st), Zq, Yq)
    # level 3 for one new point: store it as point K, check it against [0, K)
    Zb = np.vstack([Z, Zq[:1]])
    Yb = np.vstack([Y, Yq[:1]])
    gb = _core.monitor_grid(Zb, Yb)[:10]

    def full(z, y, prune):
        Zb[K], Yb[K] = z, y
        return _core.monitor_full(lo, hi, theta, 1e-9, Zb, Yb, st, K, K + 1,
                                  gb if prune else None, prune)

    r["full_ms"] = 1e3 * per_point(lambda z, y: full(z, y, True), Zq[:50], Yq[:50])
    r["pairs_per_point"] = int(full(Zq[0], Yq[0], True)[5])
    r["full_noprune_ms"] = 1e3 * per_point(lambda z, y: full(z, y, False), Zq[:5], Yq[:5], 2)
    # the whole step, through the monitor: median and largest time per step
    mon = AssumptionMonitor.for_model(sigma, Z, Y)
    ts = []
    for z, y in zip(Zq, Yq):
        t0 = time.perf_counter()
        mon.check(z[None], y[None])
        ts.append(time.perf_counter() - t0)
    r["step_ms"] = 1e3 * float(np.median(ts))
    r["step_max_ms"] = 1e3 * float(np.max(ts))
    # back at base: 500 pending points at once
    Zr, Yr = data(500, 3)

    def back():
        m2 = AssumptionMonitor(lo, hi, D_LO, D_HI, full="later", sample=0)
        m2.add(Z, Y)
        m2.check(Zr, Yr)
        t0 = time.perf_counter()
        m2.flush()
        return time.perf_counter() - t0

    r["flush500_s"] = min(back() for _ in range(2))
    r["check_data_s"] = best(lambda: check_data(Z, Y, d_lo=D_LO, d_hi=D_HI, signs=SIGNS),
                             2 if K < 10**6 else 1)
    if K <= 20_000:
        r["check_data_noprune_s"] = best(lambda: check_data(
            Z, Y, d_lo=D_LO, d_hi=D_HI, signs=SIGNS, prune=False), 1)
    return r


def numpy_full(K, Z, Y, Zq, Yq, lo, hi, theta):
    """Round 1's cost: NumPy, one point against every stored point."""
    st = np.zeros(K, np.uint8)
    idx = np.arange(K)
    return 1e3 * per_point(lambda z, y: npk.monitor_list(lo, hi, theta, 1e-9, Z, Y, z, y, idx,
                                                         st), Zq[:5], Yq[:5], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[20_000, 200_000, 2_000_000])
    ap.add_argument("--threads", type=int, nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if _core is None:
        raise SystemExit("the compiled kernel is not installed")
    threads = a.threads or sorted({1, ddma.get_num_threads()})
    lo, hi = (np.ascontiguousarray(x) for x in slopes_from_signs(SIGNS, 2, 2))
    theta = D_HI - D_LO
    part = RectangularPartition.uniform([0, -3], [30, 3], [60, 24])
    rows = []
    for K in a.sizes:
        Z, Y = data(K, 0)
        Zq, Yq = data(200, 1)
        sigma = PartitionedSimulatingMap.from_data(Z, Y, part, D_LO, D_HI, signs=SIGNS)
        for nt in threads:
            rows.append(measure(K, nt, Z, Y, Zq, Yq, sigma, lo, hi, theta))
            print(json.dumps(rows[-1]), flush=True)
        ddma.set_num_threads(1)
        rows.append({"K": K, "numpy_full_ms": numpy_full(K, Z, Y, Zq, Yq, lo, hi, theta)})
        print(json.dumps(rows[-1]), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
