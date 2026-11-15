"""Timings of the abstraction construction (SymbolicAbstraction): NumPy vs the C++ kernel.

    python benchmarks/abstraction.py              # C++ on all threads, and NumPy
    python benchmarks/abstraction.py --threads 1  # C++ on one thread
    python benchmarks/abstraction.py --skip-numpy

Three workloads, all checked to give identical abstractions with both backends:
  * monotone map on an uneven 301 x 301 x 301 grid (L4DC adaptive cruise control size,
    27 million state-input pairs, binary search on the edges);
  * two-matrix bounded-derivative model with per-output features and affine g parts
    (Lorenz system, 51^3 states x 51 inputs, 6.8 million pairs);
  * monotone map plus a known affine part g and an assumption box (3-D, 2.6 million pairs).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ddma
from ddma import (
    BoundedDerivativeModel,
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
)


def uneven(rng, lo, hi, k):
    return RectangularPartition([np.r_[a, np.sort(rng.uniform(a, b, kk - 1)), b]
                                 for a, b, kk in zip(lo, hi, k)])


def cruise_control(rng):
    """Monotone map, data-driven uneven partition (as in reproductions/l4dc_acc.py)."""
    n = 300
    X = rng.uniform([10.0, 19.0], [30.0, 21.0], (n, 2))
    U = rng.uniform(-3.9, 3.9, (n, 1))
    Y = np.stack([X[:, 0] + (X[:, 1] - 20.0) * 0.1,
                  X[:, 1] + 0.1 * (U[:, 0] - (0.1 + 5e-3 * X[:, 1] + 2.5e-4 * X[:, 1] ** 2))], 1)
    Xp = RectangularPartition.from_data(X, [10.0, 19.0], [30.0, 21.0])
    Up = RectangularPartition.from_data(U, [-3.9], [3.9])
    joint = RectangularPartition.product(Xp, Up)
    sm = PartitionedSimulatingMap.from_data(np.hstack([X, U]), Y, joint)
    return "monotone map, uneven 301^2 x 301 (27.3 M pairs)", Xp, Up, sm, {}


def lorenz(rng):
    """Two-matrix model with features (as in reproductions/tac_lorenz.py, K = 50)."""
    s, b, r, T0 = 10.0, 8.0 / 3.0, 28.0, 0.01
    lo, hi = -10 * np.ones(3), 10 * np.ones(3)
    N = 10**6
    X = rng.uniform(lo, hi, (N, 3))
    U = rng.uniform(-200, 200, (N, 1))
    x, y, z = X.T
    Xn = np.stack([x + s * (y - x) * T0, y + (r * x - x * z - y + U[:, 0]) * T0,
                   z + (x * y - b * z) * T0], 1) + rng.uniform(-0.5, 0.5, (N, 3)) * T0
    Jl = np.array([[0.9, 0.1, -1e-6, 0], [0.18, 0.98, -0.1, 0.0099], [-0.1, -0.1, 0.97, 0]])
    Jh = np.array([[0.9, 0.1, 1e-6, 0], [0.38, 0.99, 0.1, 0.0101], [0.1, 0.1, 0.98, 0]])
    Xp = RectangularPartition.uniform(lo, hi, [51] * 3)
    Up = RectangularPartition.uniform([-200.0], [200.0], [51])
    joint = RectangularPartition.product(Xp, Up)
    w = 0.005 * np.ones(3)
    m = BoundedDerivativeModel.from_data(
        X, U, Xn, joint, Jl[:, :3], Jh[:, :3], Jl[:, 3:], Jh[:, 3:], -w, w,
        features=[[0, 1], [0, 1, 2, 3], [0, 1, 2]], compact=True)
    return "two-matrix model + features, 51^3 x 51 (6.8 M pairs)", Xp, Up, m, {}


def room_with_g(rng):
    """Map of a small nonlinearity plus a known affine part, with an assumption box."""
    A = np.array([[0.9, 0.05, 0.0], [0.05, 0.9, 0.05], [0.0, 0.05, 0.9]])
    B = np.array([[-0.2], [-0.2], [-0.2]])
    lo, hi = np.zeros(3), np.ones(3)
    Xp = uneven(rng, lo, hi, [60, 60, 60])
    Up = RectangularPartition.uniform([0.0], [1.0], [12])
    N = 50_000
    X = rng.uniform(lo, hi, (N, 3))
    Y = 0.02 * np.sin(3 * X)  # what is left after the known affine part
    sm = PartitionedSimulatingMap.from_data(X, Y, Xp, -0.01, 0.01)
    e = Xp.edges[2]
    kw = dict(g_bounds=affine_bounds(A, B, c=[0.05, 0.05, 0.05]), features=[0, 1, 2],
              assumption=([-np.inf, -np.inf, e[3]], [np.inf, np.inf, e[-4]]))
    return "map + affine g + assumption, uneven 60^3 x 12 (2.6 M pairs)", Xp, Up, sm, kw


def best_of(f, repeat):
    best, out = np.inf, None
    for _ in range(repeat):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def run(case, args) -> dict:
    name, Xp, Up, sm, kw = case
    t_cpp, A = best_of(lambda: SymbolicAbstraction(Xp, Up, sm, backend="cpp", **kw), args.repeat)
    row = {"workload": name, "cpp": t_cpp, "threads": ddma.get_num_threads(),
           "enabled": float(A.enabled.mean())}
    if not args.skip_numpy:
        t_np, R = best_of(lambda: SymbolicAbstraction(Xp, Up, sm, backend="numpy", **kw), 1)
        same = (np.array_equal(A.enabled, R.enabled) and np.array_equal(A.first, R.first)
                and np.array_equal(A.last, R.last))
        if not same:
            raise SystemExit(f"{name}: backends differ")
        row["numpy"] = t_np
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="0 = all")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-numpy", action="store_true")
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()
    ddma.set_num_threads(args.threads)
    print(f"C++ threads: {ddma.get_num_threads()} (OpenMP: {ddma.openmp_available()})")
    rng = np.random.default_rng(0)
    results = []
    for make in (cruise_control, lorenz, room_with_g):
        row = run(make(rng), args)
        results.append(row)
        extra = f"   NumPy {row['numpy']:6.2f} s  (identical)" if "numpy" in row else ""
        print(f"{row['workload']:58s} C++ {row['cpp']:6.2f} s{extra}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
