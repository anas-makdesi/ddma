"""Timings of the learning kernels of the C++ core (fit_sigma, growth_vertices, growth_local).

    python benchmarks/kernels.py              # current settings (all threads)
    python benchmarks/kernels.py --threads 1  # one thread
    python benchmarks/kernels.py --mode direct --threads 1   # the v0.8 single-threaded kernel

``--mode`` chooses how data points are written into the grid (an internal switch, only for
measurements): ``auto`` (default), ``direct`` (each point written straight into the big
array, one thread: the v0.8 kernel), ``bucketed`` (points first grouped by block of the
grid, then each block filled while it sits in cache), ``private`` (each thread fills its
own copy of the grid, then the copies are merged), ``owned`` (the grid is cut into parts,
points are listed under the parts they write to, one thread per part). All modes give
identical results.

The kernel time is split into "scatter" (locate + write the points) and "sweeps" (the
running min/max over the grid), measured by running the same grid with a single point.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ddma import _core

MODES = {"auto": 0, "direct": 1, "bucketed": 2, "private": 3, "owned": 4}


def best_of(f, repeat):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def workloads(rng):
    """(name, function(Z, y-or-Y), Z, Y) tuples."""
    out = []
    # 1. Lorenz-size monotone map: 4-D, 51^4 = 6.8 M cells, 5 M samples, one output.
    n, K = 4, 5_000_000
    edges = [np.linspace(0, 1, 52) for _ in range(n)]
    Z = rng.uniform(-0.02, 1.02, (K, n))
    Y = (Z @ np.array([1.0, -0.5, 0.3, 0.2]))[:, None] + rng.uniform(-0.01, 0.01, (K, 1))
    signs = np.array([[1, -1, 1, 0]], np.int8)
    theta = np.array([0.02])
    for single in (True, False):
        name = f"fit_sigma 4-D 51^4 cells, 5M samples, {'float32' if single else 'float64'}"
        out.append((name, lambda Z, Y, s=single: _core.fit_sigma(Z, Y, edges, theta, signs, s),
                    Z, Y))
    # 2. Monotone map with three outputs: 3-D, 200^3 = 8 M cells, 2 M samples.
    n, K = 3, 2_000_000
    edges3 = [np.linspace(0, 1, 201) for _ in range(n)]
    Z3 = rng.uniform(0, 1, (K, n))
    Y3 = Z3 @ rng.uniform(-1, 1, (n, 3))
    signs3 = np.array([[1, 1, -1], [0, 1, 1], [-1, 1, 0]], np.int8)
    th3 = np.full(3, 0.01)
    out.append(("fit_sigma 3-D 200^3 cells x 3 outputs, 2M samples, float64",
                lambda Z, Y: _core.fit_sigma(Z, Y, edges3, th3, signs3, False), Z3, Y3))
    # 3. Growth cones at the vertices: small grid (50 x 50) and large grid (2000 x 2000).
    gl, gh = np.array([-1.0, -0.5]), np.array([0.8, 1.5])
    Zg = rng.uniform(0, 1, (10_000_000, 2))
    yg = np.sin(3 * Zg[:, 0]) * 0.3 + Zg[:, 1] * 0.5
    for k in (50, 2000):
        e = [np.linspace(0, 1, k + 1)] * 2
        out.append((f"growth_vertices 2-D {k}x{k} cells, 10M points",
                    lambda Z, y, e=e: _core.growth_vertices(Z, y, e, gl, gh, 0.01), Zg, yg))
    # 4. Growth cones, local maps with slabs: 3-D 30^3 and 100^3 cells, 2 M points.
    gl3, gh3 = np.array([-1.0, -0.5, 0.0]), np.array([0.8, 1.5, 0.7])
    Zl = rng.uniform(0, 1, (2_000_000, 3))
    yl = Zl @ np.array([0.3, 0.6, 0.2])
    for k in (30, 100):
        e = [np.linspace(0, 1, k + 1)] * 3
        out.append((f"growth_local slab 3-D {k}^3 cells, 2M points",
                    lambda Z, y, e=e: _core.growth_local(Z, y, e, gl3, gh3, 0.01, True), Zl, yl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="0 = all")
    ap.add_argument("--mode", choices=list(MODES), default="auto")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()
    has_switch = hasattr(_core, "set_num_threads")
    if has_switch:
        _core.set_num_threads(args.threads)
        _core._set_scatter_mode(MODES[args.mode])
        print(f"threads: {_core.get_num_threads()}, mode: {args.mode}, "
              f"OpenMP: {_core.openmp_available()}")
    else:
        print("v0.8 kernel (no thread or mode switches)")
    rng = np.random.default_rng(0)
    res = {}
    for name, f, Z, Y in workloads(rng):
        total = best_of(lambda f=f, Z=Z, Y=Y: f(Z, Y), args.repeat)
        sweeps = best_of(lambda f=f, Z=Z, Y=Y: f(Z[:1], Y[:1]), args.repeat)
        res[name] = {"total": total, "sweeps": sweeps, "scatter": max(total - sweeps, 0.0)}
        print(f"{name:62s} total {total:6.3f} s   scatter {total - sweeps:6.3f} s   "
              f"sweeps {sweeps:6.3f} s", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"threads": args.threads, "mode": args.mode, "results": res}, fh, indent=1)


if __name__ == "__main__":
    main()
