"""Lazy abstraction: synthesis that evaluates only the transitions it needs.

Three rooms in a row, a hot outside (34 C) and one shared cooling input u in [0, 1]:

    T_i+ = T_i + Ts (a (T_out - T_i) + b (neighbour coupling) - c u (T_i - T_supply)) + w_i,

with unknown heat gains w_i in [0, 0.15]. The dynamics are learned from data as a
monotone map (increasing in the temperatures, decreasing in u). The specification is
"every room stays at or below 26 C".

An eager abstraction stores a successor box for every (state cell, input cell) pair.
With ``lazy=True`` nothing is precomputed: ``safety`` evaluates a transition only when
its scan reaches it. States outside the safe set are never expanded and each safe state
stops at its first good input, so only a few percent of the pairs are evaluated, with
the same winning set and O(|Q|) memory.

Run:  python examples/lazy_synthesis.py [cells per axis, default 60]
"""

import sys
import time

import numpy as np

from ddma import PartitionedSimulatingMap, RectangularPartition, SymbolicAbstraction, safety

T_LO, T_HI, T_MAX = 16.0, 32.0, 26.0  # state box and the safety limit (deg C)
T_OUT, T_SUPPLY, TS, W_HI = 34.0, 14.0, 0.25, 0.15


def plant(T, u):
    """True dynamics without the heat gains (unknown to the learner)."""
    a, b, c = 0.05, 0.3, 0.6
    coupling = np.stack(
        [T[:, 1] - T[:, 0], T[:, 0] + T[:, 2] - 2 * T[:, 1], T[:, 1] - T[:, 2]], axis=1
    )
    return T + TS * (a * (T_OUT - T) + b * coupling - c * u * (T - T_SUPPLY))


def main(K=60, n_inputs=21, n_samples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.uniform(T_LO, T_HI, (n_samples, 3))
    U = rng.uniform(0.0, 1.0, (n_samples, 1))
    T_next = plant(T, U) + rng.uniform(0.0, W_HI, (n_samples, 3))

    Xp = RectangularPartition.uniform([T_LO] * 3, [T_HI] * 3, K)
    Up = RectangularPartition.uniform([0.0], [1.0], n_inputs)
    signs = np.array([[1, 1, 1, -1]] * 3)  # increasing in T, decreasing in u
    sigma = PartitionedSimulatingMap.from_data(
        np.hstack([T, U]), T_next, RectangularPartition.product(Xp, Up), signs=signs
    )

    results = {}
    for lazy in (False, True):
        t0 = time.perf_counter()
        A = SymbolicAbstraction(Xp, Up, sigma, lazy=lazy)
        safe = A.state_mask(lambda lo, hi: np.all(hi <= T_MAX, axis=1))
        ctrl = safety(A, safe)
        dt = time.perf_counter() - t0
        name = "lazy " if lazy else "eager"
        print(
            f"{name}: {dt:5.2f} s, transitions evaluated {A.n_evaluated:>11,} "
            f"of {A.n_states * A.n_inputs:,}, winning {ctrl.winning_fraction():.1%} "
            f"(safe set {safe.mean():.1%})"
        )
        results[lazy] = ctrl
    assert np.array_equal(results[False].domain, results[True].domain)
    print("same winning set")

    # the lazy controller computes its allowed inputs on demand
    x = np.array([24.0, 25.0, 24.5])
    print(f"at T = {x}: cooling u = {results[True](x, pick='min_norm')[0]:.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
