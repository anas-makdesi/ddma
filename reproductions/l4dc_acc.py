"""L4DC 2021, Section 5: exact data-driven abstraction (data-induced partition).

    d+ = d + (v - vc) T0,   v+ = v + alpha(u, v) T0,   alpha(u, v) = u - (f0 + f1 v + f2 v^2)/M
    vc = 20 m/s, T0 = 0.1 s, f0 = 52, M = 1370; no disturbance.

300 transitions sampled in the safe set d in [10, 30], v in [19, 21]. The input range is not
stated in the paper; we use u in [-0.4 g, 0.4 g] as in the companion 2-D experiment script.
The partition is induced by the data (one breakpoint
per sample coordinate), so the abstraction is alternatingly bisimilar to the minimal model
(L4DC Theorem 10): no conservatism beyond the data.

Paper: abstraction 923 s, invariant set 341 s (MATLAB R2020b, i7 2.9 GHz).

Run:  python reproductions/l4dc_acc.py
"""

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, Timer, save

from ddma import (
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    safety,
    simulate,
)

VC, T0, M, F0, F1, F2 = 20.0, 0.1, 1370.0, 52.0, 1.2567, 0.4342
G = 9.82
X_LO, X_HI = np.array([10.0, 19.0]), np.array([30.0, 21.0])
U_LO, U_HI = np.array([-0.4 * G]), np.array([0.4 * G])


def plant(X, U):
    d, v = X.T
    a = U[:, 0] - (F0 + F1 * v + F2 * v**2) / M
    return np.stack([d + (v - VC) * T0, v + a * T0], 1)


def main(n=300):
    rng = np.random.default_rng(0)
    X = rng.uniform(X_LO, X_HI, (n, 2))
    U = rng.uniform(U_LO, U_HI, (n, 1))
    Z, Y = np.hstack([X, U]), plant(X, U)

    Xp = RectangularPartition.from_data(X, X_LO, X_HI)
    Up = RectangularPartition.from_data(U, U_LO, U_HI)
    joint = RectangularPartition.product(Xp, Up)
    with Timer() as tf:
        sigma = PartitionedSimulatingMap.from_data(Z, Y, joint)
    with Timer() as ta:
        abst = SymbolicAbstraction(Xp, Up, sigma, chunk=2_000_000)
    with Timer() as ts:
        ctrl = safety(abst, np.ones(abst.n_states, bool))
    area = Xp.cell_bounds(Xp.all_indices())
    vol = np.prod(area[1] - area[0], axis=1)
    frac = vol[ctrl.domain].sum() / vol.sum()
    print(f"{abst.n_states:,} states x {abst.n_inputs} inputs = {abst.enabled.size:,} pairs")
    print(f"times: map {tf.dt:.2f}s, abstraction {ta.dt:.2f}s, invariant {ts.dt:.2f}s "
          f"(paper: 923 s + 341 s)")
    print(f"maximal controlled invariant: {100 * frac:.1f}% of the safe set (by area)")

    def step(x, u, rng):
        return plant(x[None], np.atleast_2d(u))[0]

    starts = Xp.cell_centers(Xp.all_indices())[ctrl.domain]
    viol = 0
    for x0 in starts[rng.choice(len(starts), 200, replace=False)]:
        xs, us = simulate(step, ctrl, x0, 500, rng=rng, pick="random")
        viol += int(len(us) < 500 or np.any((xs < X_LO) | (xs > X_HI)))
    print(f"closed loop: 200 runs x 500 steps, violations: {viol}")
    save("l4dc_acc", {
        "n_data": n, "n_states": abst.n_states, "n_inputs": abst.n_inputs,
        "time_map": tf.dt, "time_abstraction": ta.dt, "time_invariant": ts.dt,
        "invariant_area_fraction": float(frac), "closed_loop_violations": viol,
    })

    fig, ax = plt.subplots(figsize=(6.4, 4))
    e0, e1 = Xp.edges
    ax.pcolormesh(e0, e1, ctrl.domain_grid().T, cmap="Reds", vmin=0, vmax=1.6)
    ax.plot(*X.T, ".", ms=3, color="#264653", label="data points")
    ax.set(xlabel="$d$ [m]", ylabel="$v$ [m/s]",
           title="L4DC Fig. 5: invariant set (red) from 300 transitions")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "l4dc_fig5.png", dpi=150)


if __name__ == "__main__":
    main()
