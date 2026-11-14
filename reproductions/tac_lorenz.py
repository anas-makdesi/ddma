"""TAC paper, Section V-C: controlled Lorenz system (bounded derivatives).

    x+ = x + sigma (y - x) T0 + w1 T0
    y+ = y + (rho x - x z - y + u) T0 + w2 T0
    z+ = z + (x y - beta z) T0 + w3 T0
    sigma = 10, beta = 8/3, rho = 28, T0 = 0.01, |w_i| <= 0.5
    X = [-10, 10]^3, U = [-200, 200]

Step 1 (Section IV): estimate derivative and disturbance bounds with the scenario LP
(5e4 pairs) and (33) (1e6 pairs). Compared with the table of the paper.
Step 2: data-driven abstraction with the estimated bounds and maximal controlled invariant
set with Xs = X. Paper: 1e8 samples, K = 100 per axis (101^4 cells), 929 s.
Memory: each output is learned only on the variables it depends on (x+ on (x, y), z+ on
(x, y, z), y+ on (x, y, z, u)), bounds are stored as float32 (rounded outward), and the
data are processed in batches of 1e7 whose maps are intersected (TAC Prop. 3). With this,
the paper's size (--full) fits in 8 GB. Default: K = 50 (51^4 cells), 1e7 samples.

Run:  python reproductions/tac_lorenz.py [--full]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, Timer, save, tac_partition

from ddma import (
    BoundedDerivativeModel,
    SymbolicAbstraction,
    estimate_derivative_bounds,
    safety,
    scenario_confidence,
    simulate,
)
from ddma.plotting import plot_set_3d

S, B, R, T0, WB = 10.0, 8.0 / 3.0, 28.0, 0.01, 0.5
X_LO, X_HI = -10 * np.ones(3), 10 * np.ones(3)
U_LO, U_HI = np.array([-200.0]), np.array([200.0])

PAPER = {
    "J_hi": [[0.9, 0.1, 1.64e-6], [0.377, 0.99, 9.46e-2], [9.09e-2, 9.31e-2, 0.975]],
    "J_lo": [[0.9, 0.1, -2.1e-7], [0.185, 0.989, -9.27e-2], [-9.3e-2, -9.2e-2, 0.97]],
    "Ju_hi": [-1.9e-7, 9.95e-3, 2.88e-6],
    "Ju_lo": [-1.9e-7, 9.95e-3, 4.58e-5],
    "w_T0": [0.005, 5.85e-2, 5.76e-2],
}
TRUE = {
    "J_hi": [[0.9, 0.1, 0], [0.38, 0.99, 0.1], [0.1, 0.1, 0.973]],
    "J_lo": [[0.9, 0.1, 0], [0.18, 0.99, -0.1], [-0.1, -0.1, 0.973]],
    "Ju": [0, 0.01, 0],
    "w_T0": [0.005, 0.005, 0.005],
}


def lorenz(X, U, W):
    x, y, z = X.T
    return np.stack([
        x + S * (y - x) * T0,
        y + (R * x - x * z - y + U[:, 0]) * T0,
        z + (x * y - B * z) * T0,
    ], 1) + W * T0


def sample(rng, K):
    X = rng.uniform(X_LO, X_HI, (K, 3))
    U = rng.uniform(U_LO, U_HI, (K, 1))
    return X, U, lorenz(X, U, rng.uniform(-WB, WB, (K, 3)))


def main(full, w_weight):
    rng = np.random.default_rng(0)
    out = {}

    # Step 1: bounds from data
    X, U, Xn = sample(rng, 2 * 10**6)
    with Timer() as t1:
        a_lo, a_hi, w = estimate_derivative_bounds(np.hstack([X, U]), Xn, 50_000, 1_000_000,
                                                   rng=rng, w_weight=w_weight)
    np.set_printoptions(precision=3, suppress=False)
    print(f"bounds estimated in {t1.dt:.1f}s")
    print("J_x upper (data):\n", a_hi[:, :3], "\nJ_x lower (data):\n", a_lo[:, :3])
    print("J_u upper/lower (data):", a_hi[:, 3], a_lo[:, 3])
    print("w*T0 (data):", w, "  true:", TRUE["w_T0"], "  paper:", PAPER["w_T0"])
    print(f"Prop. 9 with beta = 1e-6: violation probability <= "
          f"{1 - scenario_confidence(1_000_000, 1e-6):.2e} (paper: < 2e-5)")
    out["w_weight"] = w_weight
    out["bounds"] = {"J_lo": a_lo[:, :3], "J_hi": a_hi[:, :3], "Ju_lo": a_lo[:, 3],
                     "Ju_hi": a_hi[:, 3], "w_T0": w, "time": t1.dt,
                     "paper": PAPER, "true": TRUE}

    # Step 2: abstraction and maximal controlled invariant
    K, N = (100, 10**8) if full else (50, 10**7)
    batch = 5 * 10**6
    Xp = tac_partition(X_LO, X_HI, K, 0.05)
    Up = tac_partition(U_LO, U_HI, K, 0.05)
    joint = tac_partition(np.r_[X_LO, U_LO], np.r_[X_HI, U_HI], K, 0.05)
    features = [[0, 1], [0, 1, 2, 3], [0, 1, 2]]  # structure of the Lorenz equations
    model, t_model = None, 0.0
    for start in range(0, N, batch):
        X, U, Xn = sample(rng, min(batch, N - start))
        with Timer() as tm:
            if model is None:
                model = BoundedDerivativeModel.from_data(
                    X, U, Xn, joint, a_lo[:, :3], a_hi[:, :3], a_lo[:, 3:], a_hi[:, 3:], -w, w,
                    features=features, compact=True)
            else:
                model.update(X, U, Xn)
        t_model += tm.dt
        del X, U, Xn
        print(f"  learned {start + batch:,} samples ({t_model:.1f}s)", flush=True)
    with Timer() as ta:
        abst = SymbolicAbstraction(Xp, Up, model)
    with Timer() as ts:
        ctrl = safety(abst, np.ones(abst.n_states, bool))
    total = t_model + ta.dt
    print(f"{N:.0e} samples, {joint.n_cells:,} cells: model {t_model:.1f}s, abstraction "
          f"{ta.dt:.1f}s (paper, 1e8 / 101^4: 929 s), invariant {ts.dt:.1f}s")
    print(f"maximal controlled invariant: {100 * ctrl.winning_fraction():.1f}% of the cells")

    # closed loop on the true Lorenz system
    def step(x, u, rng):
        return lorenz(x[None], np.atleast_2d(u), rng.uniform(-WB, WB, (1, 3)))[0]

    starts = Xp.cell_centers(Xp.all_indices())[ctrl.domain]
    viol = 0
    for x0 in starts[rng.choice(len(starts), min(200, len(starts)), replace=False)]:
        xs, us = simulate(step, ctrl, x0, 500, rng=rng, pick="random")
        viol += int(len(us) < 500 or np.any((xs < X_LO) | (xs > X_HI)))
    print(f"closed loop: 200 runs x 500 steps, violations: {viol}")
    out["abstraction"] = {"n_samples": N, "K": K, "n_cells": joint.n_cells,
                          "time_model": t_model, "time_abstraction": ta.dt,
                          "time_total": total, "time_invariant": ts.dt,
                          "invariant_fraction": ctrl.winning_fraction(),
                          "closed_loop_violations": viol}
    tag = f"w{w_weight:g}" + ("_full" if full else "")
    np.savez_compressed(RESULTS / f"tac_lorenz_{tag}_domain.npz", domain=ctrl.domain_grid(),
                        edges=np.array(Xp.edges, dtype=object))
    save(f"tac_lorenz_{tag}", out)

    dom = ctrl.domain_grid()
    e = Xp.edges
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, k in zip(axs, (5, dom.shape[2] // 2, dom.shape[2] - 6)):
        ax.pcolormesh(e[0], e[1], dom[:, :, k].T, cmap="Blues", vmin=0, vmax=1.4)
        if not dom[:, :, k].any():
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center",
                    color="0.35")
        ax.set(xlabel="$x$", title=f"$z \\in$ [{e[2][k]:.1f}, {e[2][k + 1]:.1f}]")
    axs[0].set_ylabel("$y$")
    note = "" if dom.any() else (f" -- EMPTY with w_weight={w_weight:g}"
                                 " (disturbance bounds too large; use 50)")
    fig.suptitle("TAC Fig. 12: maximal controlled invariant of the Lorenz system (slices)"
                 + note, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / f"tac_fig12_{tag}_slices.png", dpi=150)
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(projection="3d")
    plot_set_3d(Xp, ctrl.domain, ax=ax, color="#3a7dc1", axis_labels=("$x$", "$y$", "$z$"))
    ax.view_init(22, -50)
    ax.set_title(f"TAC Fig. 12: maximal controlled invariant of the Lorenz system\n"
                 f"({N:.0e} samples, {joint.n_cells:,} cells, w_weight={w_weight:g})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / f"tac_fig12_{tag}.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--w-weight", type=float, default=50.0,
                    help="LP weight of the disturbance term (1 = eq. (32); 50 = original code)")
    a = ap.parse_args()
    main(a.full, a.w_weight)
