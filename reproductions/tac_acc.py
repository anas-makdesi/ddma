"""TAC paper, Section V-B: cruise control with an unknown monotone part.

    d+  = d + (vf - vl) T0,   vf+ = vf + gamma(u, vf) T0,   vl+ = vl + w T0
    gamma(u, v) = u - (f0 + f1 v + f2 v^2) / M,  T0 = 0.7, w in [-2, 2]
    X = [-50, -5] x [0, 30] x [5, 25],  U = [-3, 3]

Known part g(x, u) = (d + (vf - vl) T0, vf, vl); unknown part gamma(u, vf) T0, which is
increasing in u and decreasing in vf (the paper uses the change of variable vf -> -vf;
here the sign pattern does the same). Partition K = 50 per axis, c = 0.005, |K| = 1e6
samples. Specification: assume-guarantee contract (39), bounds = inner partition box.

Run:  python reproductions/tac_acc.py
"""

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, Timer, save, tac_partition

from ddma import PartitionedSimulatingMap, SymbolicAbstraction, affine_bounds, safety, simulate
from ddma.plotting import plot_set_3d

T0, M, F0, F1, F2 = 0.7, 1370.0, 51.0, 1.2567, 0.4342
X_LO, X_HI = np.array([-50.0, 0.0, 5.0]), np.array([-5.0, 30.0, 25.0])
U_LO, U_HI = np.array([-3.0]), np.array([3.0])
WL = 2.0
C = 0.005
A_G = np.array([[1.0, T0, -T0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def gamma(u, v):
    return u - (F0 + F1 * v + F2 * v**2) / M


def g(X, U):
    return X @ A_G.T


def plant(X, U, w):
    return g(X, U) + np.stack([0 * w, gamma(U[:, 0], X[:, 1]) * T0, w * T0], 1)


def main():
    rng = np.random.default_rng(0)
    K = 10**6
    X = rng.uniform(X_LO, X_HI, (K, 3))
    U = rng.uniform(U_LO, U_HI, (K, 1))
    Xn = plant(X, U, rng.uniform(-WL, WL, K))
    Z, Y = np.hstack([X, U]), Xn - g(X, U)

    Xp = tac_partition(X_LO, X_HI, 50, C)
    Up = tac_partition(U_LO, U_HI, 50, C)
    joint = tac_partition(np.r_[X_LO, U_LO], np.r_[X_HI, U_HI], 50, C)
    d_lo, d_hi = np.array([0, 0, -WL * T0]), np.array([0, 0, WL * T0])
    signs = np.array([[0, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 0]])

    # learning over the full 4-D (x, u) grid, as in the paper (0.633 s there, MATLAB MEX)
    with Timer() as t4:
        PartitionedSimulatingMap.from_data(Z, Y, joint, d_lo, d_hi, signs=signs)
    # the same information on the (vf, u) grid only (gamma does not depend on d, vl)
    feats = [1, 3]
    with Timer() as t2:
        sigma = PartitionedSimulatingMap.from_data(Z[:, feats], Y, joint.project(feats), d_lo,
                                                   d_hi, signs=signs[:, feats])
    inner_lo = np.array([e[1] for e in Xp.edges])
    inner_hi = np.array([e[-2] for e in Xp.edges])
    assumption = (np.array([-np.inf, -np.inf, inner_lo[2]]),
                  np.array([np.inf, np.inf, inner_hi[2]]))
    with Timer() as ta:
        abst = SymbolicAbstraction(Xp, Up, sigma, g_bounds=affine_bounds(A_G), features=feats,
                                   assumption=assumption)
    safe = abst.cells_inside(inner_lo, inner_hi)
    with Timer() as ts:
        ctrl = safety(abst, safe)
    frac = ctrl.domain.sum() / safe.sum()
    print(f"fit 4-D grid ({joint.n_cells:,} cells): {t4.dt:.3f}s; (vf,u) grid: {t2.dt:.3f}s")
    print(f"abstraction {abst.n_states:,} x {abst.n_inputs} in {ta.dt:.2f}s; safety {ts.dt:.2f}s")
    print(f"controllable set: {100 * frac:.1f}% of the safe cells")

    def step(x, u, rng):
        xn = plant(x[None], np.atleast_2d(u), np.array([rng.uniform(-WL, WL)]))[0]
        xn[2] = np.clip(xn[2], inner_lo[2], np.nextafter(inner_hi[2], -np.inf))
        return xn

    starts = Xp.cell_centers(Xp.all_indices())[ctrl.domain]
    viol = 0
    for x0 in starts[rng.choice(len(starts), 200, replace=False)]:
        xs, us = simulate(step, ctrl, x0, 300, rng=rng, pick="random")
        viol += int(len(us) < 300 or np.any((xs < inner_lo) | (xs > inner_hi)))
    print(f"closed loop: 200 runs x 300 steps, violations: {viol}")
    save("tac_acc", {
        "time_fit_4d": t4.dt, "time_fit_2d": t2.dt, "n_cells_4d": joint.n_cells,
        "time_abstraction": ta.dt, "time_synthesis": ts.dt,
        "n_states": abst.n_states, "n_inputs": abst.n_inputs,
        "controllable_fraction_of_safe_cells": float(frac),
        "closed_loop_runs": 200, "closed_loop_violations": viol,
    })

    dom = ctrl.domain_grid()
    e0, e1, e2 = Xp.edges
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, k in zip(axs, (1, 25, 49)):
        ax.pcolormesh(e0, e1, dom[:, :, k].T, cmap="Greens", vmin=0, vmax=1.4)
        ax.set(xlim=(inner_lo[0], inner_hi[0]), ylim=(inner_lo[1], inner_hi[1]),
               xlabel="$d$ [m]", title=f"$v_l \\in$ [{e2[k]:.1f}, {e2[k + 1]:.1f}] m/s")
    axs[0].set_ylabel("$v_f$ [m/s]")
    fig.suptitle("TAC Fig. 11: controllable set (green), slices at three leader speeds",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "tac_fig11_slices.png", dpi=150)
    np.savez_compressed(RESULTS / "tac_acc_domain.npz", domain=dom)
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(projection="3d")
    plot_set_3d(Xp, ctrl.domain, ax=ax, color="#2a9d8f", axis_labels=("$d$", "$v_f$", "$v_l$"))
    ax.set(xlim=(inner_lo[0], inner_hi[0]), ylim=(inner_lo[1], inner_hi[1]),
           zlim=(inner_lo[2], inner_hi[2]))
    ax.view_init(22, -60)
    ax.set_title("TAC Fig. 11: controllable set of the symbolic controller\n"
                 "enforcing the assume-guarantee contract (39)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "tac_fig11.png", dpi=150)


if __name__ == "__main__":
    main()
