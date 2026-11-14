"""ADHS 2021, Section 5: data-driven abstraction of a 3-D cruise-control model.

    d+  = d + (v1 - v2) T + w1,   v1+ = v1 + alpha(u, v1) T + w2,   v2+ = v2 + w3
    alpha(u, v) = u - (f0 + f1 v + f2 v^2) / M,  T = 0.5, |w_i| <= 0.1

The leader speed is negated (v2 -> -v2) so that the fully unknown map is monotone.
Data are sampled in XD = [-100, 0] x [10, 30] x [-22, -18], UD = [-3, 3].
Partitions: N uniform cells on [a_1, a_N] plus two outer cells per axis.

mu(D, Q, P) (paper's criterion): average volume of the successor set (inner cells only)
divided by vol(W_D), over inner state and input cells.

Run:  python reproductions/adhs_acc.py [--reps R]
"""

import argparse

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
from ddma.plotting import plot_set_3d

T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
XD_LO, XD_HI = np.array([-100.0, 10.0, -22.0]), np.array([0.0, 30.0, -18.0])
UD_LO, UD_HI = np.array([-3.0]), np.array([3.0])
WB = 0.1
A_LO, A_HI = np.array([-80.0, 15.0, -21.5]), np.array([-20.0, 25.0, -18.5])
B_LO, B_HI = np.array([-2.5]), np.array([2.5])


def plant(X, U, W):
    d, v1, v2n = X.T
    a = U[:, 0] - (F0 + F1 * v1 + F2 * v1**2) / M
    return np.stack([d + (v1 + v2n) * T, v1 + a * T, v2n], axis=1) + W


def sample(rng, K):
    X = rng.uniform(XD_LO, XD_HI, (K, 3))
    U = rng.uniform(UD_LO, UD_HI, (K, 1))
    return np.hstack([X, U]), plant(X, U, rng.uniform(-WB, WB, (K, 3)))


def partitions(N1, N2, N3, Mu):
    Xp = RectangularPartition.with_margins(XD_LO, XD_HI, A_LO, A_HI, [N1, N2, N3])
    Up = RectangularPartition.with_margins(UD_LO, UD_HI, B_LO, B_HI, [Mu])
    return Xp, Up, RectangularPartition.product(Xp, Up)


def mu(sigma, Xp, Up):
    """ADHS criterion on inner cells (indices 1 .. N per axis)."""
    inner = [np.arange(1, k - 1) for k in Xp.shape + Up.shape]
    idx = np.stack(np.meshgrid(*inner, indexing="ij"), -1).reshape(-1, 4)
    lo, hi = sigma.bounds_at(idx)
    vol = np.ones(len(idx))
    for j, e in enumerate(Xp.edges):
        a = np.clip(np.nan_to_num(lo[:, j], neginf=e[0] - 1), e[0] - 1, e[-1] + 1)
        b = np.clip(np.nan_to_num(hi[:, j], posinf=e[-1] + 1), e[0] - 1, e[-1] + 1)
        first = np.clip(np.searchsorted(e, a, "right") - 1, 0, e.size - 2)
        last = np.clip(np.searchsorted(e, b, "right") - 1, 0, e.size - 2)
        cnt = np.clip(np.minimum(last, e.size - 3) - np.maximum(first, 1) + 1, 0, None)
        vol *= cnt * (e[2] - e[1])
    return float(vol.mean() / (2 * WB) ** 3)


def main(reps):
    rng = np.random.default_rng(0)
    out = {}

    # Experiment 1 (Figs. 2, 3): fixed partition, growing data
    Xp, Up, joint = partitions(30, 30, 20, 12)
    Ns = [k * 10**5 for k in range(1, 9)]
    e1 = {"n_data": Ns, "time": [], "mu": []}
    for N in Ns:
        ts, ms = [], []
        for _ in range(reps):
            Z, Y = sample(rng, N)
            with Timer() as t:
                s = PartitionedSimulatingMap.from_data(Z, Y, joint, -WB, WB)
            ts.append(t.dt)
            ms.append(mu(s, Xp, Up))
        e1["time"].append(float(np.mean(ts)))
        e1["mu"].append(float(np.mean(ms)))
        print(f"exp1 N={N:>7,}  time {np.mean(ts):.3f}s  mu {np.mean(ms):.0f}")
    out["experiment1"] = e1

    # Experiment 2 (Figs. 4, 5): N = 1e4 data, growing partitions (N1 = N2 = N3 = M = k)
    Z, Y = sample(rng, 10**4)
    e2 = {"k": [], "n_state_cells": [], "time": [], "mu": []}
    for k in range(4, 27, 2):
        Xp2, Up2, joint2 = partitions(k, k, k, k)
        with Timer() as t:
            s = PartitionedSimulatingMap.from_data(Z, Y, joint2, -WB, WB)
        e2["k"].append(k)
        e2["n_state_cells"].append(k**3)
        e2["time"].append(t.dt)
        e2["mu"].append(mu(s, Xp2, Up2))
        print(f"exp2 k={k:>2} inner state cells {k**3:>6,}  time {t.dt:.3f}s  "
              f"mu {e2['mu'][-1]:.0f}")
    out["experiment2"] = e2

    # Controller synthesis (Fig. 6): assume-guarantee contract, 1e6 samples
    Z, Y = sample(rng, 10**6)
    with Timer() as t_map:
        s = PartitionedSimulatingMap.from_data(Z, Y, joint, -WB, WB)
    assumption = (np.array([-np.inf, -np.inf, A_LO[2]]), np.array([np.inf, np.inf, A_HI[2]]))
    with Timer() as t_abs:
        abst = SymbolicAbstraction(Xp, Up, s, assumption=assumption)
    safe = abst.cells_inside(A_LO, A_HI)
    with Timer() as t_syn:
        ctrl = safety(abst, safe)
    frac = ctrl.domain.sum() / safe.sum()
    print(f"synthesis: {abst.n_states} x {abst.n_inputs} abstraction, controllable set "
          f"{100 * frac:.1f}% of the safe cells; times map {t_map.dt:.2f}s, "
          f"abstraction {t_abs.dt:.2f}s, synthesis {t_syn.dt:.2f}s")

    # closed-loop check on the true plant, leader speed within the assumption
    def step(x, u, rng):
        w = rng.uniform(-WB, WB, 3)
        xn = plant(x[None], np.atleast_2d(u), w[None])[0]
        # environment respects its assumption: leader speed stays in the assumed cells
        # [a, b) (b itself belongs to the next, half-open cell)
        xn[2] = np.clip(xn[2], A_LO[2], np.nextafter(A_HI[2], -np.inf))
        return xn

    starts = Xp.cell_centers(Xp.all_indices())[ctrl.domain]
    viol = 0
    for x0 in starts[rng.choice(len(starts), min(200, len(starts)), replace=False)]:
        xs, us = simulate(step, ctrl, x0, 300, rng=rng, pick="random")
        viol += int(len(us) < 300 or np.any((xs < A_LO) | (xs > A_HI)))
    out["synthesis"] = {
        "controllable_fraction_of_safe_cells": float(frac),
        "n_states": abst.n_states, "n_inputs": abst.n_inputs,
        "time_map": t_map.dt, "time_abstraction": t_abs.dt, "time_synthesis": t_syn.dt,
        "closed_loop_runs": 200, "closed_loop_violations": viol,
    }
    print(f"closed loop: 200 runs x 300 steps, violations: {viol}")
    save("adhs_acc", out)

    # figures
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(np.array(Ns) / 1e5, e1["mu"], "o-", color="#e76f51")
    axs[0].set(xlabel="number of data points [$10^5$]", ylabel=r"$\mu(D,Q,P)$",
               title="ADHS Fig. 3")
    axs[1].plot(e2["n_state_cells"], e2["mu"], "o-", color="#e76f51")
    axs[1].set(xlabel="inner state cells $N_1 N_2 N_3$", ylabel=r"$\mu(D,Q,P)$",
               title="ADHS Fig. 5 (N = $10^4$)")
    fig.tight_layout()
    fig.savefig(RESULTS / "adhs_fig3_fig5.png", dpi=150)

    dom = ctrl.domain_grid()
    e0, ev, el = Xp.edges
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, k in zip(axs, (1, dom.shape[2] // 2, dom.shape[2] - 2)):
        ax.pcolormesh(e0, ev, dom[:, :, k].T, cmap="Greens", vmin=0, vmax=1.4, shading="flat")
        ax.set(xlim=(A_LO[0], A_HI[0]), ylim=(A_LO[1], A_HI[1]), xlabel="$d$ [m]",
               title=f"leader speed $v_2 \\in$ [{-el[k + 1]:.2f}, {-el[k]:.2f}] m/s")
    axs[0].set_ylabel("$v_1$ [m/s]")
    fig.suptitle("ADHS Fig. 6: controllable set (green), slices at three leader speeds",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "adhs_fig6_slices.png", dpi=150)
    np.savez_compressed(RESULTS / "adhs_domain.npz", domain=dom)
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(projection="3d")
    plot_set_3d(Xp, ctrl.domain, ax=ax, color="#1a9e9a",
                axis_labels=("$d$ [m]", "$v_1$ [m/s]", "$-v_2$ [m/s]"))
    ax.set(xlim=(A_LO[0], A_HI[0]), ylim=(A_LO[1], A_HI[1]), zlim=(A_LO[2], A_HI[2]))
    ax.view_init(22, -55)
    ax.set_title("ADHS Fig. 6: controllable set of the symbolic controller\n"
                 "enforcing the assume-guarantee contract (8)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "adhs_fig6.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    main(ap.parse_args().reps)
