"""TAC paper, Section V-A (Figures 8, 9, 10): over-approximating an unknown function.

F(z) = 2 (sin z1 + sin z2 + z1 + z2) + w,  w in [-0.1, 0.1],  z in [-pi, pi]^2.

Performance criterion (paper):
    mu(D, Q) = sum_{q in Q'} vol(Z_q) * width(sigma(q)) / sum_{q in Q'} vol(Z_q) * vol(W)
over the cells Q' where sigma is bounded.

Run:  python reproductions/tac_overapproximation.py [--reps R]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, Timer, cell_volumes, save, tac_partition

from ddma import PartitionedSimulatingMap
from ddma.plotting import plot_staircase_bounds

LO, HI, WB = -np.pi, np.pi, 0.1


def F(Z):
    return 2 * (np.sin(Z[:, 0]) + np.sin(Z[:, 1]) + Z[:, 0] + Z[:, 1])


def sample(rng, K):
    Z = rng.uniform(LO, HI, (K, 2))
    return Z, (F(Z) + rng.uniform(-WB, WB, K))[:, None]


def mu(sigma):
    width = (sigma.upper - sigma.lower)[..., 0]
    ok = np.isfinite(width)
    vol = cell_volumes(sigma.partition)
    return float((vol[ok] * width[ok]).sum() / (vol[ok].sum() * 2 * WB))


def fit(Z, Y, K, backend="auto"):
    P = tac_partition([LO, LO], [HI, HI], K, 0.01)
    with Timer() as t:
        s = PartitionedSimulatingMap.from_data(Z, Y, P, -WB, WB, backend=backend)
    return s, t.dt


def main(reps):
    rng = np.random.default_rng(0)
    out = {"reps": reps}

    # Figure 8: |K| = 1e6, K1 = K2 = 30, c = 0.01, sound everywhere
    Z, Y = sample(rng, 10**6)
    s, _ = fit(Z, Y, 30)
    zq = rng.uniform(LO, HI, (200_000, 2))
    lo, hi = s.bounds(zq)
    out["fig8_sound_on_200k_queries"] = bool(
        np.all((lo[:, 0] <= F(zq) - WB + 1e-12) & (F(zq) + WB <= hi[:, 0] + 1e-12))
    )
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plot_staircase_bounds(s, ax=ax, true_fn=F, trim=1)
    ax.set(xlabel="$z_1$", ylabel="$z_2$", zlabel=r"$\sigma \circ \phi$")
    ax.view_init(25, -125)
    ax.set_title("TAC Fig. 8: $F$ with $w = 0$ (solid) and the bounds of the\n"
                 "over-approximation (transparent), $|K| = 10^6$, 31 x 31 cells", fontsize=9)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_staircase_bounds(s, ax=ax, true_fn=F, trim=1, region=(13, 18, 13, 18), alpha=0.45)
    ax.set(xlabel="$z_1$", ylabel="$z_2$", zlabel=r"$\sigma \circ \phi$")
    ax.view_init(18, -115)
    ax.set_title("zoom on 5 x 5 cells: upper and lower staircases around $F$", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "tac_fig8.png", dpi=150)

    # Figure 9: K1 = K2 = 100, |K| = 1e6 ... 1e7
    Ns = [k * 10**6 for k in range(1, 11)]
    t9, m9 = [], []
    for N in Ns:
        ts, ms = [], []
        for _ in range(reps):
            Z, Y = sample(rng, N)
            s, dt = fit(Z, Y, 100)
            ts.append(dt)
            ms.append(mu(s))
        t9.append((np.mean(ts), np.std(ts)))
        m9.append((np.mean(ms), np.std(ms)))
        print(f"fig9  |K|={N:>9,}  time {np.mean(ts):.3f}s  mu {np.mean(ms):.3f}")
    out["fig9"] = {"n_data": Ns, "time": t9, "mu": m9}

    # Figure 10: |K| = 1e6, K1 = K2 = 10 ... 100
    Ks = list(range(10, 101, 10))
    t10, m10, ncells = [], [], []
    for K in Ks:
        ts, ms = [], []
        for _ in range(reps):
            Z, Y = sample(rng, 10**6)
            s, dt = fit(Z, Y, K)
            ts.append(dt)
            ms.append(mu(s))
        ncells.append(s.partition.n_cells)
        t10.append((np.mean(ts), np.std(ts)))
        m10.append((np.mean(ms), np.std(ms)))
        print(f"fig10 cells={ncells[-1]:>6,}  time {np.mean(ts):.3f}s  mu {np.mean(ms):.3f}")
    out["fig10"] = {"K": Ks, "n_cells": ncells, "time": t10, "mu": m10}

    # same data, NumPy backend, for the timing comparison
    Z, Y = sample(rng, 10**7)
    _, dt_cpp = fit(Z, Y, 100, "cpp")
    _, dt_np = fit(Z, Y, 100, "numpy")
    out["time_1e7_K100"] = {"cpp": dt_cpp, "numpy": dt_np}
    save("tac_overapproximation", out)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, xs, tt, mm, xl, title in (
        (axs[0], np.array(Ns) / 1e6, t9, m9, "number of data points [$10^6$]", "TAC Fig. 9"),
        (axs[1], ncells, t10, m10, "number of cells $|Q|$", "TAC Fig. 10"),
    ):
        tt, mm = np.array(tt), np.array(mm)
        ax.errorbar(xs, tt[:, 0], tt[:, 1], color="#2a6f97", marker="o", ms=3)
        ax.set_xlabel(xl)
        ax.set_ylabel("time [s]", color="#2a6f97")
        ax2 = ax.twinx()
        ax2.errorbar(xs, mm[:, 0], mm[:, 1], color="#e76f51", marker="s", ms=3)
        ax2.set_ylabel(r"$\mu(D, Q)$", color="#e76f51")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "tac_fig9_fig10.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    main(ap.parse_args().reps)
