"""Thesis, Chapter 3, Section 3.4: five over-approximations of a bounded-derivative map.

F(z) = cos z1 + 0.5 cos z2 + d,  d in [-0.1, 0.1],  z in [-pi, pi]^2,
derivative bounds  df/dz1 in [-1, 1],  df/dz2 in [-0.5, 0.5].

Approaches (``BoundedDerivativeModel(method=...)``):
    1 "two_matrix"  2 "all_matrix"  3 "lin_mono"  4 "growth"  5 "lin_growth"

Figures: 3.5 (the bounds, |K| = 1e7, 51 x 51 cells), 3.6 (mu vs number of data),
3.7-3.8 (time vs number of data), 3.9 (mu vs number of cells, |K| = 1e4).
mu is the criterion of the TAC paper / thesis eq. (2.21), on the cells where the
bounds are finite.

Run:  python reproductions/thesis_ch3.py [--reps R] [--max-exp E]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, Timer, cell_volumes, save, tac_partition

from ddma import BoundedDerivativeModel

LO, HI, WB = -np.pi, np.pi, 0.1
G_LO, G_HI = np.array([-1.0, -0.5]), np.array([1.0, 0.5])
METHODS = ["two_matrix", "all_matrix", "lin_mono", "growth", "lin_growth"]
LABELS = {"two_matrix": "1: 2 matrix", "all_matrix": "2: All matrix",
          "lin_mono": "3: Lin + Mono", "growth": "4: Growth", "lin_growth": "5: Lin + Growth"}


def F(Z):
    return np.cos(Z[:, 0]) + 0.5 * np.cos(Z[:, 1])


def sample(rng, K):
    Z = rng.uniform(LO, HI, (K, 2))
    return Z, (F(Z) + rng.uniform(-WB, WB, K))[:, None]


def fit(method, Z, Y, K):
    """Bounds of F on every cell, arrays of shape (K+1, K+1), and the fit time."""
    P = tac_partition([LO, LO], [HI, HI], K, 0.01)
    with Timer() as t:
        M = BoundedDerivativeModel.from_data(Z[:, :1], Z[:, 1:], Y, P, G_LO[None, :1],
                                             G_HI[None, :1], G_LO[None, 1:], G_HI[None, 1:],
                                             -WB, WB, method=method)
    idx = P.all_indices()
    lo, hi = P.cell_bounds(idx)
    blo, bhi = M.successor_bounds(idx, lo[:, :1], hi[:, :1], lo[:, 1:], hi[:, 1:])
    return P, blo[:, 0].reshape(P.shape), bhi[:, 0].reshape(P.shape), t.dt


def mu(P, lo, hi):
    width = hi - lo
    ok = np.isfinite(width)
    vol = cell_volumes(P)
    return float((vol[ok] * width[ok]).sum() / (vol[ok].sum() * 2 * WB))


def sound(P, lo, hi, rng, n=200_000):
    z = rng.uniform(LO, HI, (n, 2))
    q = tuple(P.locate(z).T)
    return bool(np.all((lo[q] <= F(z) - WB + 1e-12) & (F(z) + WB <= hi[q] + 1e-12)))


def main(reps, max_exp):
    rng = np.random.default_rng(0)
    out = {"reps": reps}

    # Figure 3.5: the bounds with |K| = 10^max_exp points on 51 x 51 cells (approach 5)
    Z, Y = sample(rng, 10**max_exp)
    out["fig3_5"] = {}
    for m in METHODS:
        P, lo, hi, dt = fit(m, Z, Y, 50)
        out["fig3_5"][m] = {"time_s": dt, "mu": mu(P, lo, hi), "sound": sound(P, lo, hi, rng)}
        if m == "lin_growth":
            P5, lo5, hi5 = P, lo, hi
    staircase_figure(P5, lo5, hi5, max_exp, RESULTS / "thesis_fig3_5.png")

    # Figure 3.6 (mu) and 3.8 (time, LP approaches) vs number of data, 51 x 51 cells
    exps = np.arange(3, max_exp + 0.01, 0.5)
    res = {m: {"mu": [], "mu_std": [], "time": [], "time_std": []} for m in METHODS}
    for e in exps:
        acc = {m: ([], []) for m in METHODS}
        for _ in range(reps):
            Z, Y = sample(rng, int(round(10**e)))
            for m in METHODS:
                P, lo, hi, dt = fit(m, Z, Y, 50)
                acc[m][0].append(mu(P, lo, hi))
                acc[m][1].append(dt)
        for m in METHODS:
            res[m]["mu"].append(np.mean(acc[m][0]))
            res[m]["mu_std"].append(np.std(acc[m][0]))
            res[m]["time"].append(np.mean(acc[m][1]))
            res[m]["time_std"].append(np.std(acc[m][1]))
    out["fig3_6_log10_K"] = exps
    out["fig3_6"] = res

    # Figure 3.7: time vs number of data (approaches 1, 2, 4), 1e6 ... 1e7 points
    Ns = [k * 10 ** (max_exp - 1) for k in range(1, 11)]
    t7 = {m: ([], []) for m in ("two_matrix", "all_matrix", "growth")}
    for N in Ns:
        ts = {m: [] for m in t7}
        for _ in range(reps):
            Z, Y = sample(rng, N)
            for m in t7:
                ts[m].append(fit(m, Z, Y, 50)[3])
        for m in t7:
            t7[m][0].append(np.mean(ts[m]))
            t7[m][1].append(np.std(ts[m]))
    out["fig3_7_N"] = Ns
    out["fig3_7"] = {m: {"time": v[0], "time_std": v[1]} for m, v in t7.items()}

    # Figure 3.9: mu vs number of cells, |K| = 1e4
    Ks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    m9 = {m: [] for m in METHODS}
    for K in Ks:
        acc = {m: [] for m in METHODS}
        for _ in range(reps):
            Z, Y = sample(rng, 10**4)
            for m in METHODS:
                P, lo, hi, _ = fit(m, Z, Y, K)
                acc[m].append(mu(P, lo, hi))
        for m in METHODS:
            m9[m].append(np.mean(acc[m]))
    out["fig3_9_cells"] = [(K + 1) ** 2 for K in Ks]
    out["fig3_9"] = m9

    # figures
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    for m in METHODS:
        ax.errorbar(exps, res[m]["mu"], res[m]["mu_std"], label=LABELS[m], capsize=2)
    ax.set(xlabel=r"$\log_{10}|K|$", ylabel=r"$\mu$", title="Fig. 3.6: conservatism vs data")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for m in t7:
        ax.errorbar(Ns, out["fig3_7"][m]["time"], out["fig3_7"][m]["time_std"],
                    label=LABELS[m], capsize=2)
    ax.set(xlabel="$|K|$", ylabel="time (s)", title="Fig. 3.7: time vs data (approaches 1, 2, 4)")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    for m in ("lin_mono", "lin_growth"):
        ax.errorbar(exps, res[m]["time"], res[m]["time_std"], label=LABELS[m], capsize=2)
    ax.set(xlabel=r"$\log_{10}|K|$", ylabel="time (s)",
           title="Fig. 3.8: time vs data (approaches 3, 5)")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    for m in METHODS:
        ax.plot(out["fig3_9_cells"], m9[m], "o-", label=LABELS[m], ms=3)
    ax.set(xlabel="number of cells", ylabel=r"$\mu$",
           title=r"Fig. 3.9: conservatism vs cells ($|K| = 10^4$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "thesis_fig3_6_to_3_9.png", dpi=150)
    plt.close(fig)
    save("thesis_ch3", out)
    for m in METHODS:
        print(f"{LABELS[m]:16s} mu(10^{max_exp}) = {res[m]['mu'][-1]:.3f}   "
              f"mu(10^3) = {res[m]['mu'][0]:.2f}   fit time at 10^{max_exp}: "
              f"{res[m]['time'][-1]:.2f} s   sound: {out['fig3_5'][m]['sound']}")


def staircase_figure(P, lo, hi, max_exp, out):
    """Fig. 3.5 in the style of TAC Fig. 8: F and the transparent staircase bounds."""
    from types import SimpleNamespace

    from ddma.plotting import plot_staircase_bounds

    m = SimpleNamespace(partition=P, lower=lo[..., None], upper=hi[..., None])
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plot_staircase_bounds(m, ax=ax, true_fn=F, trim=1, alpha=0.12)
    ax.set(xlabel="$z_1$", ylabel="$z_2$", zlabel="bounds")
    ax.view_init(25, -125)
    ax.set_title(f"$F(z) = \\cos z_1 + 0.5\\cos z_2$ (solid) and the bounds of the growth-cone "
                 f"model\n(approach 5), $|K| = 10^{max_exp}$ samples, 51 x 51 cells", fontsize=9)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_staircase_bounds(m, ax=ax, true_fn=F, trim=1, region=(20, 26, 20, 26), alpha=0.45)
    ax.set(xlabel="$z_1$", ylabel="$z_2$", zlabel="bounds")
    ax.view_init(18, -115)
    ax.set_title("zoom on 6 x 6 cells: upper and lower staircases around $F$", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def readme_figure():
    """docs/figures/bounded_derivative_methods.png from the saved results."""
    import json
    from pathlib import Path

    d = json.loads((RESULTS / "thesis_ch3.json").read_text())
    colors = {"two_matrix": "#e76f51", "all_matrix": "#f4a261", "lin_mono": "#e9c46a",
              "growth": "#264653", "lin_growth": "#2a9d8f"}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in METHODS:
        axs[0].plot(d["fig3_6_log10_K"], d["fig3_6"][m]["mu"], "o-", ms=3, color=colors[m],
                    label=LABELS[m])
        axs[1].plot(d["fig3_9_cells"], d["fig3_9"][m], "o-", ms=3, color=colors[m])
    axs[0].set(xlabel=r"number of samples ($\log_{10}$)", ylabel=r"conservatism $\mu$",
               title="51 x 51 cells")
    axs[1].set(xlabel="number of cells", ylabel=r"conservatism $\mu$",
               title=r"$10^4$ samples", ylim=(0, 21))
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle(r"Five models of $F(z) = \cos z_1 + 0.5 \cos z_2 + [-0.1, 0.1]$"
                 r" (lower $\mu$ = tighter; 1 = exact)", fontsize=10)
    fig.tight_layout()
    figdir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out = figdir / "bounded_derivative_methods.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-exp", type=int, default=7)
    ap.add_argument("--readme-figure", action="store_true",
                    help="only redraw the README figure from the saved results")
    a = ap.parse_args()
    if not a.readme_figure:
        main(a.reps, a.max_exp)
    readme_figure()
    # 3-D staircase figure (also used in the README)
    rng = np.random.default_rng(0)
    Z, Y = sample(rng, 10**a.max_exp)
    P, lo, hi, _ = fit("lin_growth", Z, Y, 50)
    staircase_figure(P, lo, hi, a.max_exp, RESULTS / "thesis_fig3_5.png")
    from pathlib import Path
    staircase_figure(P, lo, hi, a.max_exp,
                     Path(__file__).resolve().parents[1] / "docs" / "figures"
                     / "bounded_derivative_3d.png")
