"""Safe adaptive cruise control from data, with an unknown vehicle model.

State x = (h, v): gap to the leader [m] and follower speed [m/s]. Input u: acceleration
command [m/s^2]. Leader speed v_l in [13, 17] m/s is unknown and acts as a disturbance.

    h+ = h - T v + T v_l                       known part g, disturbance T v_l
    v+ = v + T (u - (f0 + f1 v + f2 v^2)/M) + w   UNKNOWN, monotone in (v, u), |w| <= 0.05

The drag model is used only to generate data. From sampled transitions we learn the
minimal simulating map of the speed update on a grid, build a symbolic abstraction and
compute the maximal controlled invariant set of the specification

    h in [10, 50] m   and   v in [2, 28] m/s   at all times, for every leader behaviour.

Run:  python examples/adaptive_cruise_control.py   (writes docs/figures/acc_*.png)
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddma import (
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    safety,
    simulate,
    transitions_to_data,
)

T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
VL = (13.0, 17.0)
W = 0.05
X_LO, X_HI = np.array([5.0, 0.0]), np.array([60.0, 30.0])
U_LO, U_HI = np.array([-3.0]), np.array([3.0])
SAFE_LO, SAFE_HI = np.array([10.0, 2.0]), np.array([50.0, 28.0])


def speed_update(v, u):  # the unknown part (used only to simulate the plant)
    return v + T * (u - (F0 + F1 * v + F2 * v**2) / M)


def plant_step(x, u, rng, vl=None):
    vl = rng.uniform(*VL) if vl is None else vl
    h, v = x
    return np.array([h - T * v + T * vl, speed_update(v, u[0]) + rng.uniform(-W, W)])


# known part g(x, u) = (h - T v, 0); everything else (incl. T v_l and w) goes in f + D
A_G = np.array([[1.0, -T], [0.0, 0.0]])
def g(X, U):
    return X @ A_G.T


def learn_and_synthesize(n_samples, grid=(110, 60, 24), seed=0):
    rng = np.random.default_rng(seed)
    Xp = RectangularPartition.uniform(X_LO, X_HI, grid[:2])
    Up = RectangularPartition.uniform(U_LO, U_HI, grid[2])

    # 1. data: random transitions of the black-box plant
    X = rng.uniform(X_LO, X_HI, size=(n_samples, 2))
    U = rng.uniform(U_LO, U_HI, size=(n_samples, 1))
    Xn = np.array([plant_step(x, u, rng) for x, u in zip(X, U)])
    Zd, Y = transitions_to_data(X, U, Xn, g)

    # 2. minimal simulating map of f + D on the (v, u) grid; f does not depend on h
    t0 = time.perf_counter()
    feats = [1, 2]
    part = RectangularPartition.product(Xp, Up).project(feats)
    d_lo, d_hi = [T * VL[0], -W], [T * VL[1], W]
    sigma = PartitionedSimulatingMap.from_data(Zd[:, feats], Y, part, d_lo, d_hi)
    t_map = time.perf_counter() - t0

    # 3. symbolic abstraction and 4. safety synthesis
    t0 = time.perf_counter()
    abst = SymbolicAbstraction(Xp, Up, sigma, g_bounds=affine_bounds(A_G), features=feats)
    t_abs = time.perf_counter() - t0
    t0 = time.perf_counter()
    ctrl = safety(abst, abst.cells_inside(SAFE_LO, SAFE_HI))
    t_syn = time.perf_counter() - t0
    return ctrl, dict(map=t_map, abstraction=t_abs, synthesis=t_syn)


if __name__ == "__main__":
    print("samples   winning set (% of safe cells)   map / abstraction / synthesis [s]")
    results = {}
    for K in (2_000, 20_000, 200_000):
        ctrl, t = learn_and_synthesize(K)
        a = ctrl.abstraction
        frac = ctrl.domain.sum() / a.cells_inside(SAFE_LO, SAFE_HI).sum()
        results[K] = ctrl
        print(f"{K:>7}   {100 * frac:6.1f}                           "
              f"{t['map']:.3f} / {t['abstraction']:.3f} / {t['synthesis']:.3f}")

    ctrl = results[200_000]
    a = ctrl.abstraction
    print(f"abstraction: {a.n_states} state cells x {a.n_inputs} input cells")

    # Closed loop. The leader alternates between its speed bounds. The nominal
    # controller is a naive cruise controller (hold 22 m/s, ignore the leader); the
    # learned controller is used as a safety filter around it.
    rng = np.random.default_rng(1)
    N = 400
    vl_profile = np.where((np.arange(N) // 40) % 2 == 0, VL[1], VL[0])
    vl_profile = np.clip(vl_profile + rng.uniform(-0.5, 0.5, N), *VL)

    def nominal(x):
        return np.clip([0.3 * (22.0 - x[1])], U_LO, U_HI)

    def run(filtered, seed=2):
        rng = np.random.default_rng(seed)
        k = [0]

        def step(x, u, rng):
            out = plant_step(x, u, rng, vl=vl_profile[k[0]])
            k[0] += 1
            return out

        x0 = np.array([30.0, 15.0])
        if filtered:
            return simulate(step, ctrl, x0, N, rng=rng, nominal=nominal)
        xs, us = [x0], []
        for _ in range(N):  # stops at a collision (gap <= 0)
            us.append(nominal(xs[-1]))
            xs.append(step(xs[-1], us[-1], rng))
            if xs[-1][0] <= 0:
                break
        return np.array(xs), np.array(us)

    xs_raw, us_raw = run(filtered=False)
    xs, us = run(filtered=True)
    assert len(us) == N, "left the controller domain (should be impossible)"
    assert np.all((xs >= SAFE_LO) & (xs <= SAFE_HI)), "specification violated"
    viol = np.flatnonzero(np.any((xs_raw < SAFE_LO) | (xs_raw > SAFE_HI), axis=1))
    changed = np.mean(np.any(np.abs(us - np.array([nominal(x) for x in xs[:-1]])) > 1e-9, axis=1))
    print(f"nominal controller alone: gap below {SAFE_LO[0]:.0f} m at t = {viol[0] * T:.1f} s, "
          f"collision at t = {(len(xs_raw) - 1) * T:.1f} s" if viol.size else
          "nominal controller alone: no violation")
    print(f"with safety filter:       min gap {xs[:, 0].min():.1f} m, no violation in {N} steps; "
          f"filter modified the input in {100 * changed:.0f}% of steps")

    figdir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    he, ve = a.X.edges
    hc, vc = 0.5 * (he[1:] + he[:-1]), 0.5 * (ve[1:] + ve[:-1])

    # winning sets for increasing amounts of data
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.contourf(hc, vc, results[200_000].domain_grid().T.astype(float), levels=[0.5, 1.5],
                colors=["#2a9d8f"], alpha=0.35)
    ax.plot([], [], "s", color="#2a9d8f", alpha=0.5, ms=10, label="200,000 samples")
    for K, c in ((20_000, "#f4a261"), (2_000, "#e76f51")):
        ax.contour(hc, vc, results[K].domain_grid().T.astype(float), levels=[0.5],
                   colors=[c], linewidths=1.5)
        ax.plot([], [], color=c, label=f"{K:,} samples")
    ax.add_patch(plt.Rectangle(SAFE_LO, *(SAFE_HI - SAFE_LO), fill=False, ls="--", lw=1,
                               ec="k", label="specification"))
    ax.set(xlabel="gap $h$ [m]", ylabel="follower speed $v$ [m/s]", xlim=(5, 60), ylim=(0, 30))
    ax.set_title("Certified safe region (maximal controlled invariant set) vs. amount of data",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "acc_invariant.png", dpi=160)

    fig, axs = plt.subplots(3, 1, figsize=(7, 5.4), sharex=True)
    t = np.arange(N + 1) * T
    axs[0].plot(t[: len(xs_raw)], xs_raw[:, 0], color="#e76f51", lw=1.2, label="nominal only")
    axs[0].plot(t, xs[:, 0], color="#264653", lw=1.4, label="with safety filter")
    axs[0].axhline(SAFE_LO[0], ls="--", c="k", lw=0.8)
    axs[0].axhline(0, c="k", lw=0.5)
    axs[0].set_ylabel("gap [m]")
    axs[0].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    axs[1].plot(t, xs[:, 1], color="#264653", label="follower (filtered)")
    axs[1].step(t[:-1], vl_profile, where="post", color="#8d99ae", lw=1, label="leader")
    axs[1].set_ylabel("speed [m/s]")
    axs[1].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    axs[2].step(t[:-1], [nominal(x)[0] for x in xs[:-1]], where="post", color="#8d99ae",
                lw=1, label="nominal")
    axs[2].step(t[:-1], us[:, 0], where="post", color="#2a9d8f", lw=1.2, label="applied")
    axs[2].set(ylabel="$u$ [m/s$^2$]", xlabel="time [s]")
    axs[2].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("A naive cruise controller with and without the data-driven safety filter",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "acc_closed_loop.png", dpi=160)
    print(f"saved figures in {figdir}")
