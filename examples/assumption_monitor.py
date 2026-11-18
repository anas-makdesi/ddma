"""What if the assumptions are wrong? An assumption monitor on adaptive cruise control.

The safety filter of examples/adaptive_cruise_control.py is learned from data under two
assumptions: the speed update is monotone in (v, u), and the disturbance stays within
known bounds (leader speed in [13, 17] m/s, noise within 0.05). Its guarantee holds only
while they do. Before learning, ddma.check_data checks every pair of the learning data against these
assumptions. Then two things go wrong mid-run (t = 75 s), each in its own scenario, with
the same response, "widen_then_fallback" (limit: twice the original disturbance width):

  A. the leader slows down to 11 m/s, below the assumed 13 m/s: a disturbance bound is
     wrong. The needed width stays under the limit, so the bound is widened just enough to
     explain all data seen, the map and the safe set are recomputed, and the run goes on.
  B. a trailer is hitched: the car achieves 30 % of the commanded acceleration and has
     twice the drag, so the learned map no longer describes the plant. The needed width is
     3 times the original, beyond the limit: the car switches to a conservative controller.

Each scenario also runs without a monitor: the filter keeps trusting its model until the
state leaves the certified domain. In all runs the backup controller takes over outside
the domain (the guarantee is gone there anyway); after a refit the domain is smaller, and
the same rule applies: outside the new safe set, use the backup controller.
Every alarm is a proof: the new transition cannot come from a monotone map with the
assumed disturbance bounds, given the data already seen. The example prints which level
of the monitor caught it (cells, sample or full check) and when.

Run:  python examples/assumption_monitor.py   (writes docs/figures/monitor_*.png)
"""

import sys
import time
from pathlib import Path

import numpy as np

from ddma import (
    AssumptionMonitor,
    PartitionedSimulatingMap,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    check_data,
    safety,
    transitions_to_data,
)

T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
VL = (13.0, 17.0)
W = 0.05
X_LO, X_HI = np.array([5.0, 0.0]), np.array([60.0, 30.0])
U_LO, U_HI = np.array([-3.0]), np.array([3.0])
SAFE_LO, SAFE_HI = np.array([10.0, 2.0]), np.array([50.0, 28.0])
GRID = (110, 60, 24)
FEATS = [1, 2]  # the unknown part depends on (v, u) only
A_G = np.array([[1.0, -T], [0.0, 0.0]])
D_LO, D_HI = [T * VL[0], -W], [T * VL[1], W]
SIGNS = [[0, 0], [1, 1]]  # gap update: constant (T v_l); speed update: monotone in (v, u)
K_EVENT, N = 150, 500


def g(X, U):  # known part (h - T v, 0)
    return X @ A_G.T


def speed_update(v, u, trailer=False):
    drag = (F0 + F1 * v + F2 * v**2) / M
    if trailer:  # 30 % of the acceleration, twice the drag
        return v + T * (0.3 * u - 2.0 * drag)
    return v + T * (u - drag)


def plant_step(x, u, rng, vl, trailer=False):
    h, v = x
    return np.array([h - T * v + T * vl,
                     speed_update(v, u[0], trailer) + rng.uniform(-W, W)])


def synthesize(Zd, Y, d_lo, d_hi):
    Xp = RectangularPartition.uniform(X_LO, X_HI, GRID[:2])
    Up = RectangularPartition.uniform(U_LO, U_HI, GRID[2])
    part = RectangularPartition.product(Xp, Up).project(FEATS)
    sigma = PartitionedSimulatingMap.from_data(Zd, Y, part, d_lo, d_hi, signs=SIGNS)
    abst = SymbolicAbstraction(Xp, Up, sigma, g_bounds=affine_bounds(A_G), features=FEATS)
    ctrl = safety(abst, abst.cells_inside(SAFE_LO, SAFE_HI))
    return sigma, ctrl


def safe_fraction(ctrl):
    a = ctrl.abstraction
    return ctrl.domain.sum() / a.cells_inside(SAFE_LO, SAFE_HI).sum()


def nominal(x):  # naive cruise control: hold 22 m/s, ignore the leader
    return np.clip([0.3 * (22.0 - x[1])], U_LO, U_HI)


def fallback(x, x_prev):
    """Backup without a model: match the leader's speed (estimated from the last change of
    the gap) and steer the gap to 30 m."""
    vl_est = x_prev[1] + (x[0] - x_prev[0]) / T
    return np.clip([1.5 * (vl_est - x[1]) + 0.2 * (x[0] - 30.0)], U_LO, U_HI)


def leader_profile(scenario, seed=1):
    r = np.random.default_rng(seed)
    slow = np.full(N, VL[0])
    if scenario == "slow leader":
        slow[K_EVENT:] = 11.0  # below the assumed 13 m/s
    fast = (np.arange(N) // 40) % 2 == 0
    vl = np.where(fast, VL[1], slow) + r.uniform(-0.5, 0.5, N)
    return np.clip(vl, np.minimum(slow, VL[0]), VL[1])


def run(ctrl, sigma, Zd, Y, scenario, mode, seed=2):
    """scenario: "slow leader" or "trailer"; mode: "none" or "monitor"."""
    rng = np.random.default_rng(seed)
    vl = leader_profile(scenario)
    trailer = scenario == "trailer"
    mon, first, seen = None, [], []

    def on_alarm(viols):  # called before the response: the diagnosis at that moment
        seen.extend(viols)
        if not first:
            first.append(mon.report())

    if mode != "none":
        mon = AssumptionMonitor.for_model(
            sigma, Zd, Y, g=g, features=FEATS, on_violation="widen_then_fallback",
            widen_limit=2.0, margin=0.2, callback=on_alarm, input_names=["v", "u"],
            output_names=["gap update", "speed update"])
    x = x_prev = np.array([30.0, 15.0])
    xs, us = [x], []
    ev = {"alarm": None, "refits": [], "outside": None, "first": first, "seen": seen,
          "t_check": []}
    for k in range(N):
        if mon is not None and mon.fallback_active:
            u = fallback(x, x_prev)
        else:
            u = ctrl.filter(x, nominal(x))
            if u is None:  # outside the certified domain (all runs): use the backup
                ev["outside"] = ev["outside"] if ev["outside"] is not None else k
                u = fallback(x, x_prev)
        xn = plant_step(x, u, rng, vl[k], trailer=trailer and k >= K_EVENT)
        if mon is not None:
            t0 = time.perf_counter()
            found = mon.observe(x[None], np.atleast_1d(u)[None], xn[None])
            ev["t_check"].append(time.perf_counter() - t0)
            if found and ev["alarm"] is None:
                ev["alarm"] = k
            if mon.needs_refit:  # widened: refit on all data with the widened bounds
                Zs, Ys = mon.data()
                sigma, ctrl = synthesize(Zs, Ys, mon.d_lo, mon.d_hi)
                mon.set_model(sigma.bounds, sigma.partition.contains)
                ev["refits"].append((k, safe_fraction(ctrl)))
        x_prev, x = x, xn
        xs.append(x)
        us.append(np.atleast_1d(u))
    ev["fell"] = mon is not None and mon.fallback_active
    return np.array(xs), np.array(us), ev, mon


def main(n_samples=200_000):
    rng = np.random.default_rng(0)
    X = rng.uniform(X_LO, X_HI, size=(n_samples, 2))
    U = rng.uniform(U_LO, U_HI, size=(n_samples, 1))
    Xn = np.array([plant_step(x, u, rng, rng.uniform(*VL)) for x, u in zip(X, U)])
    Zd, Y = transitions_to_data(X, U, Xn, g)
    Zd = Zd[:, FEATS]
    t0 = time.perf_counter()
    chk = check_data(Zd, Y, d_lo=D_LO, d_hi=D_HI, signs=SIGNS, input_names=["v", "u"],
                     output_names=["gap update", "speed update"])
    print(f"learning data checked ({n_samples:,} points, every pair) in "
          f"{time.perf_counter() - t0:.2f} s: {chk.report()}")
    t0 = time.perf_counter()
    sigma, ctrl = synthesize(Zd, Y, D_LO, D_HI)
    print(f"learned from {n_samples:,} transitions in {time.perf_counter() - t0:.2f} s; "
          f"certified safe set: {100 * safe_fraction(ctrl):.0f} % of the specification")

    results = {}
    for scenario in ("slow leader", "trailer"):
        for mode in ("none", "monitor"):
            t0 = time.perf_counter()
            xs, us, ev, mon = run(ctrl, sigma, Zd, Y, scenario, mode)
            bad = np.flatnonzero(np.any((xs < SAFE_LO) | (xs > SAFE_HI), axis=1))
            results[scenario, mode] = (xs, us, ev, bad)
            print(f"\n--- {scenario}, {'no monitor' if mode == 'none' else 'monitor'}"
                  f" ({time.perf_counter() - t0:.1f} s): min gap {xs[:, 0].min():.1f} m, "
                  + (f"outside the specification for {bad.size} steps (first at t = "
                     f"{bad[0] * T:.1f} s)" if bad.size else "specification never violated"))
            if ev["alarm"] is not None:
                print(f"event at t = {K_EVENT * T:.1f} s, first alarm at t = "
                      f"{ev['alarm'] * T:.1f} s ({ev['alarm'] - K_EVENT + 1} transitions later)")
            if ev["outside"] is not None:
                print(f"left the certified domain at t = {ev['outside'] * T:.1f} s")
            for k, frac in ev["refits"]:
                print(f"  refit at t = {k * T:.1f} s: certified safe set {100 * frac:.0f} %")
            if mon is not None:
                tc = 1e3 * np.array(ev["t_check"])
                print(f"monitor check per step (all {mon.n_points - N:,} learning points + run): "
                      f"median {np.median(tc):.3f} ms, max {tc.max():.2f} ms")
                print("caught by (level, transition, found at):")
                for v in ev["seen"][:6]:
                    print(f"  {v.level:6s} t = {v.index * T:5.1f} s ({mon._out(v.output)}), "
                          f"found {v.found_at - v.index} steps later, gap {v.gap:.3g}")
                if len(ev["seen"]) > 6:
                    print(f"  ... {len(ev['seen']) - 6} more")
                print("At the first alarm:\n" + ev["first"][0])
                print("At the end:\n" + mon.report())
    if "--no-plot" not in sys.argv:
        plot(results)
    return results


def plot(results):
    import matplotlib.pyplot as plt

    figdir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    t = np.arange(N + 1) * T
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    titles = {"slow leader": "A. leader slower than assumed (disturbance bound wrong)",
              "trailer": "B. trailer hitched (the learned map no longer fits)"}
    for ax, scenario in zip(axs, ("slow leader", "trailer")):
        for (sc, mode), (xs, _us, ev, _bad) in results.items():
            if sc != scenario:
                continue
            c = "#e76f51" if mode == "none" else "#264653"
            lab = "no monitor" if mode == "none" else ("monitor → fallback" if ev["fell"]
                                                       else "monitor → widen + refit")
            ax.plot(t[: len(xs)], xs[:, 0], color=c, lw=1.3, label=lab)
            if ev["alarm"] is not None:
                ax.axvline(ev["alarm"] * T, color=c, ls=":", lw=1.2)
        ax.axvspan(K_EVENT * T, N * T, color="#e9c46a", alpha=0.15, lw=0)
        ax.axhspan(SAFE_LO[0], SAFE_HI[0], color="#2a9d8f", alpha=0.06, lw=0)
        ax.axhline(SAFE_LO[0], ls="--", c="k", lw=0.8)
        ax.axhline(SAFE_HI[0], ls="--", c="k", lw=0.8)
        ax.set_ylim(-5, 60)
        ax.set_ylabel("gap [m]")
        ax.set_title(titles[scenario], fontsize=9, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    axs[1].set_xlabel("time [s]")
    fig.suptitle("Assumptions broken at t = 75 s (shaded): alarm (dotted) and response",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "monitor_cruise_control.png", dpi=160)
    print(f"saved {figdir / 'monitor_cruise_control.png'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 200_000)
