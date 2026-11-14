"""A safe operating envelope from data, without a grid: three rooms, one cooling input.

Same plant as examples/lazy_synthesis.py (unknown to the method):

    T_i+ = T_i + Ts (a (T_out - T_i) + b (neighbour coupling) - c u (T_i - T_supply)) + w_i,

with unknown heat gains w_i in [0, 0.15] and cooling u in [0, 1]. The plant is monotone:
increasing in the temperatures, decreasing in u (so ``input_signs=[-1]``). Specification:
every room at or below 26 C, a lower-closed set.

``monotone_envelope`` turns the raw transitions into a handful of corner temperatures; each
comes with the least cooling that keeps every state below it there forever, whatever the
heat gains. For comparison, ddma's grid abstraction runs on 200 times more data.

Run:  python examples/envelope_room_cooling.py [number of samples, default 1000]
"""

import sys
import time

import numpy as np

from ddma import PartitionedSimulatingMap, RectangularPartition, SymbolicAbstraction, safety
from ddma.invariance import monotone_envelope

T_LO, T_HI, T_MAX = 16.0, 32.0, 26.0
T_OUT, T_SUPPLY, TS, W_HI = 34.0, 14.0, 0.25, 0.15


def plant(T, u):
    a, b, c = 0.05, 0.3, 0.6
    cpl = np.stack([T[:, 1] - T[:, 0], T[:, 0] + T[:, 2] - 2 * T[:, 1], T[:, 1] - T[:, 2]], 1)
    return T + TS * (a * (T_OUT - T) + b * cpl - c * u * (T - T_SUPPLY))


def main(N=1000, n_abstraction=200_000, seed=0, plot=True):
    rng = np.random.default_rng(seed)
    X = rng.uniform(T_LO, T_HI, (N, 3))
    U = rng.uniform(0.0, 1.0, (N, 1))
    Xn = plant(X, U) + rng.uniform(0.0, W_HI, (N, 3))

    t0 = time.perf_counter()
    env = monotone_envelope(X, Xn, U, w_lo=0.0, w_hi=W_HI, input_signs=[-1],
                            x_hi=[T_MAX] * 3, u_lo=[0.0], u_hi=[1.0])
    t_env = time.perf_counter() - t0

    # ddma's grid abstraction (30 cells per axis, 21 input cells) needs data in every
    # cell: it gets the same 200,000 transitions as examples/lazy_synthesis.py
    Xa = rng.uniform(T_LO, T_HI, (n_abstraction, 3))
    Ua = rng.uniform(0.0, 1.0, (n_abstraction, 1))
    Xan = plant(Xa, Ua) + rng.uniform(0.0, W_HI, (n_abstraction, 3))
    t0 = time.perf_counter()
    Xp = RectangularPartition.uniform([T_LO] * 3, [T_HI] * 3, 30)
    Up = RectangularPartition.uniform([0.0], [1.0], 21)
    sigma = PartitionedSimulatingMap.from_data(
        np.hstack([Xa, Ua]), Xan, RectangularPartition.product(Xp, Up),
        signs=np.array([[1, 1, 1, -1]] * 3))
    A = SymbolicAbstraction(Xp, Up, sigma, lazy=True)
    ctrl = safety(A, A.state_mask(lambda lo, hi: np.all(hi <= T_MAX, axis=1)))
    t_abs = time.perf_counter() - t0

    S = rng.uniform(T_LO, T_MAX, (200_000, 3))  # the safe temperature box
    in_env = env.contains(S)
    in_abs = ctrl.domain_grid()[tuple(Xp.locate(S).T)]
    print(f"  envelope:    {t_env:6.3f} s, {N:,} transitions, {env.n_corners} corners, "
          f"certifies {in_env.mean():.1%} of the safe box")
    print(f"  abstraction: {t_abs:6.3f} s, {n_abstraction:,} transitions, 30^3 x 21 cells, "
          f"certifies {in_abs.mean():.1%} of the safe box")
    print("  corners (deg C) and the least cooling each needs:")
    for T, u in sorted(zip(env.pairs_T.tolist(), env.pairs_U[:, 0].tolist()))[:8]:
        print(f"    {np.round(T, 2)}  u >= {u:.2f}")

    # closed loop: a lazy controller wants no cooling at all; the envelope overrides it
    x = S[in_env][:500]
    worst = -np.inf
    energy = []
    for _ in range(200):
        u = np.array([env.safe_input(xi, [0.0])[0] for xi in x])
        energy.append(u.mean())
        x = plant(x, u[:, None]) + rng.choice([0.0, W_HI], size=x.shape)
        assert env.contains(x).all()
        worst = max(worst, x.max())
    print(f"  closed loop, 500 starts x 200 steps: hottest room {worst:.2f} C "
          f"(limit {T_MAX}), 0 violations, mean cooling {np.mean(energy):.2f}")

    if plot:
        plot_slice(env, ctrl, Xp)
    return env, ctrl


def plot_slice(env, ctrl, Xp, t3=24.0):
    import matplotlib.pyplot as plt

    g = np.linspace(T_LO, T_MAX, 300)
    G1, G2 = np.meshgrid(g, g)
    P = np.column_stack([G1.ravel(), G2.ravel(), np.full(G1.size, t3)])
    E = env.contains(P).reshape(G1.shape)
    Ab = ctrl.domain_grid()[tuple(Xp.locate(P).T)].reshape(G1.shape)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.contourf(G1, G2, E, levels=[0.5, 1.5], colors=["#d1495b"], alpha=0.25)
    ax.contour(G1, G2, E, levels=[0.5], colors=["#d1495b"], linewidths=2)
    ax.contour(G1, G2, Ab, levels=[0.5], colors=["#2e5e8c"], linewidths=1.5, linestyles="--")
    ax.fill([], [], color="#d1495b", alpha=0.4, label="envelope, 1,000 transitions")
    ax.plot([], [], color="#2e5e8c", ls="--", label="grid abstraction, 200,000")
    ax.set_xlabel("room 1 [deg C]")
    ax.set_ylabel("room 2 [deg C]")
    ax.set_title(f"Certified safe temperatures, room 3 at {t3:g} C")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("envelope_room_cooling.png", dpi=150)
    print("  figure: envelope_room_cooling.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
