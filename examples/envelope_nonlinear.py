"""Safe envelopes of nonlinear systems that are not monotone, from data and Jacobian bounds.

``embedding_envelope`` examines every pair of transitions. The two points span a box of
states (and of inputs); bounds on the Jacobian turn each point into a cone that contains
the dynamics, and the two cones bound where the whole box can go. Boxes that provably map
into themselves, for every input of their input box and every disturbance, are kept; their
union is a certified safe envelope.

1. Two competing species (discrete Lotka-Volterra) with harvesting/stocking inputs
   u in [-0.02, 0.02]^2 and disturbances in [-0.005, 0.005], on [0.6, 1.1]^2.
2. A ring of ten coupled nonlinear units, x_i+ = 0.5 x_i + 0.1 sin(x_i-1) + 0.1 cos(x_i+1),
   on [-2, 2]^10 (no input, no disturbance), as in the IFAC 2026 paper.

Run:  python examples/envelope_nonlinear.py
"""

import time

import numpy as np

from ddma.invariance import embedding_envelope

LO, HI, U_MAX, W = 0.6, 1.1, 0.02, 0.005


def competition(x, u):
    return np.stack([x[:, 0] * (2 - x[:, 0] - 0.2 * x[:, 1]) + u[:, 0],
                     x[:, 1] * (2 - x[:, 1] - 0.2 * x[:, 0]) + u[:, 1]], 1)


def competition_example(N=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(LO, HI, (N, 2))
    U = rng.uniform(-U_MAX, U_MAX, (N, 2))
    Xn = competition(X, U) + rng.uniform(-W, W, (N, 2))
    # Jacobian bounds on the domain: df_i/dx_i = 2 - 2 x_i - 0.2 x_j, df_i/dx_j = -0.2 x_i
    jx_lo = np.array([[2 - 2.2 * HI, -0.2 * HI], [-0.2 * HI, 2 - 2.2 * HI]])
    jx_hi = np.array([[2 - 2.2 * LO, -0.2 * LO], [-0.2 * LO, 2 - 2.2 * LO]])
    t0 = time.perf_counter()
    env = embedding_envelope(X, Xn, U, jx_lo, jx_hi, np.eye(2), np.eye(2), w_lo=-W, w_hi=W,
                             x_lo=[LO, LO], x_hi=[HI, HI])
    dt = time.perf_counter() - t0
    S = rng.uniform(LO, HI, (200_000, 2))
    print(f"competition: {N} transitions, {env.n_pairs:,} pairs in {dt:.2f} s -> "
          f"{env.n_boxes} invariant boxes, {env.contains(S).mean():.1%} of the domain")

    # closed loop: a controller that harvests as much as possible (u = -0.02), filtered
    starts = S[env.contains(S)][:300]
    x = starts.copy()
    traj = [x.copy()]
    for _ in range(40):
        u = np.array([env.safe_input(xi, [-U_MAX, -U_MAX]) for xi in x])
        x = competition(x, u) + rng.uniform(-W, W, x.shape)
        assert env.contains(x).all()
        traj.append(x.copy())
    print("  closed loop, 300 starts x 40 steps: 0 violations")
    return env, X, np.array(traj)


def chain(x):
    return 0.5 * x + 0.1 * np.sin(np.roll(x, 1, 1)) + 0.1 * np.cos(np.roll(x, -1, 1))


def chain_example(n=10, N=1000, seed=1):
    rng = np.random.default_rng(seed)
    jx_lo = np.zeros((n, n))
    jx_hi = np.zeros((n, n))
    for i in range(n):
        jx_lo[i, i] = jx_hi[i, i] = 0.5
        for j in ((i - 1) % n, (i + 1) % n):
            jx_lo[i, j], jx_hi[i, j] = -0.1, 0.1
    X = rng.uniform(-2, 2, (N, n))
    t0 = time.perf_counter()
    env = embedding_envelope(X, chain(X), None, jx_lo, jx_hi)
    dt = time.perf_counter() - t0
    print(f"10-D chain: {N} transitions, {env.n_pairs:,} pairs in {dt:.2f} s -> "
          f"{env.n_boxes} invariant boxes")
    for k in range(env.n_boxes):  # check: random states of each box stay in it for 50 steps
        x = rng.uniform(env.lo[k], env.hi[k], (2000, n))
        for _ in range(50):
            x = chain(x)
            assert np.all((x >= env.lo[k]) & (x <= env.hi[k]))
    print("  every box checked: 2,000 random states x 50 steps stay inside")
    return env


def plot(env2, X, traj, env10):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    g = np.linspace(LO, HI, 400)
    G1, G2 = np.meshgrid(g, g)
    M = env2.contains(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
    a.contourf(G1, G2, M, levels=[0.5, 1.5], colors=["#d1495b"], alpha=0.25)
    a.contour(G1, G2, M, levels=[0.5], colors=["#d1495b"], linewidths=1.5)
    a.plot(X[:, 0], X[:, 1], ".", ms=2.5, color="#555", label="data")
    for t in traj.transpose(1, 0, 2)[:25]:
        a.plot(t[:, 0], t[:, 1], lw=0.7, color="#2e5e8c", alpha=0.7)
    a.fill([], [], color="#d1495b", alpha=0.4, label="certified envelope (union of boxes)")
    a.plot([], [], color="#2e5e8c", label="closed loop")
    a.set_xlim(LO, HI)
    a.set_ylim(LO, HI)
    a.set_xlabel("species 1")
    a.set_ylabel("species 2")
    a.set_title("Competition model, 400 transitions")
    a.legend(loc="lower left", fontsize=8)
    n = env10.lo.shape[1]
    i = np.arange(n)
    b.fill_between(i, env10.lo.min(0), env10.hi.max(0), color="#d1495b", alpha=0.12,
                   label="union of certified boxes")
    for lo, hi in zip(env10.lo, env10.hi):
        b.plot(i, hi, color="#d1495b", lw=1)
        b.plot(i, lo, color="#2e5e8c", lw=1)
    b.set_xticks(i)
    b.set_xlabel("unit i")
    b.set_ylabel("x_i")
    b.set_title("Ten coupled units: certified boxes (upper / lower bounds)")
    b.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("envelope_nonlinear.png", dpi=150)
    print("figure: envelope_nonlinear.png")


if __name__ == "__main__":
    env2, X, traj = competition_example()
    env10 = chain_example()
    try:
        plot(env2, X, traj, env10)
    except ImportError:
        pass
