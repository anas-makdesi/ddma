"""Safety for a non-monotone system: an inverted pendulum learned from data.

    theta+ = theta + T omega                                (known)
    omega+ = omega + T (a sin(theta) - b omega + c u) + d    (unknown, learned)

On theta in [-2, 2] (around upright) the unknown part is not monotone in theta
(cos theta changes sign), so the monotone models do not apply directly. Its
derivatives are bounded, which is all the bounded-derivative models need:

    d/dtheta in [T a cos 2, T a],  d/domega = -T b,  d/du = T c.

The five approaches of the thesis (Chapter 3) are learned from the same 10^6
transitions on 101 x 101 state cells and 25 input cells, and compared on the size of
the maximal controlled invariant set inside the box [-2, 2] x [-4, 4] (|u| <= 6, so
the pendulum cannot be held beyond about 0.6 rad); each controller is then run for
300 steps on the true pendulum from 100 random states of its invariant set.

Run:  python examples/pendulum_bounded_derivatives.py
"""

import time

import numpy as np

from ddma import (
    BoundedDerivativeModel,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    safety,
    simulate,
)

T, A, B, C, D = 0.05, 9.81, 0.5, 1.0, 0.01
X_LO, X_HI = np.array([-2.0, -4.0]), np.array([2.0, 4.0])
U_MAX = 6.0


def unknown(X, U):
    return T * (A * np.sin(X[:, 0]) - B * X[:, 1] + C * U[:, 0])


def known(X, U):
    return np.stack([X[:, 0] + T * X[:, 1], X[:, 1]], axis=1)


def step(x, u, rng):
    X, Ul = x[None], np.atleast_1d(u)[None]
    return (known(X, Ul) + np.stack([0 * X[:, 0], unknown(X, Ul)], 1))[0] + [
        0.0, rng.uniform(-D, D)]


def main(n_data=10**6, cells=(101, 101), n_u=25, methods=None, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(X_LO, X_HI, (n_data, 2))
    U = rng.uniform(-U_MAX, U_MAX, (n_data, 1))
    Xn = known(X, U) + np.stack([np.zeros(n_data), unknown(X, U)], 1)
    Xn[:, 1] += rng.uniform(-D, D, n_data)

    Px = RectangularPartition.uniform(X_LO, X_HI, cells)
    Pu = RectangularPartition.uniform([-U_MAX], [U_MAX], [n_u])
    P = RectangularPartition.product(Px, Pu)
    # derivative bounds of the unknown part f = (0, f_omega)
    A_lo = np.array([[0.0, 0.0], [T * A * np.cos(2.0), -T * B]])
    A_hi = np.array([[0.0, 0.0], [T * A, -T * B]])
    B_lo = np.array([[0.0], [T * C]])
    B_hi = B_lo.copy()
    g_bounds = affine_bounds([[1.0, T], [0.0, 1.0]], np.zeros((2, 1)))

    methods = methods or ["two_matrix", "all_matrix", "lin_mono", "growth", "lin_growth"]
    results = {}
    print(f"{n_data:.0e} transitions, {Px.n_cells} state cells x {Pu.n_cells} inputs")
    print(f"{'method':12s} {'model [s]':>9s} {'abstr. [s]':>10s} {'safety [s]':>10s} "
          f"{'invariant (% of cells)':>23s}  closed loop")
    for m in methods:
        t0 = time.perf_counter()
        M = BoundedDerivativeModel.from_data(
            X, U, Xn, P, A_lo, A_hi, B_lo, B_hi, [0.0, -D], [0.0, D],
            g=known, g_bounds=g_bounds, features=[[1], [0, 1, 2]], method=m)
        t1 = time.perf_counter()
        abst = SymbolicAbstraction(Px, Pu, M)
        t2 = time.perf_counter()
        ctrl = safety(abst, np.ones(Px.n_cells, bool))
        t3 = time.perf_counter()
        # closed loop on the true pendulum from random states of the invariant set
        viol = 0
        dom = np.flatnonzero(ctrl.domain)
        starts = Px.all_indices()[rng.choice(dom, 100)] if dom.size else np.zeros((0, 2), int)
        lo, hi = Px.cell_bounds(starts)
        for x0 in rng.uniform(lo, hi):
            xs, us = simulate(step, ctrl, x0, 300, rng=rng, pick="random")
            viol += len(us) < 300 or not np.all(Px.contains(xs))
        results[m] = (ctrl, t1 - t0, t2 - t1, t3 - t2)
        print(f"{m:12s} {t1 - t0:9.2f} {t2 - t1:10.2f} {t3 - t2:10.2f} "
              f"{100 * ctrl.winning_fraction():22.1f}%  {viol} violations / 100 runs")
    return results, Px


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    res, Px = main()
    c = [0.5 * (e[1:] + e[:-1]) for e in Px.edges]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"two_matrix": "#9b9b9b", "all_matrix": "#e9c46a", "lin_mono": "#f4a261",
              "growth": "#2a9d8f", "lin_growth": "#264653"}
    for m, (ctrl, *_) in res.items():
        ax.contour(c[0], c[1], ctrl.domain_grid().T.astype(float), levels=[0.5],
                   colors=colors[m], linewidths=1.5)
        ax.plot([], [], color=colors[m], label=f"{m}: {100 * ctrl.winning_fraction():.1f} %")
    ax.set(xlabel=r"$\theta$ (rad)", ylabel=r"$\omega$ (rad/s)",
           title="Maximal controlled invariant set of the learned pendulum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("pendulum_invariant_sets.png", dpi=150)
    print("saved pendulum_invariant_sets.png")
