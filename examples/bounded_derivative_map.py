"""Models of a scalar map with bounded derivative (not monotone), from 25 noisy samples.

F(z) = sin(z) + 0.3 z + [-0.1, 0.1] on [0, 2 pi], with dF/dz in [-0.7, 1.3]. The map is
not monotone. Compared on 12 cells: the two-matrix model (approach 1 of the thesis,
Chapter 3) and the growth-cone model with a linear-programming final step (approach 5).

Run:  python examples/bounded_derivative_map.py   (writes docs/figures/bounded_derivative_map.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddma import BoundedDerivativeModel, RectangularPartition

rng = np.random.default_rng(1)
D = 0.1
LO, HI = 0.0, 2 * np.pi


def f(z):  # unknown to the method; only used to make data and plots
    return np.sin(z) + 0.3 * z


Z = rng.uniform(1.0, 5.5, size=(25, 1))  # no data near the ends of the interval
Y = f(Z) + rng.uniform(-D, D, size=Z.shape)
U = np.zeros_like(Z)  # this map has no input: one input cell, zero derivative bounds

Pz = RectangularPartition.uniform([LO], [HI], [12])
P = RectangularPartition.product(Pz, RectangularPartition.uniform([0], [1], [1]))
models = {
    m: BoundedDerivativeModel.from_data(Z, U, Y, P, [[-0.7]], [[1.3]], [[0.0]], [[0.0]],
                                        -D, D, method=m)
    for m in ("two_matrix", "lin_growth")
}

zs = np.linspace(LO, HI, 1201)[:, None]
fig, ax = plt.subplots(figsize=(7, 4.2))
ylim = (-2.6, 3.0)
style = {"two_matrix": ("#f2c14e", 0.35, "two-matrix model (approach 1)"),
         "lin_growth": ("#2a9d8f", 0.45, "growth-cone model (approach 5)")}
for m, M in models.items():
    lo, hi = M.bounds(zs, np.zeros_like(zs))
    width = hi[:, 0] - lo[:, 0]
    print(f"{m:11s} bounded on {np.mean(np.isfinite(width)):.0%} of [0, 2 pi], "
          f"mean width where bounded {np.mean(width[np.isfinite(width)]):.3f}")
    c, a, label = style[m]
    ax.fill_between(zs[:, 0], np.clip(lo[:, 0], *ylim), np.clip(hi[:, 0], *ylim), step="post",
                    color=c, alpha=a, label=label)
ax.plot(zs, f(zs) - D, "k--", lw=1)
ax.plot(zs, f(zs) + D, "k--", lw=1, label=r"true $F(z) = \sin z + 0.3 z + [-0.1, 0.1]$")
ax.plot(Z, Y, "o", color="#264653", ms=4, label="data")
ax.set(xlabel="$z$", ylabel="$y$", ylim=ylim, xlim=(LO, HI))
ax.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=2)
ax.set_title("A map that is not monotone, from 25 samples and bounds on its derivative",
             fontsize=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "bounded_derivative_map.png"
fig.savefig(out, dpi=160)
print(f"saved {out}")
