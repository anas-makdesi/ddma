"""Minimal simulating map of a scalar monotone map, exact and on a fixed partition.

F(z) = z^2 + [-0.5, 0.5] on [0, 3] (the scalar example used in the thesis chapter).
Run:  python examples/scalar_map.py   (writes docs/figures/scalar_map.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddma import MinimalSimulatingMap, PartitionedSimulatingMap, RectangularPartition

rng = np.random.default_rng(0)
d_lo, d_hi = -0.5, 0.5
def f(z):  # unknown to the method; only used to make data and plots
    return z**2



Z = rng.uniform(0, 3, size=(25, 1))
Y = f(Z) + rng.uniform(d_lo, d_hi, size=Z.shape)

exact = MinimalSimulatingMap(Z, Y, d_lo, d_hi)
partition = RectangularPartition.uniform([0], [3], 8)
coarse = PartitionedSimulatingMap.from_data(Z, Y, partition, d_lo, d_hi)

zs = np.linspace(0, 3, 1201)[:, None]
lo, hi = exact.bounds(zs)
clo, chi = coarse.bounds(zs)
print(f"{len(Z)} samples; mean width of the exact map on [0.5, 2.5]: "
      f"{np.mean((hi - lo)[(zs[:, 0] > 0.5) & (zs[:, 0] < 2.5)]):.3f}")
print(f"mean width on the 8-cell partition (bounded cells): "
      f"{np.mean(coarse.widths()[np.isfinite(coarse.widths())]):.3f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
ylim = (-1.5, 10.5)
ax.fill_between(zs[:, 0], np.clip(clo[:, 0], *ylim), np.clip(chi[:, 0], *ylim), step="post",
                color="#f2c14e", alpha=0.35, label="simulating map on 8 cells")
ax.fill_between(zs[:, 0], np.clip(lo[:, 0], *ylim), np.clip(hi[:, 0], *ylim), step="post",
                color="#2a9d8f", alpha=0.45, label="minimal simulating map $S_m$")
ax.plot(zs, f(zs) + d_lo, "k--", lw=1)
ax.plot(zs, f(zs) + d_hi, "k--", lw=1, label="true $F(z) = z^2 + [-0.5, 0.5]$")
ax.plot(Z, Y, "o", color="#264653", ms=4, label="data")
ax.set(xlabel="$z$", ylabel="$y$", ylim=ylim, xlim=(0, 3))
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("Tightest model consistent with 25 noisy samples of an unknown monotone map",
             fontsize=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "scalar_map.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f"saved {out}")
