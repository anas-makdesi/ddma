"""Short demo for the README and the website: the guaranteed band of a monotone map
tightening as data arrives.

An unknown increasing function is measured with bounded noise. After every new batch of
samples, ddma's minimal simulating map gives the tightest band that is consistent with all
the data and the monotonicity assumption. The true set of possible outputs never leaves it.

Run:  python examples/demo_gif.py   (writes docs/figures/demo.gif)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ddma import MinimalSimulatingMap  # noqa: E402

INK, TEAL, MUTED, ORANGE = "#0B1220", "#2a9d8f", "#6B7280", "#FB923C"
PAPER, LINE = "#F7F7F5", "#D4D4CF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

D_LO, D_HI = -0.4, 0.4          # noise bounds (known)
Z_MAX = 3.0


def f(z):  # unknown to the method; only used to make data and draw the truth
    return 3.0 / (1.0 + np.exp(-3.0 * (z - 1.5))) + 0.6 * z


def main():
    rng = np.random.default_rng(1)
    n_max = 400
    Z = rng.uniform(0, Z_MAX, size=(n_max, 1))
    Y = f(Z) + rng.uniform(D_LO, D_HI, size=Z.shape)

    counts = sorted({int(round(c)) for c in np.geomspace(3, n_max, 46)})
    zs = np.linspace(0, Z_MAX, 600)[:, None]
    inner = (zs[:, 0] > 0.3) & (zs[:, 0] < 2.7)
    ylim = (-1.2, 6.4)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100)
    fig.patch.set_facecolor(PAPER)
    frames = []
    for n in counts:
        model = MinimalSimulatingMap(Z[:n], Y[:n], D_LO, D_HI)
        lo, hi = model.bounds(zs)
        lo, hi = lo[:, 0], hi[:, 0]
        # the true set of outputs [f + d_lo, f + d_hi] must lie inside the band
        assert np.all(lo <= f(zs[:, 0]) + D_LO + 1e-9) and np.all(f(zs[:, 0]) + D_HI <= hi + 1e-9)
        width = float(np.mean((hi - lo)[inner]))
        width_txt = f"band width {width:.2f} (noise alone: {D_HI - D_LO:.2f})" \
            if np.isfinite(width) else "band still unbounded in places"

        ax.clear()
        ax.set_facecolor(PAPER)
        ax.fill_between(zs[:, 0], np.clip(lo, *ylim), np.clip(hi, *ylim), color=TEAL,
                        alpha=0.35, lw=0, label="guaranteed band (ddma)")
        ax.plot(zs, f(zs) + D_LO, color=INK, lw=1, ls="--")
        ax.plot(zs, f(zs) + D_HI, color=INK, lw=1, ls="--", label="true outputs (unknown)")
        ax.plot(Z[:n], Y[:n], "o", color=INK, ms=3, alpha=0.8, label="measured samples")
        ax.set(xlim=(0, Z_MAX), ylim=ylim, xlabel="input", ylabel="output")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(LINE)
        ax.tick_params(colors=MUTED)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.set_title("A model with guaranteed error bounds, learned from data",
                     loc="left", fontweight="bold", color=INK, fontsize=11)
        ax.text(0.99, 0.04, f"{n} samples   ·   {width_txt}\n"
                "true outputs inside the band: always",
                transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=9.5)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    durations = [250] * len(frames)  # ms
    durations[-1] = 3000                         # hold the last frame
    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
              for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"{len(counts)} steps, {counts[0]} to {counts[-1]} samples; saved {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
