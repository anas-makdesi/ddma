---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 1. The tightest model consistent with data

**What you'll learn:** what ddma means by a *model with guarantees*, on the smallest possible
example: one unknown scalar function. Everything later in the toolbox (abstractions,
controllers, envelopes) is built on the object introduced here.

**Main calls:** {class}`~ddma.MinimalSimulatingMap`, {class}`~ddma.PartitionedSimulatingMap`,
{class}`~ddma.RectangularPartition`.

## The problem

Somebody measures an unknown function $f$ at a few points, and every measurement is corrupted
by a disturbance $d \in [-0.5, 0.5]$:

$$
y_k = f(z_k) + d_k, \qquad d_k \in [d_{lo}, d_{hi}] = [-0.5, 0.5].
$$

We don't know a formula for $f$, so we can't fit parameters. What we do know is that
**$f$ is increasing** (monotone). The question ddma answers: *given only this, where can
$f(z) + d$ be, for any $z$?* The answer must hold for every monotone function that could have
produced the data, because we can't tell them apart, so it holds for the true one.

To check the answers, we secretly use $f(z) = z^2$ on $[0, 3]$ (the scalar example of the
thesis). ddma never sees this formula.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import ddma
from ddma import MinimalSimulatingMap, PartitionedSimulatingMap, RectangularPartition

rng = np.random.default_rng(0)
d_lo, d_hi = -0.5, 0.5

def f(z):              # the unknown function: used only to make data and to check results
    return z**2

Z = rng.uniform(0, 3, size=(25, 1))                 # 25 inputs z_k (one column: n = 1)
Y = f(Z) + rng.uniform(d_lo, d_hi, size=Z.shape)    # 25 noisy outputs y_k (m = 1)
Z.shape, Y.shape
```

Data are always 2-D arrays: one row per sample, one column per input (`Z`) or output (`Y`).

## The minimal simulating map

{class}`~ddma.MinimalSimulatingMap` takes the data and the disturbance bounds. Nothing else:
no grid, no parameters.

```{code-cell} ipython3
S = MinimalSimulatingMap(Z, Y, d_lo, d_hi)
lo, hi = S.bounds([[1.5]])
print(f"every consistent system maps z = 1.5 into [{lo[0, 0]:.3f}, {hi[0, 0]:.3f}]")
print(f"the truth: f(1.5) + d lies in [{f(1.5) + d_lo:.3f}, {f(1.5) + d_hi:.3f}]")
```

Where do these two numbers come from? Take a data point $z_k \le 1.5$. Because $f$ is
increasing, $f(1.5) \ge f(z_k)$. We don't know $f(z_k)$ exactly, only that
$y_k = f(z_k) + d_k$ with $d_k \le d_{hi}$, so $f(z_k) \ge y_k - d_{hi}$. The new measurement
at 1.5 adds its own $d \ge d_{lo}$. Together:

$$
f(1.5) + d \;\ge\; y_k - d_{hi} + d_{lo} = y_k - \theta, \qquad \theta = d_{hi} - d_{lo}.
$$

Every point to the left gives a **floor**; the best floor is the largest. Symmetrically every
point to the right gives a **ceiling** $y_k + \theta$. Let's recompute both by hand:

```{code-cell} ipython3
theta = d_hi - d_lo
left, right = Z[:, 0] <= 1.5, Z[:, 0] >= 1.5
floor = (Y[left, 0] - theta).max()
ceiling = (Y[right, 0] + theta).min()
print(f"by hand: [{floor:.3f}, {ceiling:.3f}]")
```

That is the whole definition (TAC paper, Theorem 1):

$$
S_m(z) = \Big[\, \max_{z_k \le z} (y_k - \theta),\; \min_{z_k \ge z} (y_k + \theta) \,\Big].
$$

In more dimensions "$\le$" is componentwise, so a point bounds $z$ only if it is below (or
above) $z$ in *every* coordinate. With no point on one side, that bound is infinite.

Two facts make $S_m$ the right object (both proved in the TAC paper):

- **Sound:** it contains $f(z) + D$ for every monotone $f$ consistent with the data.
- **Minimal:** for every value inside $S_m(z)$ there is a monotone $f$ consistent with the
  data that reaches it. So no method can give a tighter guarantee from the same information.

## The picture

```{code-cell} ipython3
zs = np.linspace(0, 3, 1201)[:, None]
lo, hi = S.bounds(zs)

def show(ax, zs, lo, hi, label, color, alpha=0.45, ylim=(-1.5, 10.5)):
    ax.fill_between(zs[:, 0], np.clip(lo[:, 0], *ylim), np.clip(hi[:, 0], *ylim),
                    color=color, alpha=alpha, label=label, step="post")

fig, ax = plt.subplots(figsize=(7, 4))
show(ax, zs, lo, hi, "minimal simulating map $S_m$", "#2a9d8f")
ax.plot(zs, f(zs) + d_lo, "k--", lw=1)
ax.plot(zs, f(zs) + d_hi, "k--", lw=1, label="true $f(z) + [-0.5, 0.5]$")
ax.plot(Z, Y, "o", color="#264653", ms=4, label="data")
ax.set(xlabel="$z$", ylabel="$y$", xlim=(0, 3), ylim=(-1.5, 10.5))
ax.legend(loc="upper left", frameon=False, fontsize=9);
```

The band is a **staircase**: it only changes value at data points. Near $z = 0$ there is no
data point to the left, so the lower bound is $-\infty$ (clipped in the plot); near $z = 3$ the
upper bound is $+\infty$ for the same reason.

Soundness is easy to test: draw many $z$ and disturbances and check that $f(z) + d$ always
lands inside. (ddma's test suite does this for random monotone functions in several
dimensions.)

```{code-cell} ipython3
zt = rng.uniform(0, 3, (100_000, 1))
yt = f(zt) + rng.uniform(d_lo, d_hi, zt.shape)
lo_t, hi_t = S.bounds(zt)
print("all inside:", bool(np.all((lo_t <= yt) & (yt <= hi_t))))
```

## Bounds on $f$ itself

`bounds` answers "where can the next *measurement* be": it bounds $f(z) + d$. Sometimes you
want $f(z)$ itself. Removing the disturbance gives a band narrower by $\theta$ (where it is
bounded on both sides): {meth}`~ddma.MinimalSimulatingMap.f_bounds`, and its midpoint
{meth}`~ddma.MinimalSimulatingMap.estimate` is a set-membership estimate of $f$.

```{code-cell} ipython3
flo, fhi = S.f_bounds([[1.5]])
print(f"f(1.5) in [{flo[0, 0]:.3f}, {fhi[0, 0]:.3f}]  (true value {f(1.5)}),"
      f" estimate {S.estimate([[1.5]])[0, 0]:.3f}")
```

## More data, tighter model, down to a limit

```{code-cell} ipython3
inner = (zs[:, 0] > 0.5) & (zs[:, 0] < 2.5)
for K in (25, 250, 2_500, 25_000):
    Zk = rng.uniform(0, 3, (K, 1))
    Yk = f(Zk) + rng.uniform(d_lo, d_hi, Zk.shape)
    l, h = MinimalSimulatingMap(Zk, Yk, d_lo, d_hi).bounds(zs)
    print(f"{K:>6} samples: mean width of S_m on [0.5, 2.5] = {np.mean((h - l)[inner]):.3f}")
```

The width falls towards $\theta = 1$, never below: $f(z) + D$ is itself an interval of width
1, so a model of it can't be thinner. (The width of `f_bounds`, by contrast, goes to 0.)

## The same model on a grid

The exact map costs $O(K)$ per query: fine for plots, too slow inside a controller synthesis
that asks about millions of cells. ddma therefore also computes the tightest bound **per cell
of a grid**, once for all cells. A grid is a {class}`~ddma.RectangularPartition`; here 8
uniform cells on $[0, 3]$:

```{code-cell} ipython3
grid = RectangularPartition.uniform([0], [3], 8)
sigma = PartitionedSimulatingMap.from_data(Z, Y, grid, d_lo, d_hi)
print(grid)
print("lower bounds per cell:", np.round(sigma.lower[:, 0], 2))
print("upper bounds per cell:", np.round(sigma.upper[:, 0], 2))
```

`sigma` (the paper's $\sigma$) holds one interval per cell that contains $S_m(z)$ for every
$z$ in the cell. A cell is wider than a point, so the bound is looser: the lower bound of a
cell can only use data points left of the **whole** cell. The coarser the grid, the looser
the model:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for cells, color in ((8, "#f2c14e"), (40, "#e76f51")):
    s = PartitionedSimulatingMap.from_data(Z, Y, RectangularPartition.uniform([0], [3], cells),
                                           d_lo, d_hi)
    show(ax, zs, *s.bounds(zs), f"{cells} cells", color, alpha=0.3)
show(ax, zs, lo, hi, "exact $S_m$", "#2a9d8f")
ax.plot(Z, Y, "o", color="#264653", ms=4)
ax.set(xlabel="$z$", ylabel="$y$", xlim=(0, 3), ylim=(-1.5, 10.5))
ax.legend(loc="upper left", frameon=False, fontsize=9);
```

The fit takes $O(K \log |Q| + |Q|)$ time for $|Q|$ cells (one sweep over the grid, in C++,
on all cores), which is why ddma can learn from millions of samples on grids with millions
of cells. And if the grid's edges are exactly the data points
({meth}`~ddma.RectangularPartition.from_data`), the grid map *is* the exact map:

```{code-cell} ipython3
exact_grid = RectangularPartition.from_data(Z, [0], [3])
s = PartitionedSimulatingMap.from_data(Z, Y, exact_grid, d_lo, d_hi)
zq = rng.uniform(0, 3, (1000, 1))
print("equal to S_m:", np.allclose(np.r_[s.bounds(zq)], np.r_[S.bounds(zq)]))
```

## New data: intersect, don't refit

New measurements can only remove functions from the set of consistent ones, so the new model
is the old one intersected with the model of the new data.
{meth}`~ddma.PartitionedSimulatingMap.update` does this without revisiting the old data, and
gives exactly the model of all the data:

```{code-cell} ipython3
Z2 = rng.uniform(0, 3, (25, 1))
Y2 = f(Z2) + rng.uniform(d_lo, d_hi, Z2.shape)
updated = sigma.update(Z2, Y2)
refit = PartitionedSimulatingMap.from_data(np.vstack([Z, Z2]), np.vstack([Y, Y2]), grid,
                                           d_lo, d_hi)
print("update == refit:", np.array_equal(updated.lower, refit.lower)
      and np.array_equal(updated.upper, refit.upper))
```

This is what makes online learning cheap (tutorial 4).

## Decreasing, or not dependent at all: `signs`

Monotone does not have to mean increasing. `signs` gives, per output and input, $+1$
(increasing), $-1$ (decreasing) or $0$ (no dependence, so any data point bounds every $z$).
For $f(z) = 3 e^{-z}$:

```{code-cell} ipython3
Yd = 3 * np.exp(-Z) + rng.uniform(d_lo, d_hi, Z.shape)
Sd = MinimalSimulatingMap(Z, Yd, d_lo, d_hi, signs=[-1])
print("bounds at z = 1:", np.round(np.r_[Sd.bounds([[1.0]])].ravel(), 3),
      " truth:", np.round([3 * np.exp(-1) + d_lo, 3 * np.exp(-1) + d_hi], 3))
```

## When the assumption is wrong

Every guarantee rests on the assumption. Some violations are visible in the data: if a data
point lies *outside* the map built from the others, no monotone function with these
disturbance bounds can explain the data.
{meth}`~ddma.MinimalSimulatingMap.is_consistent` checks this:

```{code-cell} ipython3
Ys = 3 * np.sin(2 * Z) + rng.uniform(d_lo, d_hi, Z.shape)        # not monotone on [0, 3]
print("z^2 data consistent with 'increasing':  ", S.is_consistent())
print("sin data consistent with 'increasing':  ", MinimalSimulatingMap(Z, Ys, d_lo, d_hi).is_consistent())
```

A `True` does not prove the assumption (the data may just not have caught the violation yet),
but a `False` disproves it.

## Summary

- From data and one structural assumption, ddma computes the **tightest guaranteed model**:
  sound (contains the truth) and minimal (nothing tighter is justified).
- The exact map ({class}`~ddma.MinimalSimulatingMap`) is for queries at points; the grid map
  ({class}`~ddma.PartitionedSimulatingMap`) is computed once for all cells and is what the
  abstractions use.
- New data refine the model by intersection (`update`), with no refit.

Next: {doc}`02-cruise-control` uses a grid map of a vehicle's speed dynamics to certify a
safe controller.

## In MATLAB

::::{tab-set}
:::{tab-item} MATLAB
```matlab
ddma.seed(0);
Z = 3 * rand(25, 1);  Y = Z.^2 + (rand(25, 1) - 0.5);
[lo, hi] = ddma.msm_bounds(Z, Y, -0.5, 0.5, 1.5);       % exact map at z = 1.5
[flo, fhi] = ddma.msm_f_bounds(Z, Y, -0.5, 0.5, 1.5);   % bounds on f itself
ok = ddma.msm_is_consistent(Z, Y, -0.5, 0.5);

edges = ddma.uniform_partition(0, 3, 8);                  % 8 cells on [0, 3]
S = ddma.fit(Z, Y, edges, -0.5, 0.5);                    % S.lower, S.upper: one row per cell
Z2 = 3 * rand(25, 1);  Y2 = Z2.^2 + (rand(25, 1) - 0.5);
S = ddma.fit_update(S, Z2, Y2);                          % refine with new data
[lo, hi] = ddma.bounds(S, linspace(0, 3, 300)');
```
:::
:::{tab-item} Python
```python
S = MinimalSimulatingMap(Z, Y, -0.5, 0.5)
lo, hi = S.bounds([[1.5]])
flo, fhi = S.f_bounds([[1.5]])
ok = S.is_consistent()

grid = RectangularPartition.uniform([0], [3], 8)
sigma = PartitionedSimulatingMap.from_data(Z, Y, grid, -0.5, 0.5)
sigma = sigma.update(Z2, Y2)
lo, hi = sigma.bounds(np.linspace(0, 3, 300)[:, None])
```
:::
::::

The full script with the figure of the README: `examples/scalar_map.py` (Python) and
`matlab/examples/scalar_map.m`.
