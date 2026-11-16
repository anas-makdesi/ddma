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

# 4. Learning while running

**What you'll learn:** how to start with little data, run the system safely, and let the
certified safe region grow as measurements come in, without ever losing the guarantee. Then
how *lazy* abstractions avoid computing transitions the synthesis never needs.

**Main calls:** {meth}`~ddma.BoundedDerivativeModel.update_local`,
{meth}`~ddma.BoundedDerivativeModel.update`, {meth}`~ddma.SymbolicAbstraction.refresh`,
`safety(..., known_winning=)`, `SymbolicAbstraction(..., lazy=True)`.

## Why online refinement is safe

Three facts from the earlier tutorials combine into a simple loop:

1. New data **only shrink** the model (tutorial 1: the new model is an intersection).
2. Smaller model intervals give **smaller successor boxes** in the abstraction.
3. So a set that was controlled invariant before the update **stays** controlled invariant
   after it: the inputs that kept it inside still do. The new maximal safe set contains the
   old one.

So you can run the current certified controller, learn from what you observe, and refine. The
safe region can only grow, and the system never leaves it. The cost of each refinement
should be small, which is what the three calls of this tutorial are for:

| call | what it does | cost |
|---|---|---|
| `model.update_local(x, u, x_next, window)` | tightens the growth-cone model only on the cells within `window` cells of the new point | tiny; meant for every step |
| `model.update(X, U, X_next)` | exact batch update: same model as a refit on all the data | a fit; every few hundred steps |
| `abst.refresh()` | recomputes the transitions of the state cells whose model changed (only those) | proportional to what changed |
| `safety(abst, safe, known_winning=old)` | recomputes the maximal safe set, keeping the old one without re-testing it | cheaper than from scratch, same result |

## The setting

A damped pendulum near the bottom (not upright this time), with $\theta^+ = \theta + T\omega$
known and the velocity update unknown, learned with **loose** derivative bounds. The
specification is $|\theta| \le 0.8$, $|\omega| \le 1.2$. Initial data come only from a small
region near the bottom ($|\theta| \le 0.15$), as if from a cautious first experiment.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np

from ddma import (BoundedDerivativeModel, RectangularPartition, SymbolicAbstraction,
                  affine_bounds, safety)

T, D = 0.1, 0.005
def f_unknown(X, U):                    # used only to simulate the plant
    return T * (-np.sin(X[:, 0]) - 0.2 * X[:, 1] + U[:, 0])

def known(X, U):
    return np.stack([X[:, 0] + T * X[:, 1], X[:, 1]], axis=1)

def plant(x, u, rng):
    X, U = x[None], np.atleast_1d(u)[None]
    return known(X, U)[0] + [0.0, f_unknown(X, U)[0] + rng.uniform(-D, D)]

Px = RectangularPartition.uniform([-1, -1.5], [1, 1.5], [60, 60])
Pu = RectangularPartition.uniform([-2], [2], [7])
P = RectangularPartition.product(Px, Pu)
# loose derivative bounds of f = (0, f_omega) w.r.t. (theta, omega) and u
A_lo, A_hi = np.array([[0, 0], [-0.3, -0.1]]), np.array([[0, 0], [0.1, 0.05]])
B_lo, B_hi = np.array([[0], [0.05]]), np.array([[0], [0.15]])
g_bounds = affine_bounds([[1, T], [0, 1]], np.zeros((2, 1)))

rng = np.random.default_rng(1)
K, r0 = 2000, 0.15
X0 = np.column_stack([rng.uniform(-r0, r0, K), rng.uniform(-1.5 * r0, 1.5 * r0, K)])
U0 = rng.uniform(-2, 2, (K, 1))
Xn0 = np.array([plant(x, u, rng) for x, u in zip(X0, U0)])

model = BoundedDerivativeModel.from_data(X0, U0, Xn0, P, A_lo, A_hi, B_lo, B_hi,
                                         d_lo=[0, -D], d_hi=[0, D], g=known, g_bounds=g_bounds,
                                         method="growth", local="slab")
abst = SymbolicAbstraction(Px, Pu, model)
safe = abst.cells_inside([-0.8, -1.2], [0.8, 1.2])
ctrl = safety(abst, safe)
initial = ctrl.domain.copy()
print(f"initial model ({K} samples near the bottom): "
      f"{100 * ctrl.domain.sum() / safe.sum():.1f} % of the specification certified")
```

`local="slab"` makes the growth model also use the cones of the data points *inside* each
cell's slabs, which is tighter for data that are dense locally, as they are here.

## The loop

At each step: pick a random allowed input (this explores the edge of the safe region, the
worst case for safety and the best case for learning), apply it to the real plant, measure,
and update the model locally, on the cells within `window` cells of the measurement. Every 20
steps, refresh the abstraction where the model changed and update the safe set
incrementally. Every `batch_every` steps, apply the exact batch update with the measurements
collected since the last one.

The model, abstraction and controller are rebuilt at the start of each run so that we can
compare settings.

```{code-cell} ipython3
def run(window=3, batch_every=200, steps=1000, seed=1):
    rng = np.random.default_rng(seed)
    model = BoundedDerivativeModel.from_data(X0, U0, Xn0, P, A_lo, A_hi, B_lo, B_hi,
                                             d_lo=[0, -D], d_hi=[0, D], g=known,
                                             g_bounds=g_bounds, method="growth", local="slab")
    abst = SymbolicAbstraction(Px, Pu, model)
    ctrl = safety(abst, safe)
    dom = np.flatnonzero(ctrl.domain)
    lo, hi = Px.cell_bounds(Px.all_indices()[[rng.choice(dom)]])
    x = rng.uniform(lo[0], hi[0])
    batch, history, violations, t_refine = [], [], 0, 0.0
    for k in range(1, steps + 1):
        u = ctrl(x, pick="random", rng=rng)
        if u is None:                      # outside the certified set: cannot happen
            violations += 1
            break
        x_next = plant(x, u, rng)
        model.update_local(x[None], u[None], x_next[None], window=window)   # every step
        batch.append((x, u, x_next))
        violations += bool(np.any(np.abs(x_next) > [0.8, 1.2]))
        x = x_next
        if k % 20 == 0:
            t0 = time.perf_counter()
            if k % batch_every == 0:       # exact batch update: every state may change
                model.update(*(np.array(v) for v in zip(*batch)))
                batch = []
                abst.refresh(np.ones(Px.n_cells, bool))
                model.changed_boxes = []
            else:
                abst.refresh()             # only the cells update_local touched
            old = ctrl.domain
            ctrl = safety(abst, safe, known_winning=old)
            assert np.all(ctrl.domain[old])     # the safe set never shrinks
            t_refine += time.perf_counter() - t0
            history.append((k, 100 * ctrl.domain.sum() / safe.sum()))
    return model, abst, ctrl, np.array(history), violations, t_refine

model, abst, ctrl, history, violations, t_refine = run(window=3, batch_every=200)
print(f"1000 steps, {violations} violations, total refinement time {t_refine:.2f} s")
for k, pct in history[9::10]:
    print(f"step {k:4.0f}: {pct:5.1f} % certified")
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(history[:, 0], history[:, 1], color="#2a9d8f")
ax[0].set(xlabel="step", ylabel="% of the specification certified",
          title="The safe region only grows")
c = [0.5 * (e[1:] + e[:-1]) for e in Px.edges]
ax[1].contourf(c[0], c[1], ctrl.domain_grid().T.astype(float), levels=[0.5, 1.5],
               colors=["#2a9d8f"], alpha=0.35)
ax[1].contour(c[0], c[1], initial.reshape(Px.shape).T.astype(float), levels=[0.5],
              colors=["#e76f51"])
ax[1].plot([], [], color="#e76f51", label="initial")
ax[1].plot([], [], "s", color="#2a9d8f", alpha=0.5, label="after 1000 steps")
ax[1].add_patch(plt.Rectangle((-0.8, -1.2), 1.6, 2.4, fill=False, ls="--", ec="k"))
ax[1].set(xlabel=r"$\theta$", ylabel=r"$\omega$", title="Certified safe region")
ax[1].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout();
```

What happened: each measurement tightened the model around where the pendulum was, which is
at the edge of the current safe region (the random allowed input keeps pushing there). The
refreshed abstraction then certified a little more, the controller allowed a little more,
and the pendulum explored further, always inside a set that was already certified. The
steps at 200 and 400 are the exact batch updates: there each measurement spreads its
information over the whole grid, not only over a window of cells. After about 800 steps the
growth stops: the loose derivative bounds, not the data, are now the limit.

## How big a window?

`update_local` only touches the cells within `window` cells of the measurement. That is what
makes it cheap, and it also limits what it can teach. Without batch updates:

```{code-cell} ipython3
for window in (1, 3, 8):
    *_, h, v, t = run(window=window, batch_every=10**9)
    print(f"window = {window}: {h[0, 1]:5.1f} % -> {h[-1, 1]:5.1f} % certified "
          f"after 1000 steps ({v} violations, refinement {t:.2f} s)")
```

With `window=1` the local updates change the model, but not where it matters: the
transitions that decide whether a cell at the edge of the set is safe start from that cell
and its neighbours, a few cells away from where the pendulum measured. A window of a few
cells is the useful range here (on this small grid even `window=8` costs almost nothing);
the batch update gets everything, at the price of a full refresh. A practical pattern is the one above: local updates every step, an exact batch
update now and then.

## Checking the shortcuts

`refresh()` recomputes only the changed rows, and `known_winning` skips re-testing the old
set. Both are exact, which is easy to check against a rebuild from scratch:

```{code-cell} ipython3
fresh = SymbolicAbstraction(Px, Pu, model)
same_abst = (np.array_equal(fresh.enabled, abst.enabled)
             and np.array_equal(fresh.first[fresh.enabled], abst.first[abst.enabled])
             and np.array_equal(fresh.last[fresh.enabled], abst.last[abst.enabled]))
print("refreshed abstraction == rebuilt:", same_abst)
print("incremental safe set == from scratch:",
      np.array_equal(safety(fresh, safe).domain, ctrl.domain))
```

(ddma's tests do the same comparison for every backend and for lazy abstractions.)

## Lazy abstractions: compute only what synthesis needs

An eager abstraction stores a successor box for every (state cell, input cell) pair:
$|Q| \cdot |P|$ boxes, most of which the safety synthesis never looks at. Cells outside the
safe set are never asked about, and a safe cell is settled as soon as *one* good input is
found. With `lazy=True`, nothing is precomputed: {func}`~ddma.safety` evaluates a transition
only when its scan reaches it. Memory drops from $O(|Q||P|)$ to $O(|Q|)$, and the result is
the same.

Example: three rooms in a row, one shared cooling input, unknown heat gains in
$[0, 0.15]$ °C per step, every room must stay at or below 26 °C. The plant is monotone:
increasing in the temperatures and **decreasing** in the cooling input
(`signs = [1, 1, 1, -1]`).

```{code-cell} ipython3
from ddma import PartitionedSimulatingMap

def rooms(Tm, u):
    a, b, c = 0.05, 0.3, 0.6
    cpl = np.stack([Tm[:, 1] - Tm[:, 0], Tm[:, 0] + Tm[:, 2] - 2 * Tm[:, 1], Tm[:, 1] - Tm[:, 2]], 1)
    return Tm + 0.25 * (a * (34 - Tm) + b * cpl - c * u * (Tm - 14))

N = 200_000
Tm = rng.uniform(16, 32, (N, 3))
Uc = rng.uniform(0, 1, (N, 1))
Tn = rooms(Tm, Uc) + rng.uniform(0, 0.15, (N, 3))
Xp = RectangularPartition.uniform([16] * 3, [32] * 3, 40)
Up = RectangularPartition.uniform([0.0], [1.0], 21)
sigma = PartitionedSimulatingMap.from_data(np.hstack([Tm, Uc]), Tn,
                                           RectangularPartition.product(Xp, Up),
                                           signs=[[1, 1, 1, -1]] * 3)
for lazy in (False, True):
    t0 = time.perf_counter()
    A = SymbolicAbstraction(Xp, Up, sigma, lazy=lazy)
    ok = A.state_mask(lambda lo, hi: np.all(hi <= 26, axis=1))
    c_rooms = safety(A, ok)
    print(f"{'lazy ' if lazy else 'eager'}: {time.perf_counter() - t0:5.2f} s, "
          f"transitions evaluated {A.n_evaluated:>9,} of {A.n_states * A.n_inputs:,}, "
          f"winning {100 * c_rooms.domain.sum() / ok.sum():.1f} % of the safe cells")
print("least cooling at (24, 25, 24.5):", c_rooms([24, 25, 24.5], pick="min_norm"))
```

The lazy controller computes its allowed inputs on demand when you query it. Use lazy
abstractions for large grids and safety specifications; `reach_avoid` needs the eager one.

## Summary

- Refinement only shrinks the model, so **certified sets stay certified**; the safe region
  grows as the system runs, and the loop never leaves it.
- `update_local` (every step) + `refresh` (only changed cells) + `known_winning` (keep the old
  set) make each refinement cheap; `update` is the exact batch version.
- `lazy=True` evaluates only the transitions the safety synthesis needs.

Next: {doc}`05-rl-safety-filter` puts a learning agent inside the certified set.

## In MATLAB

`matlab/examples/online_refinement.m` is the MATLAB version of the loop above.

::::{tab-set}
:::{tab-item} MATLAB
```matlab
T = 0.1; D = 0.005;
f = @(X, U) T * (-sin(X(:, 1)) - 0.2 * X(:, 2) + U(:, 1));
A_G = [1 T; 0 1];
Xedges = ddma.uniform_partition([-1 -1.5], [1 1.5], [40 40]);
Uedges = ddma.uniform_partition(-2, 2, 7);
edges = [Xedges(:); Uedges(:)]';
g_lo = [0 0 0; -0.3, -0.1, 0.05];  g_hi = [0 0 0; 0.1, 0.05, 0.15];
ddma.seed(1); K = 2000;
X = [0.15 * (2 * rand(K, 1) - 1), 0.225 * (2 * rand(K, 1) - 1)]; U = 4 * rand(K, 1) - 2;
Y = [zeros(K, 1), f(X, U) + D * (2 * rand(K, 1) - 1)];
S = ddma.bd_fit([X U], Y, edges, g_lo, g_hi, [0 -D], [0 D], 'growth', 'slab');
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G));
safe = ddma.cells_inside(Xedges, [-0.8 -1.2], [0.8 1.2]);
[domain, allowed] = ddma.safety(A, safe);
% one online step
x = [0.05 0]; u = ddma.controller_input(A, allowed, x, 'random');
xn = x * A_G' + [0, f(x, u) + D * (2 * rand - 1)];
S = ddma.bd_update_local(S, [x u], [0, xn(2) - x * A_G(2, :)'], 3);
[A, S] = ddma.abstraction_refresh(A, S);                 % only the changed cells
[domain, allowed] = ddma.safety(A, safe, [], domain);     % known_winning = domain
fprintf('%.1f %% certified\n', 100 * nnz(domain) / nnz(safe));
```
:::
:::{tab-item} Python
```python
model.update_local(x[None], u[None], x_next[None], window=3)
abst.refresh()
ctrl = safety(abst, safe, known_winning=ctrl.domain)
```
:::
::::
