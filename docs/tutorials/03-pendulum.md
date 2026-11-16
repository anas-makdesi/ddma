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

# 3. Systems that are not monotone: bounded derivatives

**What you'll learn:** how ddma models dynamics that are not monotone, from bounds on their
derivatives; the idea of a growth cone; the five methods of `BoundedDerivativeModel` and how
to choose; how to estimate derivative bounds from data. The running example is keeping an
inverted pendulum upright.

**Main calls:** {class}`~ddma.GrowthMap`, {class}`~ddma.BoundedDerivativeModel`,
{func}`~ddma.estimate_derivative_bounds`, {func}`~ddma.scenario_confidence`.

## Why monotone is not enough

An inverted pendulum, angle $\theta$ from upright and angular velocity $\omega$, torque $u$,
$T = 0.05$ s:

$$
\theta^+ = \theta + T\omega \quad\text{(known)}, \qquad
\omega^+ = \omega + T\big(a \sin\theta - b\,\omega + c\,u\big) + d \quad\text{(unknown)}.
$$

On $\theta \in [-2, 2]$ rad, $\sin\theta$ increases up to $\pi/2$ and then decreases, so the
unknown part is **not monotone** in $\theta$. Tutorial 1's consistency check catches this
straight away if we wrongly assume monotonicity:

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np

from ddma import (BoundedDerivativeModel, GrowthMap, MinimalSimulatingMap, RectangularPartition,
                  SymbolicAbstraction, affine_bounds, safety, simulate)

T, A, B, C, D = 0.05, 9.81, 0.5, 1.0, 0.01
X_LO, X_HI = np.array([-2.0, -4.0]), np.array([2.0, 4.0])
U_MAX = 6.0

def unknown(X, U):                      # used only to make data and to test
    return T * (A * np.sin(X[:, 0]) - B * X[:, 1] + C * U[:, 0])

def known(X, U):                        # g(x, u) = (theta + T omega, omega)
    return np.stack([X[:, 0] + T * X[:, 1], X[:, 1]], axis=1)

rng = np.random.default_rng(0)
X = rng.uniform(X_LO, X_HI, (2000, 2))
U = rng.uniform(-U_MAX, U_MAX, (2000, 1))
y = unknown(X, U) + rng.uniform(-D, D, 2000)
S = MinimalSimulatingMap(np.hstack([X, U]), y, -D, D, signs=[1, -1, 1])
print("consistent with 'increasing in theta':", S.is_consistent())
```

What we *do* know from physics is how fast the unknown part can change:

$$
\frac{\partial f_\omega}{\partial \theta} = T a\cos\theta \in [T a \cos 2,\; T a], \qquad
\frac{\partial f_\omega}{\partial \omega} = -T b, \qquad
\frac{\partial f_\omega}{\partial u} = T c .
$$

Bounds like these (a Lipschitz bound per variable, with a sign when you know it) are
available for most physical systems, and they can also be estimated from data (end of this
tutorial).

## The idea in one dimension: growth cones

If $f'(z) \in [g_{lo}, g_{hi}]$ and we measured $y_k = f(z_k) + d_k$, then $f$ can only move
away from that measurement at a bounded rate. To the right of $z_k$:

$$
y_k - d_{hi} + g_{lo}(z - z_k) \;\le\; f(z) \;\le\; y_k - d_{lo} + g_{hi}(z - z_k),
$$

and mirrored to the left. Each data point therefore confines $f$ to a **cone** with its apex at
the data point. The model is the intersection of all the cones.
{class}`~ddma.GrowthMap` does this for one scalar function on a grid. Here
$f(z) = \sin z + 0.3 z$ (not monotone), with $f' \in [-0.7, 1.3]$:

```{code-cell} ipython3
f1 = lambda z: np.sin(z) + 0.3 * z
g_lo, g_hi, d = -0.7, 1.3, 0.1
Zs = np.array([[1.2], [2.4], [3.9], [5.2]])
ys = f1(Zs[:, 0]) + rng.uniform(-d, d, 4)
grid = RectangularPartition.uniform([0], [2 * np.pi], [120])
gm = GrowthMap.from_data(Zs, ys, grid, g_lo=[g_lo], g_hi=[g_hi], d_lo=-d, d_hi=d)

zs = np.linspace(0, 2 * np.pi, 600)
lo, hi = gm.bounds(zs[:, None])
fig, ax = plt.subplots(figsize=(7, 4))
for zk, yk in zip(Zs[:, 0], ys):          # the cone of each data point (bounds on f + d)
    r, l = zs >= zk, zs < zk
    up = np.where(r, yk + 2 * d + g_hi * (zs - zk), yk + 2 * d + g_lo * (zs - zk))
    dn = np.where(r, yk - 2 * d + g_lo * (zs - zk), yk - 2 * d + g_hi * (zs - zk))
    ax.plot(zs, up, color="#8d99ae", lw=0.7)
    ax.plot(zs, dn, color="#8d99ae", lw=0.7)
ax.fill_between(zs, lo, hi, color="#2a9d8f", alpha=0.4, step="post", label="GrowthMap: intersection of the cones")
ax.plot(zs, f1(zs) - d, "k--", lw=1); ax.plot(zs, f1(zs) + d, "k--", lw=1, label="true $f + [-0.1, 0.1]$")
ax.plot(Zs, ys, "o", color="#264653", label="4 data points")
ax.plot([], [], color="#8d99ae", lw=0.7, label="cone of each point")
ax.set(xlabel="$z$", ylabel="$y$", ylim=(-3, 5), xlim=(0, 2 * np.pi))
ax.legend(frameon=False, fontsize=8, loc="upper left");
```

Unlike the monotone model, **every point is bounded on both sides as soon as there is one data
point**: a cone reaches in every direction. Far from the data the cones open up and the bound
gets loose; close to data it is tight. (The width $2d$ appears because, as in tutorial 1, both
the old and the new measurement carry a disturbance.)

In $n$ dimensions the cone is a product of 1-D cones, one per coordinate, and evaluating the
intersection of $K$ cones exactly at every grid vertex sounds like $O(K \cdot |Q|)$ work.
ddma does it in **one forward and one backward sweep per axis** over the grid (the bound
propagates from vertex to vertex like a distance transform), in C++.

## Five methods

{class}`~ddma.BoundedDerivativeModel` learns a model of the system part $f$ (with $g$
included) for the abstraction. It implements the five approaches of the thesis, Chapter 3,
selected with `method=`:

| `method=` | thesis | idea | trade-off |
|---|---|---|---|
| `"two_matrix"` (default) | approach 1 | subtract the steepest linear trends so two **monotone** functions remain; learn both with tutorial 1's map; intersect | fastest; a cell stays unbounded until data lie on the right side for both transforms |
| `"all_matrix"` | approach 2 | the same with every combination of slopes ($2^n$ monotone maps) | tighter, bounds every cell |
| `"lin_mono"` | approach 3 | approach 2 with a small LP per cell at the end | tighter still, slower |
| `"growth"` | approach 4 | growth cones at the grid vertices, then the cell bounds | as tight as `all_matrix`, fast, can be updated *locally* online |
| `"lin_growth"` | approach 5 | growth cones with an LP per cell at the end | tightest, slowest |

All five are **sound** (they contain the true system whenever the derivative bounds and
disturbance bounds hold). The LP steps are certified by duality, so solver tolerances cannot
make them unsound.

The trick behind approach 1 is worth seeing once: with $a^- = \min(a_{lo}, 0)$,
$f(z) - a^- z$ has derivative $\ge 0$, so it is monotone, and tutorial 1 applies to it.
Similarly $a^+ z - f(z)$ with $a^+ = \max(a_{hi}, 0)$. Each gives bounds on $f$; their
intersection is the model.

## The pendulum, five ways

Data: 400,000 random transitions on $[-2, 2] \times [-4, 4]$, torque in $[-6, 6]$. Grid:
81 × 81 state cells × 21 input cells. Specification: stay in the box forever. With
$|u| \le 6$, gravity wins beyond about 0.6 rad, so most of the box is *not* recoverable; the
question is how much of the recoverable part each model certifies.

Two details of the call. `features=[[1], [0, 1, 2]]` says output 0 (the $\theta$ row of $f$,
which is zero) depends only on $\omega$ and output 1 on everything. The known part enters as
`g` (for the data) and `g_bounds` (for the abstraction), and the model includes it, so the
abstraction needs no `g_bounds` of its own.

```{code-cell} ipython3
K = 400_000
X = rng.uniform(X_LO, X_HI, (K, 2))
U = rng.uniform(-U_MAX, U_MAX, (K, 1))
Xn = known(X, U) + np.stack([np.zeros(K), unknown(X, U) + rng.uniform(-D, D, K)], 1)

Px = RectangularPartition.uniform(X_LO, X_HI, [81, 81])
Pu = RectangularPartition.uniform([-U_MAX], [U_MAX], [21])
P = RectangularPartition.product(Px, Pu)
A_lo = np.array([[0.0, 0.0], [T * A * np.cos(2.0), -T * B]])     # d f / d x
A_hi = np.array([[0.0, 0.0], [T * A, -T * B]])
B_lo = B_hi = np.array([[0.0], [T * C]])                         # d f / d u
g_bounds = affine_bounds([[1.0, T], [0.0, 1.0]], np.zeros((2, 1)))

def step(x, u, rng):                    # the real pendulum, for closed-loop tests
    xn = known(x[None], np.atleast_1d(u)[None])[0]
    return xn + [0.0, unknown(x[None], np.atleast_1d(u)[None])[0] + rng.uniform(-D, D)]

ctrls = {}
print(f"{'method':11s} {'model [s]':>9s} {'abstraction + safety [s]':>25s} {'safe region':>12s}")
for method in ("two_matrix", "all_matrix", "lin_mono", "growth", "lin_growth"):
    t0 = time.perf_counter()
    model = BoundedDerivativeModel.from_data(
        X, U, Xn, P, A_lo, A_hi, B_lo, B_hi, d_lo=[0.0, -D], d_hi=[0.0, D],
        g=known, g_bounds=g_bounds, features=[[1], [0, 1, 2]], method=method)
    t1 = time.perf_counter()
    ctrl = safety(SymbolicAbstraction(Px, Pu, model), np.ones(Px.n_cells, bool))
    t2 = time.perf_counter()
    ctrls[method] = ctrl
    print(f"{method:11s} {t1 - t0:9.2f} {t2 - t1:25.2f} {100 * ctrl.winning_fraction():11.1f}%")
```

The ranking is the thesis's: two-matrix < all-matrix ≈ growth < the LP versions. In
practice **`"growth"` is the usual choice**: almost as tight as the best, as fast as the
fastest, and it supports cheap local updates (tutorial 4). Use `"lin_growth"` when the last
percent matters and the grid is small.

```{code-cell} ipython3
c = [0.5 * (e[1:] + e[:-1]) for e in Px.edges]
fig, ax = plt.subplots(figsize=(7, 4.2))
best = ctrls["lin_growth"]
ax.contourf(c[0], c[1], best.domain_grid().T.astype(float), levels=[0.5, 1.5],
            colors=["#2a9d8f"], alpha=0.35)
ax.plot([], [], "s", color="#2a9d8f", alpha=0.5, label="lin_growth")
for m, col in (("growth", "#264653"), ("two_matrix", "#e76f51")):
    ax.contour(c[0], c[1], ctrls[m].domain_grid().T.astype(float), levels=[0.5], colors=[col])
    ax.plot([], [], color=col, label=m)
for x0 in [[0.3, 0.0], [-0.2, 1.0], [0.1, -1.2]]:
    xs, _ = simulate(step, best, np.array(x0), 300, rng=rng, pick="random")
    ax.plot(xs[:, 0], xs[:, 1], color="#f4a261", lw=0.8)
ax.plot([], [], color="#f4a261", label="closed loop, random allowed inputs")
ax.set(xlabel=r"$\theta$ [rad]", ylabel=r"$\omega$ [rad/s]", xlim=(-1.2, 1.2), ylim=(-4, 4),
       title="Certified safe region of the inverted pendulum (zoom)")
ax.legend(frameon=False, fontsize=8, loc="lower left");
```

The safe region is the familiar diagonal band: leaning right is recoverable only while
moving left, and vice versa. The closed-loop runs pick a *random* allowed input at every
step (the most erratic certified controller) and still stay inside.

Let's test the guarantee harder: 100 runs of 300 steps from random states of each set, on the
true pendulum.

```{code-cell} ipython3
for m, ctrl in ctrls.items():
    dom = np.flatnonzero(ctrl.domain)
    lo, hi = Px.cell_bounds(Px.all_indices()[rng.choice(dom, 100)])
    bad = 0
    for x0 in rng.uniform(lo, hi):
        xs, us = simulate(step, ctrl, x0, 300, rng=rng, pick="random")
        bad += len(us) < 300 or not np.all(Px.contains(xs))
    print(f"{m:11s}: {bad} violations in 100 runs")
```

## When you don't know the derivative bounds

{func}`~ddma.estimate_derivative_bounds` estimates them from data (TAC paper, Section IV).
Step 1 solves an LP on random pairs of samples: the tightest slopes that explain the
differences between pairs, with a disturbance term. Step 2 checks the result on many fresh
pairs and sets the disturbance bound $w$ so that all of them are explained. Step 2 comes with
a **probabilistic** guarantee (scenario approach, {func}`~ddma.scenario_confidence`): with
probability $\ge 1 - \beta$, a new pair of samples is consistent with the bounds with
probability $\ge \beta^{1/N}$.

```{code-cell} ipython3
from ddma import estimate_derivative_bounds, scenario_confidence

Zd = np.hstack([X[:20_000], U[:20_000]])
yd = (Xn[:20_000, 1] - X[:20_000, 1])[:, None]          # samples of the unknown part
print("true            lower", np.round([T * A * np.cos(2), -T * B, T * C], 3),
      " upper", np.round([T * A, -T * B, T * C], 3), f" w = {D}")
est = {}
for w_weight in (1, 50):
    a_lo, a_hi, w = estimate_derivative_bounds(Zd, yd, n_lp_pairs=5_000, n_w_pairs=200_000,
                                               rng=np.random.default_rng(1), w_weight=w_weight)
    est[w_weight] = (a_lo, a_hi, w)
    print(f"w_weight = {w_weight:<3d}  lower", np.round(a_lo[0], 3), " upper", np.round(a_hi[0], 3),
          f" w = {w[0]:.4f}")
print(f"guarantee: with probability >= 1 - 1e-6, a fresh pair is consistent with probability "
      f">= {scenario_confidence(200_000, 1e-6):.5f}")
```

Look at the result critically. The LP trades slope against disturbance, and with the cost of
the paper (`w_weight=1`) it explains the data with **one slope** for $	heta$ and a disturbance
about 13 times the true one: the pendulum's nonlinearity is absorbed into "noise". Weighting the
disturbance more (`w_weight=50`, the setting of the original experiments) gives slopes close
to physics and a disturbance close to the truth. The lower slope in $	heta$ is still
narrower than the true $Ta\cos 2 = -0.204$: the steepest negative slopes only occur near
$|	heta| = 2$, and few random pairs see them.

Both estimates explain all 200,000 check pairs, so both are usable, but the guarantee changes
nature: it now holds **with high probability** (assuming independent samples from a fixed
distribution), not deterministically. Here is the growth model built from the
`w_weight=50` estimate, with disturbance $[-w, w]$:

```{code-cell} ipython3
a_lo, a_hi, w = est[50]
model = BoundedDerivativeModel.from_data(
    X, U, Xn, P, np.vstack([[0, 0], a_lo[:, :2]]), np.vstack([[0, 0], a_hi[:, :2]]),
    np.vstack([[0], a_lo[:, 2:]]), np.vstack([[0], a_hi[:, 2:]]),
    d_lo=[0.0, -w[0]], d_hi=[0.0, w[0]], g=known, g_bounds=g_bounds,
    features=[[1], [0, 1, 2]], method="growth")
ctrl_est = safety(SymbolicAbstraction(Px, Pu, model), np.ones(Px.n_cells, bool))
print(f"safe region with estimated bounds: {100 * ctrl_est.winning_fraction():.1f} %"
      f" (with the physical bounds: {100 * ctrls['growth'].winning_fraction():.1f} %)")
```

The larger disturbance bound costs part of the safe region. When physics gives you bounds,
prefer them; when it doesn't, this is how you get usable ones, and you should state the
probabilistic nature of the guarantee.

## Summary

- Non-monotone dynamics are handled through **bounds on the partial derivatives**: each data
  point confines $f$ to a cone, and the model is the intersection.
- Five methods, all sound; `"growth"` is the usual choice, `"lin_growth"` the tightest.
- {class}`~ddma.GrowthMap` is the same model for a single scalar function (useful inside
  other tools, e.g. learning a vehicle's turn rate for an MPC).
- Derivative bounds can be estimated from data, with a probabilistic guarantee instead of a
  deterministic one.

Next: {doc}`04-online-learning` refines these models while the system runs.

## In MATLAB

`ddma.bd_fit` returns the same struct as `ddma.fit`, so it goes straight into
`ddma.abstraction`. The derivative bounds are given as one matrix per bound over all of
$(x, u)$, the data as $Z = [X\ U]$, $Y = X^+ - g(X, U)$, and $g$ goes to the abstraction as
`g_bounds`. The complete script is `matlab/examples/pendulum_bounded_derivatives.m`.

::::{tab-set}
:::{tab-item} MATLAB
```matlab
T = 0.05; a = 9.81; b = 0.5; c = 1; D = 0.01;
f = @(X, U) T * (a * sin(X(:, 1)) - b * X(:, 2) + c * U);      % unknown (data only)
ddma.seed(0); K = 100000;
X = [-2 -4] + rand(K, 2) .* [4 8];  U = -6 + 12 * rand(K, 1);
Y = [zeros(K, 1), f(X, U) + D * (2 * rand(K, 1) - 1)];          % x+ - g(x, u)
Xedges = ddma.uniform_partition([-2 -4], [2 4], [61 61]);
Uedges = ddma.uniform_partition(-6, 6, 15);
edges = [Xedges(:); Uedges(:)]';
g_lo = [0 0 0; T * a * cos(2), -T * b, T * c];                  % d f / d (theta, omega, u)
g_hi = [0 0 0; T * a,          -T * b, T * c];
S = ddma.bd_fit([X U], Y, edges, g_lo, g_hi, [0 -D], [0 D], 'growth');
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds([1 T; 0 1]));
[domain, allowed] = ddma.safety(A, true(prod(ddma.grid_shape(Xedges)), 1));
fprintf('safe region: %.1f %% of the box\n', 100 * mean(domain));
```
:::
:::{tab-item} Python
```python
model = BoundedDerivativeModel.from_data(
    X, U, Xn, P, A_lo, A_hi, B_lo, B_hi, d_lo=[0, -D], d_hi=[0, D],
    g=known, g_bounds=g_bounds, method="growth")
ctrl = safety(SymbolicAbstraction(Px, Pu, model), np.ones(Px.n_cells, bool))
```
:::
::::
