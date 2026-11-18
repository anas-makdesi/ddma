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

# 6. Safe envelopes straight from data, without a grid

**What you'll learn:** how ddma certifies a safe operating region directly from the data
points, with no grid, no abstraction and no model on a partition; the two methods
(monotone systems, and systems with bounded Jacobians); and when they beat the grid, and when
they don't.

**Main calls:** {func}`~ddma.monotone_envelope`, {func}`~ddma.embedding_envelope`,
`env.contains`, `env.safe_input`.

## Why drop the grid

Everything so far went through a grid: the model on cells, the abstraction on cell pairs.
Grids are what make the synthesis exact and simple, but their size grows exponentially with
the dimension (30 cells per axis is 27,000 cells in 3-D and $6 \cdot 10^{14}$ in 10-D), and
every cell needs data around it.

A **safe envelope** is a controlled invariant set built from the data points themselves:

- for monotone systems, a union of *lower sets* $\{x \le T\}$, each certified with a range of
  inputs ({func}`~ddma.monotone_envelope`, Makdesi & Zamani, IFAC WC 2026);
- for systems with bounded Jacobians, a union of *boxes* spanned by pairs of data points,
  each certified with a box of inputs ({func}`~ddma.embedding_envelope`, Makdesi, Zamani &
  Jafarpour, IFAC WC 2026).

Both are deterministic guarantees: every state in the envelope can be kept in it forever,
for every disturbance within the bounds, with the inputs the envelope certifies.

## Monotone systems: corners from data

Three rooms in a row, one shared cooling input $u \in [0, 1]$, unknown heat gains
$w \in [0, 0.15]$ °C per step (tutorial 4's plant). Specification: every room at or below
26 °C. That specification is **lower-closed**: if $x$ satisfies it, so does every
$x' \le x$. This is what the monotone method needs.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np

from ddma import (PartitionedSimulatingMap, RectangularPartition, SymbolicAbstraction,
                  embedding_envelope, monotone_envelope, safety)

def rooms(Tm, u):                      # the plant (unknown to the method)
    a, b, c = 0.05, 0.3, 0.6
    cpl = np.stack([Tm[:, 1] - Tm[:, 0], Tm[:, 0] + Tm[:, 2] - 2 * Tm[:, 1], Tm[:, 1] - Tm[:, 2]], 1)
    return Tm + 0.25 * (a * (34 - Tm) + b * cpl - c * u * (Tm - 14))

rng = np.random.default_rng(0)
N, W_HI, T_MAX = 1000, 0.15, 26.0
X = rng.uniform(16, 32, (N, 3))
U = rng.uniform(0, 1, (N, 1))
Xn = rooms(X, U) + rng.uniform(0, W_HI, (N, 3))

t0 = time.perf_counter()
env = monotone_envelope(X, Xn, U, w_lo=0.0, w_hi=W_HI, input_signs=[-1],
                        x_hi=[T_MAX] * 3, u_lo=[0.0], u_hi=[1.0])
print(f"{N} transitions -> {env.n_corners} corners in {time.perf_counter() - t0:.2f} s")
for T, u in sorted(zip(env.pairs_T.tolist(), env.pairs_U[:, 0].tolist()))[:5]:
    print(f"  every state <= {np.round(T, 2)} is safe with any cooling u >= {u:.2f}")
```

`input_signs=[-1]` says the dynamics *decrease* with the cooling input. The method works in
flipped coordinates where everything is increasing, so "more cooling" is "a smaller flipped
input".

### Why a corner is safe

Work in coordinates where $f$ is increasing in $x$ and in the (flipped) input $v$. Take a
corner $T$ and an input limit $V$. For every state $x \le T$ and input $v \le V$,
monotonicity gives $f(x, v) + w \le f(T, V) + w_{hi}$. So the lower set
$\{x \le T\}$ is invariant if $f(T, V) + w_{hi} \le T$.

We don't know $f(T, V)$, but a data point $(x_k, v_k, x_k^+)$ with $x_k \ge T$ and
$v_k \ge V$ bounds it: $f(T, V) \le f(x_k, v_k) = x_k^+ - w_k \le x_k^+ - w_{lo}$. That
gives the condition, **one coordinate at a time**, and each coordinate may use a different
data point $k_i$:

$$
(x_{k_i}^+)_i + (w_{hi} - w_{lo})_i \le T_i, \qquad x_{k_i} \ge T, \quad v_{k_i} \ge V,
\qquad i = 1, \dots, n.
$$

The best corner for a choice of points is $T = \min_i x_{k_i}$ (componentwise) with
$V = \min_i v_{k_i}$. The method searches these choices, keeps the valid corners, and
returns the maximal ones (the "skyline"). Let's check every returned corner against the
condition by brute force:

```{code-cell} ipython3
theta = W_HI - 0.0
V_data, V_corner = -U[:, 0], -env.pairs_U[:, 0]        # flipped inputs: v = -u
ok_all = True
for T, Vc in zip(env.pairs_T, V_corner):
    above = np.all(X >= T, axis=1) & (V_data >= Vc)     # data points that bound f(T, V)
    witness = [np.any(above & (Xn[:, i] + theta <= T[i] + 1e-12)) for i in range(3)]
    ok_all &= all(witness)
print("every corner has a witness data point in every coordinate:", ok_all)
```

### How much it certifies

The envelope certifies a fraction of the safe temperature box $[16, 26]^3$. For comparison,
the grid pipeline of tutorials 2 and 4 (30 cells per axis, 21 input cells, lazy abstraction)
gets **200 times more data**:

```{code-cell} ipython3
S = rng.uniform(16, T_MAX, (200_000, 3))                 # samples of the safe box

Na = 200_000
Xa = rng.uniform(16, 32, (Na, 3))
Ua = rng.uniform(0, 1, (Na, 1))
Xan = rooms(Xa, Ua) + rng.uniform(0, W_HI, (Na, 3))
t0 = time.perf_counter()
Xp = RectangularPartition.uniform([16] * 3, [32] * 3, 30)
Up = RectangularPartition.uniform([0.0], [1.0], 21)
sigma = PartitionedSimulatingMap.from_data(np.hstack([Xa, Ua]), Xan,
                                           RectangularPartition.product(Xp, Up),
                                           signs=[[1, 1, 1, -1]] * 3)
A = SymbolicAbstraction(Xp, Up, sigma, lazy=True)
ctrl = safety(A, A.state_mask(lambda lo, hi: np.all(hi <= T_MAX, axis=1)))
t_grid = time.perf_counter() - t0

in_env = env.contains(S)
in_grid = ctrl.domain_grid()[tuple(Xp.locate(S).T)]
print(f"envelope:    {N:>7,} transitions -> {100 * in_env.mean():.1f} % of the safe box")
print(f"grid:        {Na:>7,} transitions -> {100 * in_grid.mean():.1f} % ({t_grid:.1f} s)")
```

```{code-cell} ipython3
g = np.linspace(16, T_MAX, 300)
G1, G2 = np.meshgrid(g, g)
Pts = np.column_stack([G1.ravel(), G2.ravel(), np.full(G1.size, 24.0)])
E = env.contains(Pts).reshape(G1.shape)
Gr = ctrl.domain_grid()[tuple(Xp.locate(Pts).T)].reshape(G1.shape)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.contourf(G1, G2, E, levels=[0.5, 1.5], colors=["#d1495b"], alpha=0.25)
ax.contour(G1, G2, E, levels=[0.5], colors=["#d1495b"], linewidths=2)
ax.contour(G1, G2, Gr, levels=[0.5], colors=["#2e5e8c"], linewidths=1.5, linestyles="--")
ax.fill([], [], color="#d1495b", alpha=0.4, label=f"envelope, {N:,} transitions")
ax.plot([], [], color="#2e5e8c", ls="--", label=f"grid, {Na:,} transitions")
ax.set(xlabel="room 1 [°C]", ylabel="room 2 [°C]", title="Certified safe temperatures, room 3 at 24 °C")
ax.legend(loc="lower left", fontsize=8);
```

Why the envelope wins here: the grid loses a full cell width (about half a degree) at every
step of every transition, while a corner's condition uses the measured successors exactly.
And the corners' staircase hugs the 26 °C limit closely, because the plant can almost always
cool a room that is at 26 °C.

### Using it

`env.safe_input(x, u_wanted)` returns the certified input closest to what you want, like
`Controller.filter`. A lazy controller that never wants to cool, filtered:

```{code-cell} ipython3
x = S[in_env][:500]
hottest, cooling = -np.inf, []
for _ in range(200):
    u = np.array([env.safe_input(xi, [0.0])[0] for xi in x])
    cooling.append(u.mean())
    x = rooms(x, u[:, None]) + rng.choice([0.0, W_HI], size=x.shape)   # worst-case-ish gains
    assert env.contains(x).all()
    hottest = max(hottest, x.max())
print(f"500 starts x 200 steps: hottest room {hottest:.2f} °C (limit {T_MAX}), "
      f"mean cooling {np.mean(cooling):.2f}")
```

## Any system with bounded Jacobians: boxes from pairs

When the system is not monotone, {func}`~ddma.embedding_envelope` uses the derivative bounds
of tutorial 3 instead. Every **pair** of data points $(x_i, u_i)$, $(x_j, u_j)$ spans a box of
states $B = [\min(x_i, x_j), \max(x_i, x_j)]$ and a box of inputs. The growth cones of the two
points (tutorial 3) both contain $f$; over the box, the tighter of the two cones at each point
bounds where $f$ can go. The method computes the worst case of that bound over the box (a
small linear program, solved exactly in closed form), and keeps the box if
$f(B, U_B) + [w_{lo}, w_{hi}] \subseteq B$: the box then maps into itself for every input of
its input box. The envelope is the union of the kept boxes.

Two competing species with small harvesting/stocking inputs:
$x_i^+ = x_i(2 - x_i - 0.2 x_j) + u_i + w_i$ on $[0.6, 1.1]^2$, $|u_i| \le 0.02$,
$|w_i| \le 0.005$. The Jacobian bounds follow from the formula on that domain.

```{code-cell} ipython3
LO, HI, U_MAX, W = 0.6, 1.1, 0.02, 0.005
def competition(x, u):
    return np.stack([x[:, 0] * (2 - x[:, 0] - 0.2 * x[:, 1]) + u[:, 0],
                     x[:, 1] * (2 - x[:, 1] - 0.2 * x[:, 0]) + u[:, 1]], 1)

Xc = rng.uniform(LO, HI, (400, 2))
Uc = rng.uniform(-U_MAX, U_MAX, (400, 2))
Xcn = competition(Xc, Uc) + rng.uniform(-W, W, (400, 2))
jx_lo = np.array([[2 - 2.2 * HI, -0.2 * HI], [-0.2 * HI, 2 - 2.2 * HI]])
jx_hi = np.array([[2 - 2.2 * LO, -0.2 * LO], [-0.2 * LO, 2 - 2.2 * LO]])
t0 = time.perf_counter()
benv = embedding_envelope(Xc, Xcn, Uc, jx_lo, jx_hi, np.eye(2), np.eye(2), w_lo=-W, w_hi=W,
                          x_lo=[LO, LO], x_hi=[HI, HI])
Sc = rng.uniform(LO, HI, (100_000, 2))
print(f"{benv.n_pairs:,} pairs examined in {time.perf_counter() - t0:.2f} s -> "
      f"{benv.n_boxes} invariant boxes covering {100 * benv.contains(Sc).mean():.1f} % of the domain")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 4.6))
for k in np.argsort(-(benv.hi - benv.lo).prod(1))[:40]:          # the 40 largest boxes
    ax.add_patch(plt.Rectangle(benv.lo[k], *(benv.hi[k] - benv.lo[k]), fill=True,
                               fc="#2a9d8f", ec="#264653", alpha=0.12, lw=0.6))
ax.plot(Xc[:, 0], Xc[:, 1], ".", color="#8d99ae", ms=3, label="data")
x = Sc[benv.contains(Sc)][:5]
for _ in range(40):                                           # harvest as much as allowed
    u = np.array([benv.safe_input(xi, [-U_MAX, -U_MAX]) for xi in x])
    xn = competition(x, u) + rng.uniform(-W, W, x.shape)
    for a, b in zip(x, xn):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#e76f51", lw=0.8)
    assert benv.contains(xn).all()
    x = xn
ax.plot([], [], color="#e76f51", label="closed loop, filtered")
ax.set(xlim=(LO, HI), ylim=(LO, HI), xlabel="$x_1$", ylabel="$x_2$",
       title="Invariant boxes (the 40 largest)")
ax.legend(fontsize=8, loc="lower left");
```

The closed loop tries to harvest as much as possible (the most negative input); the envelope
allows it where certified and steers otherwise. The cost of this method is the number of
pairs, $N^2/2$, independent of the dimension: a ring of ten coupled units in 10-D
(`examples/envelope_nonlinear.py`) takes a fraction of a second from 1,000 transitions, where
a grid is out of the question.

## When envelopes are the wrong tool

Each corner or box is certified with **one range of inputs for the whole region**. That works
when a constant input can hold the region, as cooling can hold rooms below a limit. It fails
for chains of integrators. In lane keeping, for example, the lateral offset is driven by the
heading and the heading by the steering. No single steering range holds a box of offsets and
headings: the right steering depends on where in the box you are. There the grid abstraction,
which chooses an input per cell, certifies far more. Choose by the structure:

| | grid abstraction | monotone envelope | embedding envelope |
|---|---|---|---|
| system | monotone or bounded derivatives | monotone | bounded Jacobian |
| specification | any set of cells, reach-avoid | lower-closed | a box |
| input | chosen per cell | one limit per corner | one box per box |
| data needed | data around every cell | few | few |
| cost grows with | cells, exponential in the dimension | data and dimension (search) | pairs of data, $N^2$ |

## Summary

- Envelopes certify safe regions **directly from data points**: corners of lower sets for
  monotone systems, boxes spanned by pairs for systems with bounded Jacobians.
- With few data and a specification that suits them, they certify more than the grid from
  far less data, in high dimension too.
- Each region gets one input range, so systems that need a state-dependent input inside a
  region (integrator chains) are better served by the grid abstraction.

Next: {doc}`07-assumption-monitor`: what happens when the assumptions behind all of this are
wrong. Implementation details: the {doc}`../guides/envelopes` guide.

## In MATLAB

::::{tab-set}
:::{tab-item} MATLAB
```matlab
rooms = @(T, u) T + 0.25 * (0.05 * (34 - T) + 0.3 * [T(:, 2) - T(:, 1), ...
        T(:, 1) + T(:, 3) - 2 * T(:, 2), T(:, 2) - T(:, 3)] - 0.6 * u .* (T - 14));
ddma.seed(0); N = 1000;
X = 16 + 16 * rand(N, 3);  U = rand(N, 1);
Xn = rooms(X, U) + 0.15 * rand(N, 3);
opts = struct('w_lo', 0, 'w_hi', 0.15, 'input_signs', -1, 'x_hi', [26 26 26], ...
              'u_lo', 0, 'u_hi', 1);
E = ddma.monotone_envelope(X, Xn, U, opts);
in = ddma.envelope_contains(E, [24 25 24.5]);
u = ddma.envelope_input(E, [24 25 24.5], 0);            % least cooling that is certified
```
:::
:::{tab-item} Python
```python
env = monotone_envelope(X, Xn, U, w_lo=0, w_hi=0.15, input_signs=[-1], x_hi=[26, 26, 26],
                        u_lo=[0], u_hi=[1])
inside = env.contains([[24, 25, 24.5]])
u = env.safe_input([24, 25, 24.5], [0.0])
```
:::
::::

`matlab/examples/envelope_room_cooling.m` and `matlab/examples/envelope_nonlinear.m` are the
complete scripts.
