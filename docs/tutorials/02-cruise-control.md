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

# 2. From data to a certified safe controller

**What you'll learn:** the whole pipeline on a real control problem: data → model →
abstraction → maximal safe set → a controller and a safety filter, with a guarantee for the
real system. This is the TAC paper's cruise-control example.

**Main calls:** {func}`~ddma.transitions_to_data`, {class}`~ddma.PartitionedSimulatingMap`,
{class}`~ddma.SymbolicAbstraction`, {func}`~ddma.affine_bounds`, {func}`~ddma.safety`,
{class}`~ddma.Controller`, {func}`~ddma.simulate`.

## The problem

A follower car drives behind a leader. The state is $x = (h, v)$: the gap $h$ [m] and the
follower's speed $v$ [m/s]. The input $u$ is an acceleration command in $[-3, 3]$ m/s². The
time step is $T = 0.5$ s.

$$
\begin{aligned}
h^+ &= h - T v + T v_l, && v_l \in [13, 17] \text{ m/s (the leader, unknown)} \\
v^+ &= \underbrace{v + T\big(u - (f_0 + f_1 v + f_2 v^2)/M\big)}_{\text{unknown}} + w, && |w| \le 0.05
\end{aligned}
$$

The gap equation is pure kinematics, so we treat it as known. The speed equation contains
drag and actuator effects that we pretend not to know. We only assume what any engineer
would accept: **the next speed increases with the current speed and with the command.**
The leader can do anything within $[13, 17]$ m/s.

**Specification:** keep $h \in [10, 50]$ m (no collision, don't lose the leader) and
$v \in [2, 28]$ m/s, forever, whatever the leader does.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np

from ddma import (PartitionedSimulatingMap, RectangularPartition, SymbolicAbstraction,
                  affine_bounds, safety, simulate, transitions_to_data)

T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
VL = (13.0, 17.0)                          # leader speed range
W = 0.05                                   # bound on the speed disturbance
X_LO, X_HI = np.array([5.0, 0.0]), np.array([60.0, 30.0])     # state box (grid)
U_LO, U_HI = np.array([-3.0]), np.array([3.0])
SAFE_LO, SAFE_HI = np.array([10.0, 2.0]), np.array([50.0, 28.0])

def plant_step(x, u, rng, vl=None):
    """The real car (a black box for ddma: only used to make data and to test)."""
    vl = rng.uniform(*VL) if vl is None else vl
    h, v = x
    v_next = v + T * (u[0] - (F0 + F1 * v + F2 * v**2) / M) + rng.uniform(-W, W)
    return np.array([h - T * v + T * vl, v_next])
```

## Step 1: data, and splitting off what we know

We log random transitions of the black-box car. The known part is
$g(x, u) = (h - Tv,\; 0)$, a linear map $x \mapsto A_g x$. Everything else goes into the unknown
part: $f_1(x, u) = 0$ with "disturbance" $T v_l \in [6.5, 8.5]$ for the gap, and the speed map
with disturbance $w \in [-0.05, 0.05]$.
{func}`~ddma.transitions_to_data` subtracts $g$ and stacks $z = (x, u)$:

```{code-cell} ipython3
rng = np.random.default_rng(0)
K = 20_000
X = rng.uniform(X_LO, X_HI, size=(K, 2))
U = rng.uniform(U_LO, U_HI, size=(K, 1))
Xn = np.array([plant_step(x, u, rng) for x, u in zip(X, U)])

A_G = np.array([[1.0, -T], [0.0, 0.0]])          # g(x, u) = A_G x
def g(X, U):
    return X @ A_G.T

Z, Y = transitions_to_data(X, U, Xn, g)          # Z = (h, v, u), Y = x+ - g(x, u)
d_lo, d_hi = [T * VL[0], -W], [T * VL[1], W]     # bounds of D, per output
Z.shape, Y.shape
```

Note how the leader enters: it is not a state we model but a **bounded disturbance on the gap
output**. ddma will guarantee safety against every leader behaviour in $[13, 17]$ m/s, not
against a typical one.

## Step 2: the model, on the inputs it depends on

The unknown part does not depend on the gap $h$ at all: only on $(v, u)$. Telling ddma so
(`features=[1, 2]`, the coordinates of $z$ to use) keeps the model 2-D instead of 3-D: less
memory, and every data point informs every gap.

The grid: 110 × 60 state cells and 24 input cells. The model lives on the projection of the
state × input grid onto $(v, u)$:

```{code-cell} ipython3
Xgrid = RectangularPartition.uniform(X_LO, X_HI, [110, 60])
Ugrid = RectangularPartition.uniform(U_LO, U_HI, 24)
features = [1, 2]
model_grid = RectangularPartition.product(Xgrid, Ugrid).project(features)   # (v, u): 60 x 24

t0 = time.perf_counter()
sigma = PartitionedSimulatingMap.from_data(Z[:, features], Y, model_grid, d_lo, d_hi)
print(f"model learned in {time.perf_counter() - t0:.3f} s;"
      f" {100 * sigma.bounded_fraction():.1f} % of the (v, u) cells bounded")
```

What does the model know about the speed update? Take $v = 15$ m/s:

```{code-cell} ipython3
us = np.linspace(-3, 3, 7)
lo, hi = sigma.bounds(np.column_stack([np.full_like(us, 15.0), us]))
true = 15 + T * (us - (F0 + F1 * 15 + F2 * 15**2) / M)
for u, l, h, t in zip(us, lo[:, 1], hi[:, 1], true):
    print(f"u = {u:+.0f}: v+ in [{l:6.2f}, {h:6.2f}]   (true {t:6.2f} ± {W})")
```

The intervals are about 1 m/s wide: each covers a whole cell of $(v, u)$ (0.5 m/s ×
0.25 m/s²), the disturbance, and the gaps between data points. They always contain the truth.
At $u = \pm 3$, the edge of the input range, one side is unbounded: no data point lies beyond
the edge to bound it (tutorial 1). Output 0, the gap, is just
$[T v_l^{min}, T v_l^{max}] = [6.5, 8.5]$ everywhere, as it should be.

## Step 3: the abstraction

For every state cell $q$ and input cell $p$, the abstraction computes a box of cells that
contains every possible successor: the model's interval over the cell pair, plus the known
part $g$ over the cell ({func}`~ddma.affine_bounds` gives the exact interval image of a linear
map over a box).

```{code-cell} ipython3
t0 = time.perf_counter()
abst = SymbolicAbstraction(Xgrid, Ugrid, sigma, g_bounds=affine_bounds(A_G), features=features)
print(f"{abst.n_states} state cells x {abst.n_inputs} input cells, "
      f"built in {time.perf_counter() - t0:.2f} s")
```

Let's look at one transition: the cell containing $(h, v) = (30, 15)$ with the input cell
containing $u = 0$.

```{code-cell} ipython3
q = np.ravel_multi_index(tuple(Xgrid.locate([[30.0, 15.0]])[0]), Xgrid.shape)
p = np.ravel_multi_index(tuple(Ugrid.locate([[0.0]])[0]), Ugrid.shape)
succ = abst.successors(q, p)
lo_q, hi_q = Xgrid.cell_bounds(Xgrid.all_indices()[[q]])
s_lo, s_hi = Xgrid.cell_bounds(Xgrid.all_indices()[succ])
print("from cell", lo_q[0], "-", hi_q[0])
print(f"-> {succ.size} successor cells covering h in [{s_lo[:, 0].min():.1f}, {s_hi[:, 0].max():.1f}],"
      f" v in [{s_lo[:, 1].min():.1f}, {s_hi[:, 1].max():.1f}]")
```

The gap can end up anywhere in a range 3.5 m wide: 1 m from the leader's unknown speed, the
rest from the width of the cell itself (0.5 m of gap, 0.5 m/s of speed) and from rounding the
box outward to whole cells. This over-approximation is what makes the
guarantee hold for every real trajectory; the price is conservatism, which finer grids and
more data reduce.

## Step 4: the maximal safe set

{func}`~ddma.safety` takes the abstraction and a mask of safe state cells, and computes the
**maximal controlled invariant set**: start with all safe cells, repeatedly delete every cell
that has no input whose successor box stays inside the remaining set, stop when nothing
changes. What remains is the largest set from which the abstraction, and so the real car,
can be kept safe forever.

```{code-cell} ipython3
safe = abst.cells_inside(SAFE_LO, SAFE_HI)
t0 = time.perf_counter()
ctrl = safety(abst, safe)
print(f"synthesis in {time.perf_counter() - t0:.3f} s: "
      f"{100 * ctrl.domain.sum() / safe.sum():.1f} % of the specification is certified safe")
```

The result is a {class}`~ddma.Controller`: `ctrl.domain` (the winning cells) and
`ctrl.allowed[q, p]` (which input cells keep cell $q$ in the set). Let's see the set, and how
it grows with data:

```{code-cell} ipython3
def learn(K, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(X_LO, X_HI, size=(K, 2))
    U = rng.uniform(U_LO, U_HI, size=(K, 1))
    Xn = np.array([plant_step(x, u, rng) for x, u in zip(X, U)])
    Z, Y = transitions_to_data(X, U, Xn, g)
    s = PartitionedSimulatingMap.from_data(Z[:, features], Y, model_grid, d_lo, d_hi)
    a = SymbolicAbstraction(Xgrid, Ugrid, s, g_bounds=affine_bounds(A_G), features=features)
    return safety(a, a.cells_inside(SAFE_LO, SAFE_HI))

ctrls = {2_000: learn(2_000), 20_000: ctrl, 100_000: learn(100_000)}
he, ve = Xgrid.edges
hc, vc = (he[1:] + he[:-1]) / 2, (ve[1:] + ve[:-1]) / 2
fig, ax = plt.subplots(figsize=(7, 4.2))
for (k, c), col in zip(ctrls.items(), ("#e76f51", "#2a9d8f", "#264653")):
    ax.contour(hc, vc, c.domain_grid().T.astype(float), levels=[0.5], colors=[col])
    ax.plot([], [], color=col, label=f"{k:,} samples: {100 * c.domain.sum() / safe.sum():.1f} %")
ax.add_patch(plt.Rectangle(SAFE_LO, *(SAFE_HI - SAFE_LO), fill=False, ls="--", ec="k"))
ax.set(xlabel="gap $h$ [m]", ylabel="speed $v$ [m/s]", xlim=(5, 60), ylim=(0, 30),
       title="Certified safe region (inside each curve)")
ax.legend(loc="upper left", frameon=False, fontsize=9);
```

The shape makes physical sense. Near $h = 10$ m only low speeds are safe: if the leader
drives at 13 m/s, the follower must be able to slow down before the gap closes. Near
$h = 50$ m the follower must be fast enough to keep up with a 17 m/s leader. More data
shrink the model's intervals, which shrinks the successor boxes, which enlarges the set.

## Step 5: using the controller

A controller gives you, at any state of its domain, inputs that are certified safe:

```{code-cell} ipython3
x = np.array([30.0, 15.0])
print("allowed input cells:", ctrl.input_cells(x))
print("one certified input:", ctrl(x))                  # centre of the first allowed cell
print("filter of u = +3:   ", ctrl.filter(x, [3.0]))    # closest certified input to +3
```

The most useful mode is the **safety filter**: you keep your own controller (an MPC, a PID,
a neural network, a human) and pass every command through `ctrl.filter`. It passes the
command unchanged when it is safe and replaces it by the closest safe one otherwise.

Here the nominal controller is a naive cruise control that holds 22 m/s and ignores the
leader, which is faster than the leader can ever go. The leader alternates between fast
and slow phases:

```{code-cell} ipython3
N = 400
rng = np.random.default_rng(1)
vl_profile = np.clip(np.where((np.arange(N) // 40) % 2 == 0, VL[1], VL[0])
                     + rng.uniform(-0.5, 0.5, N), *VL)

def nominal(x):
    return np.clip([0.3 * (22.0 - x[1])], U_LO, U_HI)

def make_step():
    k = [0]
    def step(x, u, rng):
        out = plant_step(x, u, rng, vl=vl_profile[k[0]])
        k[0] += 1
        return out
    return step

# without the filter (stop at a collision)
step, x, raw = make_step(), np.array([30.0, 15.0]), [np.array([30.0, 15.0])]
for _ in range(N):
    raw.append(step(raw[-1], nominal(raw[-1]), rng))
    if raw[-1][0] <= 0:
        break
raw = np.array(raw)

# with the filter: ddma.simulate calls ctrl.filter(x, nominal(x)) at every step
xs, us = simulate(make_step(), ctrl, np.array([30.0, 15.0]), N, rng=rng, nominal=nominal)

t = np.arange(N + 1) * T
fig, ax = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
ax[0].plot(t[:len(raw)], raw[:, 0], color="#e76f51", label="naive controller")
ax[0].plot(t[:len(xs)], xs[:, 0], color="#264653", label="naive controller + safety filter")
ax[0].axhline(10, ls="--", c="k", lw=0.8)
ax[0].set(ylabel="gap [m]")
ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax[1].step(t[:len(us)], [nominal(x)[0] for x in xs[:-1]], where="post", color="#8d99ae",
           lw=1, label="wanted")
ax[1].step(t[:len(us)], us[:, 0], where="post", color="#2a9d8f", label="applied")
ax[1].set(ylabel="$u$ [m/s²]", xlabel="time [s]")
ax[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
print(f"naive alone: collision at t = {(len(raw) - 1) * T:.1f} s" if raw[-1][0] <= 0
      else "naive alone: no collision")
print(f"with filter: {len(us)} steps, min gap {xs[:, 0].min():.1f} m, "
      f"inside the specification: {bool(np.all((xs >= SAFE_LO) & (xs <= SAFE_HI)))}")
changed = np.any(np.abs(us - np.array([nominal(x) for x in xs[:-1]])) > 1e-9, axis=1)
print(f"the filter changed the command in {100 * changed.mean():.0f} % of the steps")
```

The naive controller always wants to accelerate (it is below 22 m/s), so it runs into the
leader within seconds. With the filter the gap settles just above 10 m whenever the leader is
slow. The filter changes the command in most steps here, because the naive controller keeps
pushing against the edge of the safe set; and the applied input **chatters**: the filter has
no memory and no notion of comfort, it returns the closest certified input at each step.
That is the job of the controller inside the filter (an MPC that knows about the filter, or
an agent penalised for corrections, tutorial 5), not of the filter. **Why this is a guarantee, not a test result:** every state the
filtered loop visits is in `ctrl.domain`, and from every such state the applied input has
successor box inside `ctrl.domain` for *every* system consistent with the data, including
the real car with any leader in $[13, 17]$ m/s. The loop cannot leave, so it never violates
the specification (as long as the assumptions hold: monotone speed dynamics, $|w| \le 0.05$).

## Reaching a target: `reach_avoid`

Safety keeps you in a set; {func}`~ddma.reach_avoid` brings you somewhere. Here: reach a
comfortable following distance $h \in [15, 35]$ m at $v \in [12, 18]$ m/s without leaving the
specification on the way. `value[q]` is the guaranteed worst-case number of steps from cell
$q$ (`inf` if the target can't be guaranteed).

```{code-cell} ipython3
from ddma import reach_avoid

target = abst.cells_inside([15.0, 12.0], [35.0, 18.0])
reach = reach_avoid(abst, target, avoid=~safe)
x0 = np.array([48.0, 20.0])
q0 = np.ravel_multi_index(tuple(Xgrid.locate([x0])[0]), Xgrid.shape)
print(f"{100 * reach.domain.mean():.1f} % of the cells can reach the target; "
      f"from {x0}: at most {reach.value[q0]:.0f} steps ({reach.value[q0] * T:.0f} s)")
xs_r, _ = simulate(make_step(), reach, x0, 200, rng=rng, stop_in=target)
print(f"simulated: reached in {len(xs_r) - 1} steps, final state {np.round(xs_r[-1], 1)}")
q1 = np.ravel_multi_index(tuple(Xgrid.locate([[45.0, 10.0]])[0]), Xgrid.shape)
print("from (45, 10):", reach.value[q1], "steps")
```

From $(48, 20)$ the car closes the gap in a guaranteed number of steps, and the simulation
arrives sooner because the real leader is not the worst case. From $(45, 10)$ there is no
guarantee: the car is slower than any leader, so the gap first grows, and a 17 m/s leader
would take it past 50 m before the car is fast enough. A smaller target (say $h \in [20,
30]$) would be reachable from far fewer cells: each step adds up to 3.5 m of uncertainty on
the gap, so a narrow band of gaps can't be hit for sure from far away.

## Summary

- Known parts ($g$) are subtracted; external influences (the leader) become bounded
  disturbances; the unknown part is learned only on the coordinates it depends on
  (`features`).
- The abstraction over-approximates every behaviour of every consistent system, so the safe
  set and its inputs are certified for the real system.
- More data → tighter model → larger certified set.
- The controller works best as a **safety filter** around any other controller.

Next: {doc}`03-pendulum` handles a system that is *not* monotone.

## In MATLAB

The same pipeline, with functions on structs (`matlab/examples/acc_safety_filter.m` is the
complete script):

::::{tab-set}
:::{tab-item} MATLAB
```matlab
T = 0.5; VL = [13 17]; W = 0.05;
speed = @(v, u) v + T * (u - (51 + 1.2567 * v + 0.4342 * v.^2) / 1370);   % unknown
A_G = [1 -T; 0 0];                                       % known part g(x) = A_G x
ddma.seed(0); K = 20000;
X = [5 0] + rand(K, 2) .* [55 30];  U = -3 + 6 * rand(K, 1);
Xn = [X(:, 1) - T * X(:, 2) + T * (13 + 4 * rand(K, 1)), speed(X(:, 2), U) + W * (2 * rand(K, 1) - 1)];
[Z, Y] = ddma.transitions_to_data(X, U, Xn, @(X, U) X * A_G');

Xedges = ddma.uniform_partition([5 0], [60 30], [110 60]);
Uedges = ddma.uniform_partition(-3, 3, 24);
features = [2 3];                                        % (v, u), 1-based
S = ddma.fit(Z(:, features), Y, {Xedges{2}, Uedges{1}}, [T * VL(1), -W], [T * VL(2), W]);
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G), features);
safe = ddma.cells_inside(Xedges, [10 2], [50 28]);
[domain, allowed] = ddma.safety(A, safe);
fprintf('%.1f %% certified\n', 100 * nnz(domain) / nnz(safe));
u = ddma.controller_input(A, allowed, [30 15], 3);      % safety filter of u = 3
```
:::
:::{tab-item} Python
```python
Z, Y = transitions_to_data(X, U, Xn, g)
Xgrid = RectangularPartition.uniform([5, 0], [60, 30], [110, 60])
Ugrid = RectangularPartition.uniform([-3], [3], 24)
features = [1, 2]                                        # (v, u), 0-based
sigma = PartitionedSimulatingMap.from_data(
    Z[:, features], Y, RectangularPartition.product(Xgrid, Ugrid).project(features), d_lo, d_hi)
abst = SymbolicAbstraction(Xgrid, Ugrid, sigma, g_bounds=affine_bounds(A_G), features=features)
safe = abst.cells_inside([10, 2], [50, 28])
ctrl = safety(abst, safe)
u = ctrl.filter([30, 15], [3.0])
```
:::
::::
