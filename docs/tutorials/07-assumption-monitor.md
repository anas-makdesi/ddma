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

# 7. What if the assumptions are wrong?

**What you'll learn:** how ddma checks the "if" in every one of its guarantees. First the
learning data, before you learn from them; then each new transition, while the system runs.
What an alarm proves, how the monitor says which assumption is wrong, and how to respond:
widen the assumption and relearn, or fall back to a controller that doesn't need the model.

**Main calls:** {func}`~ddma.check_data`, {class}`~ddma.AssumptionMonitor`
(`for_model`, `observe`, `report`, `needed`, `data`, `flush`).

## The question

Every result in tutorials 1–6 reads: *if* the system satisfies the assumptions (monotone, or
derivatives within bounds; disturbance within bounds), *then* the model contains the true
system and the safe set is safe. Nothing in the learning checks the "if". The monitor asks,
for every data point:

> Could this point and all the data seen so far come from **one** function that satisfies
> the assumptions?

If the answer is no, an assumption is wrong, or a sensor is. That is an alarm, and it is a
**proof**, not a statistical warning.

## Two points that contradict each other

Write every assumption on one output as an interval of slopes per input,
$\partial f/\partial z_j \in [g^{lo}_j, g^{hi}_j]$ (a sign $+1$ is $[0, \infty)$, a sign
$-1$ is $(-\infty, 0]$, "does not depend" is $[0, 0]$), plus the disturbance width
$\theta = d_{hi} - d_{lo}$. Tutorial 3's growth cones then give the exact condition for two
points $a$, $b$ to be consistent:

$$
y_a - y_b \;\le\; \theta + \sum_j s_j(z_{a,j} - z_{b,j}), \qquad
s_j(t) = \begin{cases} g^{hi}_j\, t & t > 0 \\ g^{lo}_j\, t & t < 0 \\ 0 & t = 0\end{cases}
$$

in both directions. If it fails, no function satisfying the assumptions can produce both
points: no false alarms. (Only the *width* of the disturbance appears: an unknown $f$ can
absorb any constant shift, so data can't tell where the interval sits, only how wide it is.)

An example by hand. Assumptions: slope in $[0, 3]$, disturbance in $[-0.01, 0.01]$, so
$\theta = 0.02$. A stored point $b = (0.2, 0.6)$ and a new point $a = (0.5, 0.1)$: $a$ is to
the right of $b$ but lower by 0.5, more than the noise allows for a non-decreasing function.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np

from ddma import (AssumptionMonitor, PartitionedSimulatingMap, RectangularPartition,
                  SymbolicAbstraction, affine_bounds, check_data, safety, transitions_to_data)

mon = AssumptionMonitor(slope_lo=[[0.0]], slope_hi=[[3.0]], d_lo=-0.01, d_hi=0.01)
mon.add([[0.2]], [0.6])                 # a stored point b
print(mon.check([[0.5]], [0.1]))        # the new point a
print()
print(mon.report())
```

The report goes one step further: for each assumption, the **smallest change of that one
assumption** that would make all data consistent. Here the disturbance width must grow from
0.02 to 0.5, or the lower slope bound must drop to −1.6 (which would contradict "non-
decreasing"), and the upper slope bound can't help at all. Widening a bound only enlarges
cones, so a pair that was consistent stays consistent, and these numbers are exact, not
estimates: the most demanding violating pair decides. You get them as numbers from
`mon.needed()`.

## Before learning: `check_data`

The cheapest moment to catch a wrong assumption or a bad sensor is before you learn
anything. {func}`~ddma.check_data` checks **every pair** of the learning data (in C++, on
all cores, skipping only the parts of the data where no pair can conflict, so the result is
exact). We use tutorial 2's cruise control: the gap output does not depend on $(v, u)$ apart
from the leader (sign 0, disturbance $[6.5, 8.5]$), the speed output is monotone.

```{code-cell} ipython3
T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
VL, W = (13.0, 17.0), 0.05
X_LO, X_HI = np.array([5.0, 0.0]), np.array([60.0, 30.0])
U_LO, U_HI = np.array([-3.0]), np.array([3.0])
SAFE_LO, SAFE_HI = np.array([10.0, 2.0]), np.array([50.0, 28.0])
A_G = np.array([[1.0, -T], [0.0, 0.0]])
D_LO, D_HI = [T * VL[0], -W], [T * VL[1], W]
SIGNS = [[0, 0], [1, 1]]               # outputs (gap update, speed update) x inputs (v, u)
NAMES = dict(input_names=["v", "u"], output_names=["gap update", "speed update"])

def g(X, U):
    return X @ A_G.T

def speed_update(v, u, trailer=False):
    drag = (F0 + F1 * v + F2 * v**2) / M
    return v + T * (0.3 * u - 2 * drag) if trailer else v + T * (u - drag)

def plant_step(x, u, rng, vl, trailer=False):
    return np.array([x[0] - T * x[1] + T * vl, speed_update(x[1], u[0], trailer) + rng.uniform(-W, W)])

rng = np.random.default_rng(0)
K = 50_000
X = rng.uniform(X_LO, X_HI, (K, 2))
U = rng.uniform(U_LO, U_HI, (K, 1))
Xn = np.array([plant_step(x, u, rng, rng.uniform(*VL)) for x, u in zip(X, U)])
Z, Y = transitions_to_data(X, U, Xn, g)
Z = Z[:, [1, 2]]                        # the unknown part depends on (v, u) only

t0 = time.perf_counter()
res = check_data(Z, Y, d_lo=D_LO, d_hi=D_HI, signs=SIGNS, **NAMES)
print(f"{time.perf_counter() - t0:.2f} s, {res.pairs_evaluated:,} pairs computed "
      f"of {K * (K - 1) // 2:,}")
print(res.report())
```

Clean. Now a faulty speed sensor: four logged speeds are off by +0.3 m/s, six times the
noise bound.

```{code-cell} ipython3
bad = rng.choice(K, 4, replace=False)
Yc = Y.copy()
Yc[bad, 1] += 0.3
res = check_data(Z, Yc, d_lo=D_LO, d_hi=D_HI, signs=SIGNS, **NAMES)
print(res.report())
print("\nthe corrupted points:", sorted(bad.tolist()))
print("most conflicts:     ", sorted(res.suspects(4).tolist()))
```

Each corrupted point conflicts with many honest ones, while an honest point conflicts with
at most a few corrupted ones, so counting conflicts per point (`res.conflicts`,
`res.suspects()`) points at the culprits. The report also reads the evidence the other way:
widening the noise bound would explain everything too. Data can't tell a broken sensor from
a noisier plant; you decide which is plausible.

## While running: the monitor

{meth}`AssumptionMonitor.for_model <ddma.AssumptionMonitor.for_model>` builds a monitor from
a fitted model (it reads the assumptions from the model) and stores the learning data. Then
`observe(x, u, x_next)` checks each new transition, at up to three **levels**:

| level | the new point is checked against | cost |
|---|---|---|
| `"cells"` | the model's bounds at its cell | constant |
| `"sample"` | a random sample of stored points (only with `full="later"`) | constant |
| `"full"` | **every** earlier point | grows slowly: a few thousand pairs even with millions stored |

By default (`full="step"`) levels 1 and 3 run in every step, so a contradiction is found the
moment it exists in the data. Under 1 ms per step with 2 million stored points.

Two things go wrong at $t = 75$ s, each in its own scenario:

- **A. The leader slows to 11 m/s**, below the assumed 13 m/s: a disturbance bound is wrong.
- **B. A trailer is hitched**: the car gets 30 % of the commanded acceleration and twice
  the drag. The learned speed map no longer describes the car.

The response is `"widen_then_fallback"`: if the smallest change that explains the data is
at most twice the original disturbance width (`widen_limit=2`), widen it, refit the model
and recompute the safe set from the monitor's clean data, and carry on; beyond that, stop
trusting the model and switch to a backup controller.

```{code-cell} ipython3
Xp = RectangularPartition.uniform(X_LO, X_HI, [110, 60])
Up = RectangularPartition.uniform(U_LO, U_HI, 24)
part = RectangularPartition.product(Xp, Up).project([1, 2])

def synthesize(Z, Y, d_lo, d_hi):
    sigma = PartitionedSimulatingMap.from_data(Z, Y, part, d_lo, d_hi, signs=SIGNS)
    abst = SymbolicAbstraction(Xp, Up, sigma, g_bounds=affine_bounds(A_G), features=[1, 2])
    safe = abst.cells_inside(SAFE_LO, SAFE_HI)
    ctrl = safety(abst, safe)
    return sigma, ctrl, ctrl.domain.sum() / safe.sum()

sigma0, ctrl0, frac0 = synthesize(Z, Y, D_LO, D_HI)
print(f"certified safe set: {100 * frac0:.0f} % of the specification")
```

The run: a naive controller that wants 22 m/s inside the safety filter, as in tutorial 2. A
backup controller that needs no model (match the leader's estimated speed, steer the gap to
30 m) takes over when the monitor falls back, and in *every* run whenever the state is
outside the certified set, where no guarantee holds anyway.

```{code-cell} ipython3
N, K_EVENT = 500, 150

def leader(scenario, seed=1):
    r = np.random.default_rng(seed)
    slow = np.full(N, VL[0])
    if scenario == "slow leader":
        slow[K_EVENT:] = 11.0
    fast = (np.arange(N) // 40) % 2 == 0
    return np.clip(np.where(fast, VL[1], slow) + r.uniform(-0.5, 0.5, N),
                   np.minimum(slow, VL[0]), VL[1])

def nominal(x):
    return np.clip([0.3 * (22.0 - x[1])], U_LO, U_HI)

def backup(x, x_prev):
    vl_est = x_prev[1] + (x[0] - x_prev[0]) / T
    return np.clip([1.5 * (vl_est - x[1]) + 0.2 * (x[0] - 30.0)], U_LO, U_HI)

def run(scenario, monitor, full="step", seed=2):
    rng = np.random.default_rng(seed)
    vl, sigma, ctrl = leader(scenario), sigma0, ctrl0
    mon = AssumptionMonitor.for_model(
        sigma, Z, Y, g=g, features=[1, 2], on_violation="widen_then_fallback",
        widen_limit=2.0, margin=0.2, full=full, **NAMES) if monitor else None
    x = x_prev = np.array([30.0, 15.0])
    xs, log, trans = [x], [], []
    for k in range(N):
        u = None
        if mon is None or not mon.fallback_active:
            u = ctrl.filter(x, nominal(x))
        if u is None:
            u = backup(x, x_prev)
        xn = plant_step(x, u, rng, vl[k], trailer=scenario == "trailer" and k >= K_EVENT)
        trans.append((x, np.atleast_1d(u), xn))
        if mon is not None:
            for v in mon.observe(x[None], np.atleast_1d(u)[None], xn[None]):
                log.append(f"t = {k * T:5.1f} s: level {v.level!r:8s} {mon.output_names[v.output]},"
                           f" {v.gap:.3f} beyond the assumptions")
            if mon.needs_refit:              # widened: relearn with the widened bounds
                sigma, ctrl, frac = synthesize(*mon.data(), mon.d_lo, mon.d_hi)
                mon.set_model(sigma.bounds, sigma.partition.contains)
                log.append(f"           widened, refit: safe set {100 * frac:.0f} %")
        x_prev, x = x, xn
        xs.append(x)
    return np.array(xs), mon, log, trans

results = {}
for scenario in ("slow leader", "trailer"):
    for monitor in (False, True):
        xs, mon, log, _ = run(scenario, monitor)
        results[scenario, monitor] = xs
        out = np.any((xs < SAFE_LO) | (xs > SAFE_HI), axis=1).sum()
        print(f"--- {scenario}, {'monitor' if monitor else 'no monitor'}: "
              f"{out} steps outside the specification")
        if monitor:
            print("\n".join(log[:6] + (["           ..."] if len(log) > 6 else [])))
            print("fallback:", mon.fallback_active, "|", mon.fallback_reason or "")
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
t = np.arange(N + 1) * T
for ax, scenario in zip(axs, ("slow leader", "trailer")):
    ax.plot(t, results[scenario, False][:, 0], color="#e76f51", lw=1.2, label="no monitor")
    ax.plot(t, results[scenario, True][:, 0], color="#264653", lw=1.2, label="monitor")
    ax.axvspan(K_EVENT * T, N * T, color="#e9c46a", alpha=0.15, lw=0)
    for b in (SAFE_LO[0], SAFE_HI[0]):
        ax.axhline(b, ls="--", c="k", lw=0.8)
    ax.set(ylabel="gap [m]", ylim=(-5, 60), title=scenario)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
axs[1].set_xlabel("time [s] (shaded: assumptions broken)")
fig.tight_layout();
```

How to read this:

- **A (slower leader)** is caught by the **cells** level: the new gap update falls outside
  the model's bounds the moment the leader drops below 13 m/s. Only a wider gap-update
  disturbance explains it, and the width needed stays within the limit, so the monitor
  widens it and the model and safe set are relearned. The safe set shrinks, because a slower
  leader is harder to follow safely, and the car stays (almost) inside the specification.
- **B (trailer)** is caught by the **full** check: no cell bound is broken yet, but the new
  transition contradicts a stored one. Explaining it would take several times the speed
  noise bound, beyond the limit, so the monitor falls back to the backup controller with the
  reason in `fallback_reason`.
- **The honest limit:** the specification can break in the very step the world changes,
  before any data arrive. The monitor can't prevent that; what it gives is the first moment
  you *know*, with a proof and a diagnosis.

## Collect now, check later

On a vehicle with little computing power you may want only the constant-cost levels during a
run, and the full check back at base. `full="later"` does that: levels 1 and 2 run in every
step, and `flush()` runs level 3 on all pending points at once, in parallel. It finds
exactly the same violations as checking in every step, only later:

```{code-cell} ipython3
_, _, _, trans = run("trailer", monitor=False)          # the same 500 transitions for both
Xr, Ur, Xnr = (np.array(c) for c in zip(*trans))

def monitor(full, sample=2000):
    return AssumptionMonitor.for_model(sigma0, Z, Y, g=g, features=[1, 2], full=full,
                                       sample=sample, **NAMES)

mon_step, mon_later = monitor("step"), monitor("later", sample=0)
for k in range(N):                                       # step by step, as in a run
    mon_step.observe(Xr[k:k + 1], Ur[k:k + 1], Xnr[k:k + 1])
    mon_later.observe(Xr[k:k + 1], Ur[k:k + 1], Xnr[k:k + 1])
during = [v for v in mon_later.violations]
t0 = time.perf_counter()
mon_later.flush()                                        # back at base
t_flush = time.perf_counter() - t0

def found(mon):
    return sorted((v.index, v.output, v.level, v.gap, v.witness) for v in mon.violations)

print(f"during the run, level 1 caught {len(during)} violations, the first at "
      f"t = {min(v.index for v in during) * T:.1f} s")
print(f"flush() of {N} points: {1e3 * t_flush:.0f} ms")
print(f"after flush(): {len(found(mon_later))} violations, identical to checking in every step:",
      found(mon_step) == found(mon_later))
```

Here the run is the trailer scenario without any response, so both monitors see the same
transitions. During the run the cheap cell level caught the trailer two seconds after the
change, where checking every earlier point in every step caught it after half a second.
That is the price of postponing the full check. Back at base, `flush()` found exactly what
step mode found: the same points, levels, gaps and witnesses. (The comparison turns level 2
off, `sample=0`, because a point that the random sample catches first is recorded as a
`"sample"` violation instead of a `"full"` one. With the sample on, the warning during the
run usually comes earlier.)

## Choosing a response

- **`alarm`** (default): record and tell you; change nothing.
- **`widen`**: keep using a model, relearned with a weaker assumption. Its guarantee now
  rests on a bound fitted to the data rather than given in advance; the safe set usually
  shrinks. Suits small, plausible changes: a leader a bit slower, noise a bit larger.
- **`fallback`**: stop trusting the model; a backup controller that needs no model takes
  over. Suits large or implausible changes: a trailer, a sensor fault.
- **`widen_then_fallback`**: chooses by size, as above.

Signs are never widened (a sign has no scale, and widening it makes a monotone model
non-monotone); a finite slope bound is widened only if you ask for it and give a limit.

## Limits

- **Single changes only.** `needed()` gives the smallest change of *one* assumption. The
  smallest *combination* would need a linear program over the violating pairs.
- **Trust in order.** When two points conflict, the earlier one is trusted (unless it is
  itself quarantined). A bad point that arrives first in an empty region is trusted until
  something contradicts it.
- **Silence is not proof.** No alarm means the data seen so far are consistent, not that the
  assumptions are true.

## Summary

- An alarm is a proof that an assumption (or a sensor) is wrong: two points that no
  function satisfying the assumptions could produce.
- `check_data` checks the learning data before learning; the monitor checks each new
  transition at three levels, under 1 ms per step.
- The report says which single change would explain the data, and by how much; the
  response widens the assumption and relearns, or falls back.

More detail: the {doc}`../guides/monitor` guide.

## In MATLAB

::::{tab-set}
:::{tab-item} MATLAB
```matlab
% the two-point example
M = ddma.monitor(0, 3, -0.01, 0.01);
M = ddma.monitor_add(M, 0.2, 0.6);
[M, viol] = ddma.monitor_check(M, 0.5, 0.1);
disp(viol.gap); disp(ddma.monitor_report(M));

% learning data first: monotone speed update, constant gap update
ddma.seed(0); K = 20000; T = 0.5;
v = 30 * rand(K, 1); u = 6 * rand(K, 1) - 3;
Y = [T * (13 + 4 * rand(K, 1)), ...
     v + T * (u - (51 + 1.2567 * v + 0.4342 * v.^2) / 1370) + 0.05 * (2 * rand(K, 1) - 1)];
R = ddma.check_data([v u], Y, [], [], [6.5 -0.05], [8.5 0.05], ...
                    struct('signs', [0 0; 1 1]));
disp(R.report);
Y(17, 2) = Y(17, 2) + 0.3;                               % a faulty measurement
R = ddma.check_data([v u], Y, [], [], [6.5 -0.05], [8.5 0.05], ...
                    struct('signs', [0 0; 1 1]));
disp(R.suspects(1:3)');                                   % point 17 first
```
:::
:::{tab-item} Python
```python
mon = AssumptionMonitor([[0.0]], [[3.0]], -0.01, 0.01)
mon.add([[0.2]], [0.6])
viol = mon.check([[0.5]], [0.1])
print(mon.report())

res = check_data(Z, Y, d_lo=D_LO, d_hi=D_HI, signs=SIGNS)
print(res.report(), res.suspects(3))
```
:::
::::

`examples/assumption_monitor.py` and `matlab/examples/assumption_monitor.m` are the complete
scripts, with 200,000 learning points.
