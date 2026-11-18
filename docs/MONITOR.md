# Assumption monitor

Every guarantee in ddma is conditional: the models are sound *if* the unknown system satisfies the assumptions given to them (a monotonicity sign pattern, bounds on partial derivatives, bounds on the disturbance). ddma checks data against these assumptions twice:

- **before learning**, `ddma.check_data` checks every pair of the learning data;
- **while the system runs**, `ddma.AssumptionMonitor` (MATLAB: `ddma.monitor*`) checks each new transition, says which assumption the data contradict and by how much, sets bad points aside, and responds.

## One form for all models

Each assumption becomes an interval of slopes per output *i* and input *j*, plus a disturbance width θᵢ = d_hi,i − d_lo,i:

| assumption | slope interval of ∂fᵢ/∂zⱼ |
|---|---|
| sign +1 (non-decreasing) | [0, ∞) |
| sign −1 (non-increasing) | (−∞, 0] |
| sign 0, or *j* not a feature of output *i* | [0, 0] |
| bounded derivatives (any method) | [g_lo, g_hi] (for `two_matrix`, `all_matrix`, `lin_mono`: [min(g_lo, 0), max(g_hi, 0)], the interval these methods rely on) |

Two data points *a*, *b* are consistent with the assumptions if and only if, for every output *i*,

  y_a,i − y_b,i ≤ θᵢ + Σⱼ sᵢⱼ(z_a,j − z_b,j),  sᵢⱼ(t) = g_hi,ij t if t > 0, g_lo,ij t if t < 0, 0 if t = 0.

This is the growth cone of point *b* containing point *a*; for sign patterns it is the consistency condition of the minimal simulating map (TAC paper). A pair that violates it is a **proof**: it cannot happen under correct assumptions and correct data, so there are no false alarms.

## Before learning: `check_data`

```python
res = ddma.check_data(Z, Y, d_lo=d_lo, d_hi=d_hi, signs=signs)   # or slope_lo=, slope_hi=
print(res.report())
res.ok, res.violating, res.suspects()   # conflicting pairs per output, most suspicious points
```

It needs only the data and the assumptions, no model, and checks every pair exactly (in the C++ core, on all cores). `conflicts[k]` counts the points that point *k* conflicts with; corrupted readings come out on top of `suspects()`. `needed` is the same diagnosis as the monitor's (below). With checked learning data, the model's empty cells are no longer needed as a sign of inconsistent data.

## While the system runs: three levels

Each new point is checked at up to three levels. Every alarm, from any level, is a proof; each `Violation` records the level that found it and when (`found_at`).

| level | the new point against … | when |
|---|---|---|
| 1. `"cells"` | the model's bounds at its cell (`for_model`) | every step |
| 2. `"sample"` | a fresh random sample of `sample` stored points (default 2,000) | every step, with `full="later"` |
| 3. `"full"` | **every** earlier point | `full="step"` (default): in the same step; `full="later"`: all pending points at once in `flush()` |

`full="later"` is for runs where time is short: collect during the run (levels 1 and 2 only), then check everything back at base with `flush()`, before the model is updated, and deal with what it finds (a sensor problem or an assumption problem). `flush()` gives exactly the same violations, gaps and witnesses as checking each point in its own step. Level 2 draws its samples from the MINSTD sequence, so Python and MATLAB draw the same points.

**Exact pruning.** The pair checks of levels 2 and 3 and `check_data` run in the C++ core with OpenMP; results are bit-identical for any number of threads, and to the NumPy and `.m` code. A grid of cells (`prune=True`, default) keeps, for each cell, the box of its points and the smallest and largest value of each output. From these, an upper bound of the gap of every pair with the cell follows. A cell whose bound is ≤ tol is skipped. The bound is computed with the same rounded operations, in the same order, as the pair gap (with each input at its worst value); rounding never turns a larger input into a smaller result, so a computed pair gap never exceeds the computed bound. Pruning therefore changes the work, never the result. In the timings below it leaves about 1,000–2,300 pairs per new point, whether 20,000 or 2 million points are stored.

## Quarantine

A point that fails a check is **quarantined**: it counts for the diagnosis and the report, but it is no longer reference data, so later points are not blamed for disagreeing with it. When two points conflict, the **earlier one is trusted**, unless it is itself quarantined (then neither is blamed by that pair). `data()` returns the clean points (`data(quarantined=True)`: all), `quarantine()` lists the quarantined ones with their violations. After a widening all data are consistent, and the quarantined points rejoin the data.

## Which assumption?

A violated pair can be repaired by changing one assumption, keeping all others fixed:

- the disturbance width θᵢ must grow by the gap;
- the upper bound g_hi,ij can repair it only if z_a,j > z_b,j, and must reach r_j / (z_a,j − z_b,j), where r_j = y_a,i − y_b,i − θᵢ − Σ_{k≠j} sᵢₖ(z_a,k − z_b,k);
- the lower bound g_lo,ij can repair it only if z_a,j < z_b,j, and must reach the same ratio.

Widening one bound never breaks a pair that was consistent before. So the extreme of these values over all violating pairs is **exactly the smallest change of that one assumption that makes all data consistent** (`needed()`, over all stored points, quarantined ones included; exact once no full check is pending). A bound that cannot repair some violating pair is ruled out.

Data alone cannot always tell the assumptions apart: a wrong sign and a large disturbance can explain the same pair. The report lists every assumption that could explain the violations on its own, with the change it needs, and the ones that cannot. The size of the change usually settles it.

## Responses

| `on_violation` | what happens |
|---|---|
| `"alarm"` (default) | record the violation, call `callback(violations)` |
| `"fallback"` | latch `fallback_active` (until `reset()`), so the application switches to a backup controller |
| `"widen"` | apply the smallest change (× (1 + `margin`)) of the `widen` target; `needs_refit` is set and the model must be relearned from `data()`. No limit. |
| `"widen_then_fallback"` | widen while the needed width stays within `widen_limit` × the original width (default 2×); beyond, fall back, with the reason in `fallback_reason` and the report |

After a widening **the guarantees hold for assumptions fitted to the data seen so far, not given in advance.** The default target is the disturbance. A slope bound is widened only on explicit request (`widen=("slope_hi", i, j)`), only if both of its ends are finite, and only with an absolute `slope_limit`: a sign has no scale ("2×" means nothing for it), and widening a sign makes a monotone model non-monotone, so for monotone models only the disturbance can be widened (a bounded-derivative model is needed otherwise).

**Application rule.** After a refit the certified safe set is usually smaller. If the current state is outside the new safe set, use the backup controller (the example does this).

## The realistic loop

Online model updates are not gated by the monitor. The loop that fits the timings below is:

- **every step:** the monitor's check (well under a millisecond) and the last certified safe set;
- **every N steps, or in the background:** model update and safe-set recomputation (`known_winning` makes it incremental);
- **after an alarm:** fall back until a new safe set is ready.

This rests on one property: while the assumptions hold, an update only tightens the model, so the old safe set stays valid in the meantime. The monitor is what tells you that the assumptions still hold.

## Timings

Measured with `benchmarks/monitor.py` on a cloud VM (2 cores), 1 / 2 threads; timings vary by about ±30 % between runs. Data: the cruise-control learning data (two inputs, two outputs), new points from the same plant. Per new point unless marked:

| stored points | 20 k | 200 k | 2 M |
|---|---|---|---|
| level 1, cells | 0.03 ms | 0.03 ms | 0.03 ms |
| level 2, sample of 2,000 | 0.06 ms | 0.05 ms | 0.05 ms |
| level 3, C++ with pruning | 0.05 / 0.03 ms | 0.12 / 0.10 ms | 0.56 / 0.37 ms |
| … pairs checked | about 1,100 | about 1,200 | about 2,300 |
| level 3, C++ without pruning | 0.54 / 0.39 ms | 3.9 / 2.0 ms | 42 / 21 ms |
| level 3, NumPy (v0.12 round 1) | 1.7 ms | 26 ms | 310 ms |
| whole step (`check()`, median) | 0.19 / 0.12 ms | 0.22 / 0.19 ms | 0.73 / 0.67 ms |
| `flush()` of 500 pending points | 22 / 8 ms | 59 / 30 ms | 270 / 124 ms |
| `check_data`, all pairs | 0.20 / 0.09 s | 1.7 / 1.0 s | 35 / 18 s |
| `check_data` without pruning | 4.5 / 2.4 s | not measured | not measured |

`add()` reserves room for at least half as many new points as are stored, so a run never triggers a copy of the storage; the largest step measured with 2 million points took about 1.3 ms.

## Example

`examples/assumption_monitor.py` (MATLAB: `matlab/examples/assumption_monitor.m`) uses the cruise-control safety filter, learned from 200,000 transitions that `check_data` first confirms (all pairs, about 3 s). At t = 75 s, one assumption breaks in each of two scenarios; the response is `widen_then_fallback` with the default 2× limit in both.

| scenario | caught by | response | no monitor | with monitor |
|---|---|---|---|---|
| leader slows to 11 m/s (assumed ≥ 13) | level 1 (cells), same step | needs width 2.77 < limit 4: widen (2 → 3.0), refit: safe set 38 % → 30 % | 54 steps outside the specification | 3 steps (right after the change) |
| trailer: 30 % of the acceleration, twice the drag | level 3 (full check), same step | needs width 0.30 > limit 0.2: fallback | 6 steps outside the specification | 0 steps |

![monitor](figures/monitor_cruise_control.png)

A violation of the assumptions can break the specification within the very step it happens (here, a leader braking harder than assumed while the follower is already at the edge of the safe set). No monitor can prevent that. What the monitor gives is the first moment when it is *known* that the guarantee no longer holds, together with the size of the problem.

## Limits

- **Single changes only.** The diagnosis gives the smallest change of *one* assumption; the smallest combination would be an LP over all violating pairs.
- **Trust in order.** When two points conflict, the earlier one is trusted. A bad point that arrives before any data in its region is trusted until a later contradiction, and that later point is then quarantined instead.
- **Pruning in many dimensions.** The grid has about K / 128 cells (at most 8,192); with many inputs each input gets few cells, and pruning skips less.
