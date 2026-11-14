# Python ↔ MATLAB

ddma has two interfaces on one C++ core. Every feature exists in both, and the MATLAB results are tested against Python reference results in CI (`matlab/tests/run_tests.m`, reference data written by `matlab/tests/make_reference.py`; run under Octave on every push).

Python uses classes; MATLAB uses functions on structs (`S = ddma.bd_update(S, ...)`). The table maps one to the other.

## Models

| Python | MATLAB | Tested against Python |
|---|---|---|
| `PartitionedSimulatingMap.from_data(Z, Y, partition, d_lo, d_hi, signs=)` | `S = ddma.fit(Z, Y, edges, d_lo, d_hi, signs)` | exact |
| `..., compact=True` / `.compact()` | `ddma.fit(..., [], [], true)` / `ddma.compact(S)` | exact (single, rounded outward) |
| `.update(Z, Y)`, `.intersect(other)` | `ddma.fit_update(S, Z, Y)`, `ddma.intersect(S, T)` | exact |
| `.bounds(z)`, `.f_bounds(z)`, `.contains(z, y)` | `ddma.bounds(S, z)`, `ddma.f_bounds(S, z)`, `ddma.contains(S, z, y)` | exact |
| `.widths()`, `.bounded_fraction()`, `.is_empty()` | `ddma.widths(S)`, `ddma.bounded_fraction(S)`, `ddma.is_empty(S)` | exact |
| `MinimalSimulatingMap(Z, Y, d_lo, d_hi, signs).bounds(z)` | `ddma.msm_bounds(Z, Y, d_lo, d_hi, z, signs)` | exact |
| `.f_bounds(z)`, `.estimate(z)`, `.is_consistent()` | `ddma.msm_f_bounds`, `ddma.msm_estimate`, `ddma.msm_is_consistent` | exact |
| `.merge(other)` | concatenate the data: `[Z1; Z2]`, `[Y1; Y2]` | — |
| `BoundedDerivativeModel.from_data(..., method=, local=, features=)` | `S = ddma.bd_fit(Z, Y, edges, g_lo, g_hi, d_lo, d_hi, method, local, backend, features)` | ≤ 1e-15 (all five methods, `local`, features) |
| `.update(X, U, Xn)` | `S = ddma.bd_update(S, Z, Y)` | ≤ 1e-9 |
| `.update_local(X, U, Xn, window)` | `S = ddma.bd_update_local(S, Z, Y, window)` | ≤ 1e-9 |
| `GrowthMap(...)`, `.bounds`, `.f_bounds` | `ddma.growth_map(Z, y, edges, g_lo, g_hi, d_lo, d_hi, local, final)`, `ddma.bounds`, `ddma.f_bounds` | ≤ 1e-9 |
| `estimate_derivative_bounds(Z, Y, n_lp_pairs, n_w_pairs, w_weight=)` | `ddma.estimate_derivative_bounds(Z, Y, n_lp_pairs, n_w_pairs, w_weight, pairs)` | same LP optimum on the same pairs |
| `scenario_confidence(n, beta)` | `ddma.scenario_confidence(n, beta)` | exact |

The data of `bd_fit` are `Z = [X U]` and `Y = X' - g(X, U)`, as for `ddma.fit` (`ddma.transitions_to_data`); the known part g goes to `ddma.abstraction` as `g_bounds`.

## Abstractions and synthesis

| Python | MATLAB | Tested against Python |
|---|---|---|
| `SymbolicAbstraction(X, U, model, g_bounds=, features=, assumption=)` | `A = ddma.abstraction(Xedges, Uedges, S, g_bounds, features, assumption)` | exact transitions |
| `..., lazy=True` | `ddma.abstraction(..., true)` (7th argument) | same winning set and inputs |
| `.refresh(states=None)` | `[A, S, mask] = ddma.abstraction_refresh(A, S, states)` | exact (mask, transitions) |
| `.cells_inside`, `.cells_meeting` | `ddma.cells_inside(edges, lo, hi)`, `ddma.cells_meeting(edges, lo, hi)` | exact |
| `safety(A, safe, known_winning=, check=)` | `[domain, allowed] = ddma.safety(A, safe, backend, known_winning, check)` | exact |
| `reach_avoid(A, target, avoid)` | `[domain, allowed, value] = ddma.reach_avoid(A, target, avoid)` | exact |
| `Controller(x, pick=)`, `.filter(x, u)` | `ddma.controller_input(A, allowed, x, pick)` (pick = desired input for the filter) | same inputs |
| `simulate(step, ctrl, x0, n, pick=, stop_in=, nominal=)` | `[xs, us] = ddma.simulate(step, A, allowed, x0, n, pick, stop_in, nominal)` | same trajectory |
| `affine_bounds(A, B, c)` | `ddma.affine_bounds(A, B, c)` | exact |

## Reinforcement learning

| Python | MATLAB | Tested against Python |
|---|---|---|
| `ddma.rl.SafetyFilterWrapper(env, controller, state_fn=, penalty=, reset_in_domain=)` (Gymnasium) | `ddma.safe_step(step, A, allowed, state_fcn, penalty)`, `ddma.safe_reset(reset, A, allowed, state_fcn)` (step/reset functions in the form of the RL Toolbox's `rlFunctionEnv`) | same safe actions (300 random states) |
| `ddma.rl.CruiseControlEnv` | `matlab/examples/cruise_control_env.m` | same dynamics and reward |

## Safe envelopes (no grid)

| Python | MATLAB | Tested against Python |
|---|---|---|
| `monotone_envelope(X, Xn, U, w_lo=, w_hi=, x_lo=, x_hi=, state_signs=, input_signs=, u_lo=, u_hi=)` | `ddma.monotone_envelope(X, Xn, U, opts)` (same option names in a struct) | same corners and (corner, input) pairs |
| `embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, w_lo=, w_hi=, x_lo=, x_hi=, pairs=)` | `ddma.embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, opts)` (pairs 1-based) | same boxes, input boxes and pairs |
| `env.contains(x)` | `ddma.envelope_contains(E, x)` | same on 2,000 states |
| `env.safe_input(x, u_desired)` | `ddma.envelope_input(E, x, u_desired)` | same inputs |
| `ddma.invariance.maximal_points`, `max_min_cones` | `ddma.maximal_points`, `ddma.max_min_cones` | equal |

Tests: `matlab/tests/run_invariance_tests.m` (reference written by `matlab/tests/make_reference_invariance.py`).

## Partitions and plots

| Python | MATLAB |
|---|---|
| `RectangularPartition.uniform(lo, hi, n)` | `ddma.uniform_partition(lo, hi, n)` |
| `RectangularPartition.with_margins(...)` | `ddma.margin_partition(...)` |
| `RectangularPartition.from_data(Z, lo, hi)` | `ddma.data_partition(Z, lo, hi)` (tested) |
| `RectangularPartition.product(P1, P2)`, `.project(dims)` | `[E1(:); E2(:)]`, `E(dims)` |
| `.locate(Z)`, `.cell_bounds(idx)` | `ddma.locate(edges, Z)`, `ddma.cell_bounds(edges, sub)` |
| `ddma.plotting.plot_staircase_bounds`, `plot_set_3d`, `plot_set_2d` | `ddma.plot_bounds`, `ddma.plot_set` |

## Examples

| Python (`examples/`) | MATLAB (`matlab/examples/`) |
|---|---|
| `scalar_map.py` | `scalar_map.m` |
| `bounded_derivative_map.py` | `bounded_derivative_map.m` |
| `adaptive_cruise_control.py` | `acc_safety_filter.m` |
| `pendulum_bounded_derivatives.py` | `pendulum_bounded_derivatives.m` (coarser grid) |
| `lazy_synthesis.py` | `lazy_synthesis.m` |
| `rl_safety_filter.py` (PPO with Stable-Baselines3) | `rl_safety_filter.m` (exploring agents; with the RL Toolbox, train any agent on `rlFunctionEnv(..., safeStep, safeReset)`) |
| `envelope_room_cooling.py` | `envelope_room_cooling.m` |
| `envelope_nonlinear.py` | `envelope_nonlinear.m` |
| — | `online_refinement.m` |

The examples use random data, so the two languages draw different samples and print slightly different numbers; on the same data the results are equal (tested).

## Differences that remain

- **Paper reproductions** (`reproductions/`) are Python scripts; `matlab/examples/paper_figures.m` redraws the paper figures in MATLAB.
- **LP final steps** (`'lin_mono'`, `'lin_growth'`) and `estimate_derivative_bounds` need `linprog` (Optimization Toolbox) in MATLAB; Octave has `glpk` built in. The LP bounds are certified by weak duality in both languages, so they are sound whatever the solver.
- **RL training** in MATLAB needs the Reinforcement Learning Toolbox; ddma's filter functions do not, and the MATLAB example runs without it.
- **Compiled kernels**: Python builds them on install; MATLAB with `build_mex` (plain MATLAB code is used otherwise, with the same results).
