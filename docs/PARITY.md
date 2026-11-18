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
| `SymbolicAbstraction(X, U, model, g_bounds=, features=, assumption=, backend=)` | `A = ddma.abstraction(Xedges, Uedges, S, g_bounds, features, assumption, lazy, backend)` | exact transitions (C++ kernel on all threads by default; `backend` `"numpy"` / `'matlab'` for the plain code) |
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

## Assumption monitor

| Python | MATLAB | Tested against Python |
|---|---|---|
| `check_data(Z, Y, slope_lo, slope_hi, d_lo, d_hi, signs=, tol=, prune=, backend=)` → `DataCheck` (`ok`, `violating`, `worst`, `worst_pair`, `conflicts`, `needed`, `suspects()`, `report()`) | `R = ddma.check_data(Z, Y, slope_lo, slope_hi, d_lo, d_hi, opts)` (struct with the same fields; `suspects`, `report` as fields; indices 1-based) | same counts, worst pair, conflicts, needed changes |
| `AssumptionMonitor(slope_lo, slope_hi, d_lo, d_hi, on_violation=, widen=, widen_limit=, slope_limit=, margin=, full=, sample=, prune=, callback=, tol=, seed=, backend=)` | `M = ddma.monitor(slope_lo, slope_hi, d_lo, d_hi, opts)` (same option names in a struct; `widen` = `{'slope_lo', i, j}`, 1-based; unknown options are an error) | — |
| `AssumptionMonitor.monotone(signs, d_lo, d_hi, m=, n=)` | `ddma.monitor_monotone(signs, d_lo, d_hi, opts)` | — |
| `AssumptionMonitor.for_model(model, Z, Y)` | `ddma.monitor_for_model(S, Z, Y, opts)` (S from `ddma.fit`, `ddma.bd_fit`, `ddma.growth_map`) | same slope intervals |
| `.add(Z, Y)`, `.check(Z, Y)` → list of `Violation` (`index`, `output`, `level`, `gap`, `witness`, `found_at`) | `ddma.monitor_add(M, Z, Y)`, `[M, viol] = ddma.monitor_check(M, Z, Y)` (struct array, indices 1-based, no witness = `nan`) | same violations, levels, gaps (≤ 1e-12), witnesses and `found_at`, also for level 2 (same random samples) |
| `.flush()`, `.status()`, `.pending` | `[M, viol] = ddma.monitor_flush(M)`, `ddma.monitor_status(M)` | same violations |
| `.observe(X, U, Xn)` (with `g=`, `features=`) | `ddma.monitor_check(M, Z, X' - g)` | — |
| `.needed()` | `ddma.monitor_needed(M)` | ≤ 1e-12 |
| `.widen(target)` | `ddma.monitor_widen(M, target)` | same widened bounds |
| `.data(quarantined=False)`, `.quarantine()` | `[Z, Y] = ddma.monitor_data(M, quarantined)`, `ddma.monitor_quarantine(M)` | same quarantine flags |
| `.consistent()`, `.reset()`, `.report()` | `ddma.monitor_consistent(M)`, `ddma.monitor_reset(M)`, `ddma.monitor_report(M)` | same verdict |
| `.rebuild_index()` | `ddma.monitor_rebuild_index(M)` | — |
| `.ok`, `.fallback_active`, `.fallback_reason`, `.needs_refit`, `.set_model(...)` | `isempty(M.violating)`, `M.fallback_active`, `M.fallback_reason`, `M.needs_refit`, `M.model = S` | same fallback |
| pair checks in C++ (`backend="cpp"` / `"numpy"`) | MEX `ddma_monitor_mex` (`backend` `'mex'` / `'matlab'`) | MEX = `.m` code bit for bit, 1–3 threads, with and without pruning |

Tests: `matlab/tests/run_monitor_tests.m` (reference written by `matlab/tests/make_reference_monitor.py`).

## Safe envelopes (no grid)

| Python | MATLAB | Tested against Python |
|---|---|---|
| `monotone_envelope(X, Xn, U, w_lo=, w_hi=, x_lo=, x_hi=, state_signs=, input_signs=, u_lo=, u_hi=)` | `ddma.monotone_envelope(X, Xn, U, opts)` (same option names in a struct) | same corners and (corner, input) pairs |
| `embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, w_lo=, w_hi=, x_lo=, x_hi=, pairs=)` | `ddma.embedding_envelope(X, Xn, U, jx_lo, jx_hi, ju_lo, ju_hi, opts)` (pairs 1-based) | same boxes, input boxes and pairs |
| returned objects `MonotoneEnvelope`, `BoxEnvelope` | structs with the same fields | — |
| `env.contains(x)` | `ddma.envelope_contains(E, x)` | same on 2,000 states |
| `env.safe_input(x, u_desired)` | `ddma.envelope_input(E, x, u_desired)` | same inputs |
| `ddma.invariance.maximal_points`, `max_min_cones` | `ddma.maximal_points`, `ddma.max_min_cones` | equal |

Tests: `matlab/tests/run_invariance_tests.m` (reference written by `matlab/tests/make_reference_invariance.py`).

## Threads

| Python | MATLAB | Tested |
|---|---|---|
| `ddma.set_num_threads(n)` (`None` or 0: default, all cores or `OMP_NUM_THREADS`) | `ddma.set_num_threads(n)` (`[]` or 0: default) | same bits for 1, 2 and all threads (`tests/test_threads.py`, `matlab/tests/run_threads_tests.m`) |
| `ddma.get_num_threads()`, `ddma.openmp_available()` | `[n, openmp] = ddma.get_num_threads()` | — |
| `ddma.compiled_backend_available()` | `exist('ddma_fit_mex', 'file') == 3` | — |

See [PERFORMANCE.md](PERFORMANCE.md).

## Partitions and plots

| Python | MATLAB |
|---|---|
| `RectangularPartition.uniform(lo, hi, n)` | `ddma.uniform_partition(lo, hi, n)` |
| `RectangularPartition.with_margins(...)` | `ddma.margin_partition(...)` |
| `RectangularPartition.from_data(Z, lo, hi)` | `ddma.data_partition(Z, lo, hi)` (tested) |
| `RectangularPartition.product(P1, P2)`, `.project(dims)` | `[E1(:); E2(:)]`, `E(dims)` |
| `.locate(Z)`, `.cell_bounds(idx)` | `ddma.locate(edges, Z)`, `ddma.cell_bounds(edges, sub)` |
| `ddma.plotting.plot_staircase_bounds`, `plot_set_3d`, `plot_set_2d` | `ddma.plot_bounds`, `ddma.plot_set` |

## Helpers

MATLAB needs a few small functions that Python gets from NumPy or keeps internal:

| Python | MATLAB |
|---|---|
| `partition.shape` | `ddma.grid_shape(edges)` |
| `partition.search(i, z, side)` (= `numpy.searchsorted`) | `ddma.searchsorted(e, z, side)` |
| `numpy.ravel_multi_index` / `numpy.unravel_index` | `ddma.flat_index(shape, sub)` / `ddma.all_subs_of(shape, f)`, `ddma.all_subs(shape)` (MATLAB order, 1-based indices) |
| `numpy.random.default_rng(n)` | `ddma.seed(n)` (MATLAB and Octave) |
| internal: successor-box test of the synthesis | `ddma.box_all_in(A, W, backend)` |
| internal: certified LP bound of the `lin_*` methods (`ddma._box_lp.max_min_affine`) | `ddma.max_min_affine(...)` |

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
- **Compiled kernels**: Python builds them on install; MATLAB with `build_mex` (plain MATLAB code is used otherwise, with the same results). Both use threads (OpenMP) when the compiler supports it; the plain MATLAB code runs on one thread.
