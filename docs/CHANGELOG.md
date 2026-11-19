# Changelog

## 0.12.2 (19 November 2026)

The toolbox code is the same as 0.12.1; this release updates the package description shown on PyPI.

- README rewritten for first-time visitors (GitHub and PyPI): install first, one short example in Python and MATLAB, the tutorials listed instead of repeated; all links and figures absolute, so they also work on the PyPI page.
- Corrected: the Apple-silicon wheels of 0.12.1 need macOS 15 or newer (not 14); on macOS 14 with Apple silicon pip builds from source.
- Roadmap: wheels and the MATLAB toolbox moved from planned to done; the remaining packaging items listed.
- Documentation: new guide "How it's tested"; the home page's install section shows `pip install ddma` and the MATLAB toolbox.

## 0.12.1 (19 November 2026)

The toolbox code is the same as 0.12.0; this release adds ready-made packages and the documentation site.

**Pre-built packages**
- Python wheels for CPython 3.10–3.15 on Linux (x86_64, ARM), macOS (Apple silicon: macOS 15 or newer; Intel: macOS 14 or newer) and Windows (x86_64): `pip install ddma`. Each contains the compiled kernels with threads and the OpenMP runtime, so no compiler and no Homebrew `libomp` is needed. Every wheel passes the whole test suite in CI and is checked again on a clean machine.
- MATLAB toolbox `ddma-<version>.mltbx` (R2023b or newer) with pre-built MEX files for Linux, Windows and macOS (Apple silicon), attached to each GitHub release. The MEX files are built and tested in MATLAB R2023b on all three systems.
- `build_mex`: the environment variable `DDMA_MEX_CXX` chooses the C++ compiler (Linux, macOS); MEX files built with an older GCC also load in older MATLAB releases.
- Workflow `.github/workflows/packages.yml`: builds and tests all packages; on a version tag it publishes the wheels to PyPI (Trusted Publishing, after approval) and attaches the toolbox to a GitHub release. `pyproject.toml`: SPDX license, project links, supported Python versions.

**Documentation site** (`docs/`, Sphinx + MyST-NB)
- Seven tutorials, executed on every build: the minimal simulating map of a scalar function; from data to a certified safe controller (cruise control, safety filter, reach-avoid); bounded-derivative models (the five methods on the pendulum, estimating derivative bounds); online learning (local and batch updates, `refresh`, `known_winning`, lazy abstractions); a learning agent inside a safety filter; safe envelopes without a grid; the assumption monitor. Each ends with the MATLAB version of its main calls, run under Octave in CI (`docs/check_matlab_snippets.py`).
- API reference for Python (from the docstrings) and MATLAB (from the help text of every `+ddma` function; the build fails if a function is missing from the reference).
- Getting started (installation, core ideas and notation) and the existing guides (PARITY, INVARIANCE, MONITOR, PERFORMANCE, REPRODUCTION).
- CI job `docs` (`sphinx-build -W`, HTML kept as an artifact) on every branch; workflow `docs-pages` publishes the site on GitHub Pages from `main`; `pip install -e ".[docs]"`.
- `matlab/examples/online_refinement.m`: local updates now use a window of 3 cells (setting `window`); with 1 they barely grew the safe set between the batch updates.
- Docstrings for the remaining public methods and properties; `tests/test_docs.py` checks that every public name has one.

## 0.12.0 (18 November 2026)

**Assumption monitor** (`ddma.monitor`)
- `check_data` checks every pair of the learning data against the assumptions before learning (no model needed): conflicting pairs per output, worst pair, conflicts per point (`suspects()` finds corrupted readings) and the same diagnosis as the monitor. Exact, in the C++ core on all cores (200,000 points in about 1–2 s on two cores).
- `AssumptionMonitor` checks new transitions online against the assumptions of a model: sign patterns, derivative bounds and disturbance bounds, all written as slope intervals plus a disturbance width (the growth-cone consistency condition, so every alarm is a proof). Works with `MinimalSimulatingMap`, `PartitionedSimulatingMap`, `GrowthMap` and `BoundedDerivativeModel` (all five methods).
- Three levels, each `Violation` recording which one found it and when: the model's cell bounds; a fresh random sample of stored points (`sample=`, with `full="later"`); the full check against every earlier point, in the same step (`full="step"`, default) or for all pending points at once with `flush()` (`full="later"`, e.g. back at base; same result).
- Pair checks in the C++ core with OpenMP (`cpp/ddma_monitor.hpp`), bit-identical for any number of threads and to the NumPy and `.m` code; exact pruning with a grid of cells (`prune=True`): with 2 million stored points a step takes under a millisecond.
- Quarantine: a point that fails a check is kept for the diagnosis but no longer used as reference, so it causes no follow-up alarms (`data()`: clean points; `quarantine()`).
- `needed()`: for each assumption, the exact smallest change that alone makes all stored data consistent, or `nan` if that assumption cannot explain the violations. `report()` explains it in plain words.
- Responses: `on_violation="alarm"` (callback), `"fallback"` (latched flag), `"widen"` (smallest widening × (1 + `margin`), then refit from `data()`), `"widen_then_fallback"` (widen up to `widen_limit` × the original width, default 2×, then fall back, with the reason). Slope bounds are widened only on request, only if finite, and only up to `slope_limit`.
- MATLAB: `ddma.check_data`, `ddma.monitor`, `monitor_monotone`, `monitor_for_model`, `monitor_add`, `monitor_check`, `monitor_flush`, `monitor_status`, `monitor_data`, `monitor_quarantine`, `monitor_needed`, `monitor_widen`, `monitor_consistent`, `monitor_reset`, `monitor_report`, `monitor_rebuild_index`; MEX `ddma_monitor_mex`; tested against Python (same random samples).
- Example `examples/assumption_monitor.py` (and `.m`): cruise control; the learning data are checked first, then a leader slower than assumed (widened within the limit: 54 → 3 steps outside the specification) and a trailer (beyond the limit, fallback: 6 → 0). `benchmarks/monitor.py`, `docs/MONITOR.md` with the timings.

**Requirements**
- Python 3.10 or newer (3.9 reached end of life in October 2025). CI tests 3.10, 3.12 and 3.13 on the current Ubuntu runner.

## 0.11.0 (16 November 2026)

**Abstractions built in C++, on all cores**
- `SymbolicAbstraction` computes its transitions in the C++ core (`cpp/ddma_abstraction.hpp`), on several threads, with the same result as before for any number of threads: 7–12× faster than NumPy on one thread (e.g. 27 million state-input pairs in about 0.5 s instead of 7 s), more with threads. Eager and lazy abstractions and `refresh` all use it; `backend="numpy"` keeps the NumPy code.
- Monotone maps (also with `features`), the two-matrix bounded-derivative model (with per-output features) and `g_bounds` from `affine_bounds` are evaluated entirely in the kernel; other models and other `g_bounds` functions are evaluated in NumPy per block of pairs and the kernel does the rest (clipping, checks, cell ranges).
- `affine_bounds` now returns an `AffineBounds` object (callable as before) whose `parts` gives the state and input parts separately.
- Cell ranges from a bucket table with an exact final scan: constant time per value, also on uneven partitions.
- MATLAB: MEX `ddma_abs_mex` (built by `build_mex`), used by `ddma.abstraction`, `ddma.abstraction_refresh` and lazy abstractions; `ddma.abstraction(..., lazy, backend)` with `'auto'`, `'mex'`, `'matlab'`. About 4.5× faster than the `.m` code on one thread.
- MATLAB: a NaN bound clipped to an assumption box now stays NaN (the pair is not enabled), as in Python; before, `max`/`min` replaced it by the box bound.
- Tests `tests/test_abstraction_backends.py`, `matlab/tests/run_abstraction_tests.m`; benchmark `benchmarks/abstraction.py`; `docs/PERFORMANCE.md`.

## 0.10.0 (15 November 2026)

**Multithreaded learning** (OpenMP)
- The learning kernels (`fit_sigma` for the monotone maps and the `two_matrix`, `all_matrix`, `lin_mono` models; `growth_vertices` and `growth_local` for `GrowthMap` and the growth models) run on all cores by default. Results are bit-identical for any number of threads, and identical to v0.9.
- `ddma.set_num_threads(n)`, `ddma.get_num_threads()`, `ddma.openmp_available()`; MATLAB `ddma.set_num_threads(n)`, `ddma.get_num_threads()`.
- Faster on one thread too: monotone maps sort their writes by block of the grid so they stay in the cache, and the sweeps read memory in order.
- Builds: GCC/Clang (`-fopenmp`), Microsoft's compiler (`/openmp`), Apple's compiler with Homebrew's `libomp`; without OpenMP the kernels run on one thread with the same results (`DDMA_NO_OPENMP=1`, `build_mex(false)`). Errors inside the threads (e.g. out of memory) are raised normally.
- Tests `tests/test_threads.py`, `matlab/tests/run_threads_tests.m`; `docs/PERFORMANCE.md`; `benchmarks/kernels.py`.

## 0.9.0 (14 November 2026)

**Safe envelopes from data, without a grid** (`ddma.invariance`; methods of Makdesi & Zamani and Makdesi, Zamani & Jafarpour, IFAC World Congress 2026)
- `monotone_envelope`: controlled invariant set of a monotone system with an additive bounded disturbance and a lower-closed specification, as maximal corners of lower sets, each with its largest certified input. Layered search with dominance pruning and a sort-filter skyline; returns exactly the corners of exhaustive enumeration. `state_signs` / `input_signs` for systems that are monotone after flipping coordinates.
- `embedding_envelope`: invariant boxes spanned by pairs of data points, for any system with interval bounds on its Jacobian (states and inputs) and an additive bounded disturbance. The pair LP (maximum over the box of the minimum of two growth cones) is solved exactly in closed form (`max_min_cones`), all pairs in vectorized blocks.
- Envelopes answer `contains(x)` and `safe_input(x, u_desired)` (the certified input closest to the desired one).
- Examples: `envelope_room_cooling.py` (1,000 transitions certify about 98 % of the safe temperature range; the grid abstraction certifies about 74 % from 200,000) and `envelope_nonlinear.py` (two-species competition model, ring of ten coupled units). Figures in `docs/figures/`, notes in `docs/INVARIANCE.md`.
- MATLAB: `ddma.monotone_envelope`, `ddma.embedding_envelope`, `ddma.envelope_contains`, `ddma.envelope_input`, `ddma.maximal_points`, `ddma.max_min_cones`, examples of the same names; tested against Python (`matlab/tests/run_invariance_tests.m`).
- Authors: Majid Zamani and Saber Jafarpour added as co-authors of these methods (README, `CITATION.cff`, `pyproject.toml`).

## 0.8.0 (14 November 2026)

**Safety filter for reinforcement learning**
- `ddma.rl.SafetyFilterWrapper` (Gymnasium): every action goes through `Controller.filter` (unchanged if safe, nearest safe action otherwise); initial states redrawn into the controller's domain (`reset_in_domain=True`); optional reward `penalty` per unit of correction; `info` reports the safe action, whether it was corrected, and by how much.
- `ddma.rl.CruiseControlEnv`: the adaptive cruise control problem of the TAC paper as an RL task.
- `examples/rl_safety_filter.py`: PPO (Stable-Baselines3), 3 seeds × 100,000 steps: 266–319 violations in training without the filter, 0 with it (and 0 in evaluation), slightly higher return. Figure `docs/figures/rl_safety_filter.png`.
- MATLAB: `ddma.safe_step`, `ddma.safe_reset` (step/reset functions in the form of the Reinforcement Learning Toolbox's `rlFunctionEnv`; no toolbox needed), `examples/cruise_control_env.m`, `examples/rl_safety_filter.m`; filter tested against Python on 300 random states.
- Extras: `pip install "ddma[rl]"` (gymnasium), `"ddma[rl-demo]"` (+ stable-baselines3, matplotlib).

## 0.7.0 (14 November 2026)

**Python and MATLAB at parity.** Every feature is now in both interfaces, tested against each other; `docs/PARITY.md` maps each Python name to its MATLAB counterpart.

New in MATLAB / Octave:
- Lazy abstractions: `ddma.abstraction(..., lazy)`; `ddma.safety` runs the same on-demand pointer scan as Python (transitions evaluated only when needed; `allowed` is then a struct and `ddma.controller_input` computes the allowed inputs of a state on demand). Example `lazy_synthesis.m`.
- `ddma.estimate_derivative_bounds`, `ddma.scenario_confidence` (TAC Sec. IV; LP with linprog or glpk).
- Per-output `features` in `ddma.bd_fit` (each output learned on its own sub-grid; handled by `bd_update`, `bd_update_local`, `abstraction`, `abstraction_refresh`, `bounds`).
- `ddma.growth_map`, `ddma.f_bounds`; `ddma.fit_update`, `ddma.intersect`, `ddma.compact` and `ddma.fit(..., compact)` (single, rounded outward), `ddma.contains`, `ddma.widths`, `ddma.bounded_fraction`, `ddma.is_empty`.
- `ddma.msm_f_bounds`, `ddma.msm_estimate`, `ddma.msm_is_consistent`; `ddma.simulate`, `ddma.cells_meeting`, `ddma.transitions_to_data`, `ddma.data_partition`.
- Examples `scalar_map.m`, `bounded_derivative_map.m`, `lazy_synthesis.m` (all Python examples now have a MATLAB version).
- Internal helpers moved to `+ddma/+internal`.

Python:
- `BoundedDerivativeModel(method="two_matrix")`: a coordinate whose derivative bounds are both zero is treated as one the output does not depend on (sign 0) instead of as increasing. Tighter (cells beyond the data are now bounded too), still sound, and equal to the two-pattern construction and to MATLAB in that case. No change in the reproductions or the pendulum example; in `examples/bounded_derivative_map.py` approach 1 now bounds the whole interval.

## 0.6.0 (14 November 2026)

- Growth-cone fits in the shared C++ core: the vertex values (scatter + one forward and one backward running min/max per axis) and the `local="cell"` / `"slab"` maps. The slab map is computed per "pattern" (left of / in / right of the point's cell on each axis) with one scatter and exclusive running min/max sweeps, so its cost no longer depends on the slab size. Same model as NumPy (within ~1e-15); 10 million points on a 50 x 50 grid: about 0.3 s, 0.6 s with `local="slab"` (NumPy: about 5 s, and the slab map did not fit in memory). `GrowthMap`, `GrowthComponent` and the growth methods of `BoundedDerivativeModel` take `backend="auto" | "cpp" | "numpy"`.
- MATLAB / Octave: `ddma.bd_fit` with the five bounded-derivative models (`'two_matrix'`, `'all_matrix'`, `'lin_mono'`, `'growth'`, `'lin_growth'`, and `local` `'cell'` / `'slab'`), in plain MATLAB and through the new MEX `ddma_growth_mex` (same C++ kernels as Python). It returns the same struct as `ddma.fit`, so `ddma.abstraction`, `ddma.safety` and `ddma.reach_avoid` take it unchanged. LP final steps use `linprog` (MATLAB) or `glpk` (Octave), with the bounds certified by weak duality as in Python (`ddma.max_min_affine`). Tested against Python on 54 random cases (all methods, 1-D to 3-D) and on a pendulum safety problem (same invariant sets). New example `matlab/examples/pendulum_bounded_derivatives.m`.
- MATLAB / Octave online refinement, as in Python: `ddma.bd_update` (exact: same model as a refit), `ddma.bd_update_local` (growth methods: a window of cells around each point; changed boxes recorded in `S.changed`), `ddma.abstraction_refresh` (recomputes the changed states only; the abstraction keeps its options in `A.opts`), and `ddma.safety(A, safe, backend, known_winning, check)` (incremental update, also in the MEX). Tested against Python on an online sequence (fit, exact update, windowed update, refresh, incremental safety: same bounds, transitions and sets) and for update = refit on all five methods. New example `matlab/examples/online_refinement.m`: the safe region grows from about 50 % to 80 % of the specification in 1000 steps, 0 violations.

## 0.5.0 (14 November 2026)

- `GrowthMap`: the growth-cone model of one scalar function with bounded derivatives, usable on its own (`from_data`, `update`, `update_local`, `bounds`, `f_bounds`, `copy`). Same bounds as `BoundedDerivativeModel(method="growth")` for one output; `local="slab"` by default.
- Growth-cone models: small batches (online use) are added point by point, each cone evaluated directly at all vertices and on its slabs, and the all-cell final step reuses per-corner offsets. Same model as before (tested against the batch refit); a single point on a 40 x 40 grid now takes about 0.1 ms instead of 0.6 ms.

## 0.4.0 (14 November 2026)

**Models of systems with bounded derivatives** (thesis, Chapter 3)
- `BoundedDerivativeModel(method=...)` offers the five approaches of the thesis: `"two_matrix"` (default, unchanged), `"all_matrix"`, `"lin_mono"`, `"growth"`, `"lin_growth"`. All plug into `SymbolicAbstraction`, `safety` and `reach_avoid` unchanged.
- Growth cones: exact bounds at the grid vertices with one forward and one backward running min/max per axis.
- Linear-programming final steps: bounds certified by weak duality, sound whatever the solver tolerance.
- `local="cell"` / `"slab"` (growth methods): intersection with the cones of the data points in each cell / in its slabs.
- `update()` refits exactly; `update_local()` (growth methods) updates a window of cells around each new point.

**Online refinement**
- `SymbolicAbstraction.refresh()`: recomputes only the transitions of the states changed by `update_local`.
- `safety(..., known_winning=previous_set)`: exact update of the maximal controlled invariant set when the model was refined (C++, NumPy and lazy backends).

**Examples and reproductions**
- `examples/pendulum_bounded_derivatives.py`: safety for a non-monotone system with the five models.
- `reproductions/thesis_ch3.py`: thesis Section 3.4 (Figs. 3.5-3.9); see `docs/REPRODUCTION.md`, note 6.

Python only for now; the MATLAB interface is unchanged.

## 0.3.0

Exact safety and reach-avoid fixed points in the shared C++ core (Python and MATLAB); lazy abstractions.
