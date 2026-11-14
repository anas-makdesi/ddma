# Changelog

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
