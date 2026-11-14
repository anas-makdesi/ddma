# Changelog

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
