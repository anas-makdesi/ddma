# Speed: threads in the learning kernels and the abstraction

The kernels that learn models from data, and the one that builds the abstraction, run on
several threads (OpenMP) in both Python and MATLAB, because both use the same C++ core
(`cpp/`):

| Kernel | Used by |
|---|---|
| `fit_sigma` (`cpp/ddma_core.hpp`) | monotone maps: `PartitionedSimulatingMap`, `BoundedDerivativeModel` (`two_matrix`, `all_matrix`, `lin_mono`); MATLAB `ddma.fit`, `ddma.bd_fit` |
| `growth_vertices`, `growth_local` (`cpp/ddma_growth.hpp`) | growth cones: `GrowthMap`, `BoundedDerivativeModel` (`growth`, `lin_growth`, `local=`); MATLAB `ddma.growth_map`, `ddma.bd_fit` |
| abstraction (`cpp/ddma_abstraction.hpp`, since v0.11) | `SymbolicAbstraction` (eager, lazy and `refresh`); MATLAB `ddma.abstraction`, `ddma.abstraction_refresh`, lazy transitions |

**Results are bit-identical for any number of threads.** The tests check this for 1, 2 and
all threads (`tests/test_threads.py`, `tests/test_abstraction_backends.py`,
`matlab/tests/run_threads_tests.m`, `matlab/tests/run_abstraction_tests.m`).

## Setting the number of threads

Default: the value of `OMP_NUM_THREADS` if it is set, otherwise all cores.

```python
import ddma
ddma.set_num_threads(4)      # 4 threads
ddma.set_num_threads(None)   # back to the default
ddma.get_num_threads(), ddma.openmp_available()
```

```matlab
ddma.set_num_threads(4);          % 4 threads
ddma.set_num_threads();           % back to the default
[n, openmp] = ddma.get_num_threads();
```

OpenMP is optional. Where the compiler has no OpenMP, the kernel is built without it and
runs on one thread, with the same results. `DDMA_NO_OPENMP=1 pip install .` forces such a
build; in MATLAB, `build_mex(false)`.

**macOS.** Apple's compiler needs the OpenMP library from Homebrew. Install it once, then
install ddma (or run `build_mex` in MATLAB):

```
brew install libomp
pip install .        # finds libomp by itself; another folder: LIBOMP_PREFIX=/path pip install .
```

Without `libomp` the build still works, on one thread, and prints a note.

## How the kernels work (in short)

Each kernel has two phases.

1. **Scatter.** Every data point writes a few numbers into a large grid array. A cell
   keeps the smallest (upper bounds) or largest (lower bounds) number written to it.
2. **Sweeps.** A running minimum or maximum along each axis of the grid spreads these
   numbers to the other cells.

Why threads cannot change the result: a minimum or maximum is exact (no rounding) and does
not depend on the order of its inputs, so it does not matter which thread writes first;
and in the sweeps every grid line is handled by one thread, with the same operations in
the same order as on one thread.

How the scatter is shared among threads (`cpp/ddma_parallel.hpp`, chosen automatically):

- **bucketed** (`fit_sigma` on grids larger than a few MB, also on one thread): the values
  are first sorted by block of the grid (a block fits in the processor's cache), then each
  block is filled by one thread. Writing into a block that sits in cache is much faster
  than writing all over a big array.
- **owned parts** (growth cones on large grids, which write many values per point): the
  grid is cut into parts; points are listed under the parts they write to; each thread
  fills whole parts, so no two threads write to the same cell.
- **private copies** (small grids): each thread fills its own copy of the grid with its
  share of the points; the copies are then merged.
- **direct** (growth cones on one thread, and small updates): each value is written
  straight into the grid, as in v0.8. Online updates with a few points always use this
  path.

The sweeps visit memory contiguously: all outputs of a monotone map in one pass, and, for
`growth_local`, all patterns of a cell together. This alone makes some kernels faster on
one thread.

## The abstraction kernel

For every state-input cell pair the abstraction needs the successor box (the map's bounds
at the pair, plus the bounds of the known part g), clipped to an assumption box if there is
one, then the range of cells the box meets. In NumPy this goes through several arrays of
the size of all pairs; the kernel does it pair by pair without building them:

- the map's bounds are read at `offq[q] + offp[p]`, two small offset tables computed once
  from the coordinates the map depends on (`features`, per-output features);
- g from `affine_bounds` is kept as a state part and an input part (`AffineBounds.parts`),
  added per pair with the same operations as `affine_bounds` itself; any other `g_bounds`
  function is evaluated in Python per block of pairs and passed in;
- for the two-matrix bounded-derivative model, both maps and both affine parts are combined
  per pair (`max(gm + sigma_minus, gp - sigma_plus)`); other models compute their boxes in
  Python and the kernel does the rest;
- the cell range on each axis comes from a table of equal-width buckets plus a short exact
  scan, so the cost does not grow with the number of cells (and uneven partitions cost the
  same as uniform ones).

Pairs do not depend on each other: threads split the state cells and write to disjoint
parts of the output. The arithmetic is that of the NumPy code (same operations, same
order), so both backends give identical abstractions; `backend="numpy"` (MATLAB
`'matlab'`) keeps the old code.

Measured in a 2-core cloud VM (`python benchmarks/abstraction.py`; best of 3, rounded):

| Workload | NumPy | C++, 1 thread | C++, 2 threads |
|---|---:|---:|---:|
| monotone map, uneven 301² states × 301 inputs (27 M pairs) | 6.7 s | 0.55 s | 0.36 s |
| two-matrix model with features (Lorenz), 51³ × 51 (6.8 M pairs) | 4.0 s | 0.32 s | 0.17 s |
| map + affine g + assumption, uneven 60³ × 12 (2.6 M pairs) | 1.1 s | 0.18 s | 0.10 s |

In MATLAB (Octave, 150² × 150 = 3.4 M pairs) the MEX kernel takes 0.2 s on one thread
against 0.9 s for the `.m` code. With a `g_bounds` function, evaluating g in MATLAB takes
most of the time (0.4 s against 0.6 s).

## Measuring

```
python benchmarks/kernels.py               # all threads
python benchmarks/kernels.py --threads 1   # one thread
python benchmarks/kernels.py --threads 1 --mode direct   # the v0.8 scatter
```

The script splits each kernel's time into scatter and sweeps. Gains depend on the machine
(cores, cache sizes, memory bandwidth): the scatter is limited by memory access, so it
gains less from threads than the sweeps.

For the abstraction:

```
python benchmarks/abstraction.py              # C++ on all threads, and NumPy
python benchmarks/abstraction.py --threads 1
```
