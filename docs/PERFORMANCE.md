# Speed: threads in the learning kernels

The kernels that learn models from data run on several threads (OpenMP) in both Python and
MATLAB, because both use the same C++ core (`cpp/`):

| Kernel | Used by |
|---|---|
| `fit_sigma` (`cpp/ddma_core.hpp`) | monotone maps: `PartitionedSimulatingMap`, `BoundedDerivativeModel` (`two_matrix`, `all_matrix`, `lin_mono`); MATLAB `ddma.fit`, `ddma.bd_fit` |
| `growth_vertices`, `growth_local` (`cpp/ddma_growth.hpp`) | growth cones: `GrowthMap`, `BoundedDerivativeModel` (`growth`, `lin_growth`, `local=`); MATLAB `ddma.growth_map`, `ddma.bd_fit` |

**Results are bit-identical for any number of threads.** The tests check this for 1, 2 and
all threads (`tests/test_threads.py`, `matlab/tests/run_threads_tests.m`).

## Setting the number of threads

Default: the value of `OMP_NUM_THREADS` if it is set, otherwise all cores.

```python
from ddma.threads import set_num_threads, get_num_threads, openmp_available
set_num_threads(4)      # 4 threads
set_num_threads(None)   # back to the default
```

```matlab
ddma.set_num_threads(4);          % 4 threads
ddma.set_num_threads();           % back to the default
[n, openmp] = ddma.get_num_threads();
```

OpenMP is optional. Where the compiler has no OpenMP (Apple clang without `libomp`), the
kernel is built without it and runs on one thread, with the same results.
`DDMA_NO_OPENMP=1 pip install .` forces such a build; in MATLAB, `build_mex(false)`.

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

## Measuring

```
python benchmarks/kernels.py               # all threads
python benchmarks/kernels.py --threads 1   # one thread
python benchmarks/kernels.py --threads 1 --mode direct   # the v0.8 scatter
```

The script splits each kernel's time into scatter and sweeps. Gains depend on the machine
(cores, cache sizes, memory bandwidth): the scatter is limited by memory access, so it
gains less from threads than the sweeps.
