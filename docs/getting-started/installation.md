# Installation

## Python

ddma needs Python 3.9 or newer and NumPy. From a clone of the repository:

```bash
pip install -e .              # NumPy only; builds the C++ kernels if a compiler is available
pip install -e ".[examples]"  # + matplotlib, scipy (examples, derivative-bound estimation)
pip install -e ".[rl]"        # + gymnasium (the safety filter for reinforcement learning)
pip install -e ".[docs]"      # + everything needed to build this site and run the tutorials
```

Check what you got:

```python
import ddma
ddma.compiled_backend_available()   # True: the C++ kernels were built
ddma.openmp_available()             # True: they run on several threads
ddma.get_num_threads()              # all cores by default
```

**Without a C++ compiler** the package falls back to NumPy. The results are identical, only
slower.

**Threads.** The compiled kernels learn models and build abstractions on all cores (OpenMP).
The results are bit-identical for any number of threads, so the setting only changes the
speed. Limit it with {func}`ddma.set_num_threads`. On macOS, run `brew install libomp` once
before installing, otherwise the kernels run on one thread. See
{doc}`../guides/performance`.

## MATLAB and Octave

The `matlab/` folder holds the `+ddma` package: the same methods as functions on structs,
with the same results (tested against Python in CI). From that folder:

```matlab
build_mex                 % optional: compiles the same C++ kernels as MEX files (with threads)
addpath tests; run_tests  % checks the MATLAB code against Python reference results
run examples/acc_safety_filter
```

Without `build_mex` the pure `.m` code runs, with the same results. Each Python name and its
MATLAB counterpart are listed in {doc}`../guides/parity`.

## Building this site

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The tutorials are notebooks stored as Markdown (MyST). They are executed during the build,
so the outputs you read are the outputs of the current code. To open one in Jupyter, install
`jupytext` (included in `[docs]`): Jupyter then opens the `.md` file as a notebook, or convert
it with `jupytext --to ipynb docs/tutorials/01-scalar-map.md`.
