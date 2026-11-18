# Installation

## Python

ddma needs Python 3.10 or newer and NumPy.

```bash
pip install ddma                # pre-built for Linux, macOS and Windows, threads included
pip install "ddma[examples]"    # + matplotlib, scipy (examples, derivative-bound estimation)
pip install "ddma[rl]"          # + gymnasium (the safety filter for reinforcement learning)
```

Pre-built packages (wheels) exist for CPython 3.10 to 3.15 on Linux (x86_64 and ARM),
macOS (Apple silicon and Intel) and Windows (x86_64). They contain the compiled C++ kernels
and the OpenMP runtime, so nothing else is needed, also on macOS. On other systems pip builds
the kernels from source.

From a clone of the repository, to run the examples and tests or to change the code:

```bash
git clone https://github.com/anas-makdesi/ddma && cd ddma
pip install -e ".[dev]"         # + pytest, ruff; builds the C++ kernels
pip install -e ".[docs]"        # + everything needed to build this site and run the tutorials
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
speed. Limit it with {func}`ddma.set_num_threads`. When building from source on macOS, run
`brew install libomp` once first, otherwise the kernels run on one thread. See
{doc}`../guides/performance`.

## MATLAB and Octave

**Toolbox (MATLAB R2023b or newer).** Download `ddma-<version>.mltbx` from the
[releases page](https://github.com/anas-makdesi/ddma/releases) and open it in MATLAB, or run
`matlab.addons.install("ddma-<version>.mltbx")`. It contains the `+ddma` package, the
examples, and pre-built MEX files (with threads) for Linux, Windows and macOS on Apple
silicon. On other systems the `.m` code runs, with the same results.

**From a clone (MATLAB or Octave).** The `matlab/` folder holds the `+ddma` package: the same
methods as functions on structs, with the same results (tested against Python in CI). From
that folder:

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
