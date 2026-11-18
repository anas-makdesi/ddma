"""Run on every built wheel (cibuildwheel's test step, and again on clean machines).

Fails unless the installed ddma has its compiled C++ kernels and they were built with
OpenMP (so the kernels run on several threads), and prints where ddma was imported from,
so the log shows that the wheel was tested and not the source folder.
"""

import sys

import ddma

print(f"ddma {ddma.__version__} from {ddma.__file__}")
print(f"compiled kernels: {ddma.compiled_backend_available()}, "
      f"OpenMP: {ddma.openmp_available()}, threads: {ddma.get_num_threads()}")
if not (ddma.compiled_backend_available() and ddma.openmp_available()):
    sys.exit("the wheel lacks the compiled kernels or OpenMP")
