"""Number of threads used by ddma's compiled learning kernels.

The kernels that learn models from data (the monotone maps of ``PartitionedSimulatingMap``
and ``BoundedDerivativeModel``, and the growth cones of ``GrowthMap`` and the growth
methods) run on several threads when the compiled kernel was built with OpenMP.

Results are **bit-identical for any number of threads**: the kernels only keep minima and
maxima of the data, which do not depend on the order in which threads write them, and
every running min/max along the grid is computed by one thread in a fixed order.

By default the kernels use OpenMP's default: the value of the environment variable
``OMP_NUM_THREADS`` if it is set, otherwise all cores.

>>> from ddma.threads import set_num_threads, get_num_threads
>>> set_num_threads(4)      # use 4 threads
>>> set_num_threads(None)   # back to the default
"""

from __future__ import annotations

from .partitioned import _core

__all__ = ["set_num_threads", "get_num_threads", "openmp_available"]


def set_num_threads(n: int | None = None) -> None:
    """Use ``n`` threads in the compiled kernels (``None`` or 0: OpenMP's default).

    Has no effect when the compiled kernel is not installed or was built without OpenMP
    (then everything runs on one thread, with the same results).
    """
    if n is None:
        n = 0
    n = int(n)
    if n < 0:
        raise ValueError("number of threads must be >= 0")
    if _core is not None:
        _core.set_num_threads(n)


def get_num_threads() -> int:
    """Number of threads the compiled kernels will use (1 without OpenMP)."""
    return int(_core.get_num_threads()) if _core is not None else 1


def openmp_available() -> bool:
    """Whether the compiled kernel was built with OpenMP (threads)."""
    return bool(_core is not None and _core.openmp_available())
