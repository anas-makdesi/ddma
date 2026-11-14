"""ddma: data-driven abstractions of monotone systems.

Implements the minimal simulating maps of

    A. Makdesi, A. Girard, L. Fribourg, "Data-driven models of monotone systems",
    IEEE Transactions on Automatic Control,

and the finite abstractions and controller synthesis built on them.
"""

from .abstraction import SymbolicAbstraction, affine_bounds, transitions_to_data
from .bounded_derivatives import (
    BoundedDerivativeModel,
    estimate_derivative_bounds,
    scenario_confidence,
)
from .msm import MinimalSimulatingMap
from .partition import RectangularPartition
from .partitioned import PartitionedSimulatingMap, compiled_backend_available
from .synthesis import Controller, reach_avoid, safety, simulate

__version__ = "0.3.0"

__all__ = [
    "BoundedDerivativeModel",
    "estimate_derivative_bounds",
    "scenario_confidence",
    "Controller",
    "MinimalSimulatingMap",
    "PartitionedSimulatingMap",
    "compiled_backend_available",
    "RectangularPartition",
    "SymbolicAbstraction",
    "affine_bounds",
    "reach_avoid",
    "safety",
    "simulate",
    "transitions_to_data",
]
