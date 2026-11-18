"""ddma: data-driven models and abstractions.

Implements the minimal simulating maps of

    A. Makdesi, A. Girard, L. Fribourg, "Data-driven models of monotone systems",
    IEEE Transactions on Automatic Control,

models of systems with bounded derivatives (TAC Sec. III-C and the five approaches of
A. Makdesi, "Data-driven abstractions for the safe learning of nonlinear systems",
PhD thesis, Universite Paris-Saclay, 2023, Chapter 3), and the finite abstractions and
controller synthesis built on them, safe envelopes (controlled invariant sets)
computed from data without a grid (A. Makdesi, M. Zamani, S. Jafarpour, IFAC World
Congress 2026), and an online monitor of the models' assumptions (:mod:`ddma.monitor`).
"""

from .abstraction import AffineBounds, SymbolicAbstraction, affine_bounds, transitions_to_data
from .bounded_derivatives import (
    BoundedDerivativeModel,
    estimate_derivative_bounds,
    scenario_confidence,
)
from .growth import GrowthMap
from .invariance import (
    BoxEnvelope,
    MonotoneEnvelope,
    embedding_envelope,
    monotone_envelope,
)
from .monitor import AssumptionMonitor, DataCheck, Violation, check_data
from .msm import MinimalSimulatingMap
from .partition import RectangularPartition
from .partitioned import PartitionedSimulatingMap, compiled_backend_available
from .synthesis import Controller, reach_avoid, safety, simulate
from .threads import get_num_threads, openmp_available, set_num_threads

__version__ = "0.12.0"

__all__ = [
    "AssumptionMonitor",
    "Violation",
    "DataCheck",
    "check_data",
    "BoundedDerivativeModel",
    "GrowthMap",
    "MonotoneEnvelope",
    "BoxEnvelope",
    "monotone_envelope",
    "embedding_envelope",
    "estimate_derivative_bounds",
    "scenario_confidence",
    "Controller",
    "MinimalSimulatingMap",
    "PartitionedSimulatingMap",
    "compiled_backend_available",
    "get_num_threads",
    "openmp_available",
    "set_num_threads",
    "RectangularPartition",
    "SymbolicAbstraction",
    "AffineBounds",
    "affine_bounds",
    "reach_avoid",
    "safety",
    "simulate",
    "transitions_to_data",
]
