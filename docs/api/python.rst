Python
======

Everything below is importable from ``ddma`` unless the section names a submodule.

.. currentmodule:: ddma

Data and partitions
-------------------

.. autoclass:: RectangularPartition

.. autofunction:: transitions_to_data

Models of monotone systems
--------------------------

.. autoclass:: PartitionedSimulatingMap

.. autoclass:: MinimalSimulatingMap

Models of systems with bounded derivatives
------------------------------------------

.. autoclass:: BoundedDerivativeModel

.. autoclass:: GrowthMap
   :inherited-members:

.. autofunction:: estimate_derivative_bounds

.. autofunction:: scenario_confidence

Abstraction
-----------

.. autoclass:: SymbolicAbstraction

.. autofunction:: affine_bounds

.. autoclass:: AffineBounds
   :special-members: __call__

Synthesis and closed loop
-------------------------

.. autofunction:: safety

.. autofunction:: reach_avoid

.. autoclass:: Controller
   :special-members: __call__

.. autofunction:: simulate

Safe envelopes without a grid (``ddma.invariance``)
---------------------------------------------------

.. automodule:: ddma.invariance
   :no-members:

.. currentmodule:: ddma

.. autofunction:: monotone_envelope

.. autofunction:: embedding_envelope

.. autoclass:: MonotoneEnvelope

.. autoclass:: BoxEnvelope

.. autofunction:: ddma.invariance.maximal_points

.. autofunction:: ddma.invariance.max_min_cones

Assumption monitor
------------------

.. automodule:: ddma.monitor
   :no-members:

.. currentmodule:: ddma

.. autofunction:: check_data

.. autoclass:: ddma.monitor.DataCheck

.. autoclass:: AssumptionMonitor

.. autoclass:: Violation

.. autofunction:: ddma.monitor.slopes_from_signs

Reinforcement learning (``ddma.rl``)
------------------------------------

Needs ``gymnasium`` (``pip install "ddma[rl]"``).

.. automodule:: ddma.rl
   :no-members:

.. currentmodule:: ddma

.. autoclass:: ddma.rl.SafetyFilterWrapper

.. autoclass:: ddma.rl.CruiseControlEnv

Threads and backends
--------------------

.. autofunction:: set_num_threads

.. autofunction:: get_num_threads

.. autofunction:: openmp_available

.. autofunction:: compiled_backend_available

Plotting (``ddma.plotting``)
----------------------------

Needs ``matplotlib``.

.. automodule:: ddma.plotting
   :members:
