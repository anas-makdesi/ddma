# Roadmap

Planned work, roughly in order. Contributions and suggestions are welcome through issues.

## Applications
- **Safe learning-based MPC**: the online refinement of the model and the safety controller is in v0.4 (`update_local`, `refresh`, incremental `safety`); the compatible estimates and the MPC that uses them (compatible-models approach) are developed in a separate package that uses `GrowthMap` (v0.5).

## Methods
- **Safe envelopes in the safety filters**: use the gridless envelopes of v0.9 as the safe set of the RL safety filter and of the safe learning-based MPC.
- **Richer specifications**: reach-while-stay, recurrence, and specifications given by automata.

## Performance
- Streaming fit with constant memory for any number of samples.
- Non-uniform and adaptive partitions that refine where the specification is decided.

## Distribution
Pre-built wheels and the MATLAB toolbox file are available since 0.12.1. Still to come:
- Wheels for macOS 14 on Apple silicon (today pip builds from source there).
- Wheels for musl-based Linux (Alpine) and for free-threaded Python.
