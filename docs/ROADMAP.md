# Roadmap

Planned work, roughly in order. Contributions and suggestions are welcome through issues.

## Applications
- **Safety filter for reinforcement learning**: a Gymnasium wrapper that puts a certified ddma safety filter around any agent, with a demo where a PPO agent learns while staying safe.
- **Online refinement and safe learning-based MPC**: refine the model and the safety controller while the system runs, as data arrive (compatible-models approach).

## Methods
- **More models for systems with bounded derivatives**: growth-cone models evaluated on the cell vertices, transformations with all sign combinations of the derivative bounds, and local updates when new data arrive.
- **Partition optimisation**: choose the cut points of the rectangular partition to minimise the over-approximation for a given number of cells (CCTA 2025).
- **Controlled invariance via monotone embeddings**: data-driven controlled invariant sets for systems that are not monotone.
- **Richer specifications**: reach-while-stay, recurrence, and specifications given by automata.

## Performance
- Multithreaded learning kernel (OpenMP).
- Streaming fit with constant memory for any number of samples.
- Non-uniform and adaptive partitions that refine where the specification is decided.
- Lazy abstractions in the MATLAB interface (Python has them since v0.3).

## Distribution
- Documentation site with API reference and tutorial notebooks.
- Pre-built wheels on PyPI, so no compiler is needed.
- MATLAB toolbox file (.mltbx) with pre-built MEX files for Windows, macOS and Linux.
