# Tutorials

Six notebooks that walk through the toolbox in the order its ideas build on each other. Each
one runs in a few seconds (they are executed every time this site is built, with small
settings), and each explains *why* as well as *how*.

| | tutorial | you learn |
|---|---|---|
| 1 | {doc}`01-scalar-map` | what a data-driven model with guarantees is: the minimal simulating map of a scalar monotone function, and the same on a grid |
| 2 | {doc}`02-cruise-control` | the full pipeline on a real problem: data → model → abstraction → certified safe controller → safety filter |
| 3 | {doc}`03-pendulum` | systems that are not monotone: bounded-derivative models, the five methods, `GrowthMap` |
| 4 | {doc}`04-online-learning` | learning while running: local updates, `refresh`, `known_winning`, lazy abstractions |
| 5 | {doc}`05-rl-safety-filter` | a reinforcement-learning agent inside a certified safety filter |
| 6 | {doc}`06-gridless-envelopes` | safe envelopes straight from data, without a grid |

Each tutorial ends with the MATLAB version of its main calls.
