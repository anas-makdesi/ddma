# ddma: data-driven models and abstractions

**Learn a guaranteed model of an unknown system from data, then build controllers that are
provably safe for the real system.**

ddma takes measured transitions of a system whose dynamics are partly unknown and returns a
*set-valued* model that is guaranteed to contain the true dynamics. It needs no model class:
only a structural assumption (the unknown part is **monotone**, or its **derivatives are
bounded**) and a bound on the disturbance. From that model it builds a finite abstraction and
synthesizes controllers (safety, reach-avoid) whose guarantees carry over to the real system.
It can also compute **safe operating envelopes directly from the data**, without a grid.

```{image} figures/scalar_map.png
:alt: Tightest model consistent with 25 noisy samples of an unknown monotone map
:width: 560px
:align: center
```

## The pipeline in four steps

::::{grid} 1 2 4 4
:gutter: 2

:::{grid-item-card} 1. Data
Transitions $(x_k, u_k, x_k^+)$ of the real system: logged, simulated or measured online.
:::

:::{grid-item-card} 2. Model
The tightest set-valued model consistent with the data and the assumptions.
:::

:::{grid-item-card} 3. Abstraction
A finite system on a grid of cells that over-approximates every possible behaviour.
:::

:::{grid-item-card} 4. Controller
The maximal safe set and the inputs that keep you in it: a controller, or a safety filter
around any other one.
:::
::::

## Where to start

- New to ddma: read {doc}`getting-started/concepts`, then work through the
  {doc}`tutorials <tutorials/index>` in order. Each one builds on the previous one.
- Working in MATLAB: every function has a MATLAB counterpart, see {doc}`guides/parity`.
  The tutorials show the MATLAB calls next to the Python ones.
- Looking for a function: {doc}`api/index`.

## Install

```bash
pip install -e .     # from a clone; builds the C++ kernels if a compiler is available
```

Details, including MATLAB and OpenMP on macOS: {doc}`getting-started/installation`.

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/concepts
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
tutorials/01-scalar-map
tutorials/02-cruise-control
tutorials/03-pendulum
tutorials/04-online-learning
tutorials/05-rl-safety-filter
tutorials/06-gridless-envelopes
tutorials/07-assumption-monitor
```

```{toctree}
:hidden:
:caption: Guides

guides/parity
guides/envelopes
guides/monitor
guides/performance
guides/reproduction
```

```{toctree}
:hidden:
:caption: API reference

api/index
api/python
api/matlab
```

```{toctree}
:hidden:
:caption: About

about/citing
about/changelog
about/roadmap
```
