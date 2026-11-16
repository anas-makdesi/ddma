# Core ideas

This page fixes the vocabulary used everywhere else: the kind of system ddma handles, what you
must know about it, what ddma computes, and what each result guarantees.

## The system

ddma handles discrete-time systems of the form

$$
x^+ \in f(x, u) + g(x, u) + D, \qquad D = [d_{lo}, d_{hi}],
$$

- $x \in \mathbb{R}^n$ is the state and $u \in \mathbb{R}^p$ the input;
- $f$ is **unknown**: it is what ddma learns from data;
- $g$ is an optional **known** part (often the kinematics, e.g. $\theta^+ = \theta + T\omega$),
  passed as a function for the data and as {func}`~ddma.affine_bounds` for the abstraction;
- $D$ is a **bounded disturbance** with known bounds, which can also absorb external
  inputs such as another vehicle's speed.

The data are transitions $(x_k, u_k, x_k^+)$, $k = 1, \dots, K$. With $z = (x, u)$ and
$y = x^+ - g(x, u)$ ({func}`~ddma.transitions_to_data`) every data point satisfies
$y_k \in F(z_k) = f(z_k) + D$, and learning $f$ becomes learning a set-valued map $F$ from
pairs $(z_k, y_k)$.

## What you must know about $f$

ddma fits no model class. Instead it uses one structural assumption about $f$:

| assumption | meaning | where it comes from | ddma class |
|---|---|---|---|
| **monotone** | $z \le z' \Rightarrow f(z) \le f(z')$ (componentwise); each dependence can also be decreasing or absent (`signs=`) | physics: more heat in, higher temperature; more throttle, more speed | {class}`~ddma.PartitionedSimulatingMap`, {class}`~ddma.MinimalSimulatingMap` |
| **bounded derivatives** | $\partial f_i/\partial z_j \in [a^{lo}_{ij}, a^{hi}_{ij}]$ | physics, or estimated from data with a probabilistic guarantee ({func}`~ddma.estimate_derivative_bounds`) | {class}`~ddma.BoundedDerivativeModel`, {class}`~ddma.GrowthMap` |

plus the disturbance bounds $d_{lo}, d_{hi}$. **Every guarantee below holds as long as these
assumptions hold.** If the real system is not monotone, or a derivative leaves its bounds, the
model can exclude the truth and the guarantees are lost.

## Step 1: the tightest model consistent with the data

For a monotone $f$, the **minimal simulating map** (TAC paper) is

$$
S_m(z) = \Big[\, \sup_{z_k \le z} \big(y_k - \theta\big),\;
                 \inf_{z_k \ge z} \big(y_k + \theta\big) \,\Big], \qquad \theta = d_{hi} - d_{lo}.
$$

Read it as: a data point *below* $z$ is a floor for every monotone function at $z$ (minus the
width of the disturbance, because the measurement may have been lucky); a data point *above* $z$
is a ceiling. Two properties make it the right object:

- **sound**: $F(z) \subseteq S_m(z)$ for every monotone system that could have produced the
  data, so in particular for the true one;
- **minimal**: no smaller set-valued map has that guarantee.

Where no data point lies below (above) $z$, the lower (upper) bound is $-\infty$ ($+\infty$).

In practice ddma computes the tightest such bound **per cell of a grid**
({class}`~ddma.RectangularPartition`), which is what the abstraction needs: one interval per
cell, computed for all cells at once in $O(|K| \log |Q| + |Q|)$ time, and refined by
intersection when new data arrive. For bounded-derivative systems, each data point bounds $f$
inside a cone set by the derivative bounds instead; {doc}`../tutorials/03-pendulum` compares
the five ways ddma does this.

## Step 2: a finite abstraction

The state and input spaces are cut into cells, $q$ for states and $p$ for inputs. For every
pair $(q, p)$ the model gives a box that contains every possible successor of every state in
$q$ under every input in $p$; the abstraction ({class}`~ddma.SymbolicAbstraction`) stores the
cells that box meets. A pair whose box leaves the grid is disabled: ddma cannot say where it
goes.

This finite system **over-approximates** the real one: every real trajectory is matched by a
trajectory of the abstraction. Whatever a controller certifies on the abstraction therefore
holds for the real system.

## Step 3: a certified controller

- {func}`~ddma.safety` computes the **maximal controlled invariant set** inside a safe set:
  the cells from which some input keeps every successor inside the set, forever. It returns a
  {class}`~ddma.Controller` with the allowed input cells of every state cell.
- {func}`~ddma.reach_avoid` computes the cells that can be driven into a target while avoiding
  a set, with the number of steps.

A controller can drive the system itself (`ctrl(x)`) or act as a **safety filter** around any
other controller (`ctrl.filter(x, u_wanted)`): it returns the wanted input when it is safe and
the closest safe input otherwise. From any state in `ctrl.domain`, the real system then stays
safe forever, whatever the other controller does. {doc}`../tutorials/05-rl-safety-filter` puts
a learning agent inside such a filter.

## Without a grid: safe envelopes

Grids grow exponentially with the dimension. {mod}`ddma.invariance` computes a controlled
invariant set **directly from the data points**, without a grid and without a model:
corners of lower sets for monotone systems ({func}`~ddma.monotone_envelope`), boxes spanned by
pairs of data points for systems with bounded Jacobians ({func}`~ddma.embedding_envelope`).
See {doc}`../tutorials/06-gridless-envelopes`.

## Notation and code

| on paper | in Python | in MATLAB |
|---|---|---|
| data $(z_k, y_k)$ | `Z` (K, n+p), `Y` (K, m) | `Z`, `Y` |
| disturbance $[d_{lo}, d_{hi}]$ | `d_lo`, `d_hi` | `d_lo`, `d_hi` |
| monotonicity pattern | `signs` ($+1$, $-1$, $0$ per output and input) | `signs` |
| grid | {class}`~ddma.RectangularPartition` | cell array of edge vectors (`ddma.uniform_partition`) |
| model on the grid, $\sigma$ | {class}`~ddma.PartitionedSimulatingMap` | struct from `ddma.fit` |
| abstraction | {class}`~ddma.SymbolicAbstraction` | struct from `ddma.abstraction` |
| controller | {class}`~ddma.Controller` (`domain`, `allowed`) | `[domain, allowed] = ddma.safety(...)` |

## Where the methods come from

- A. Makdesi, A. Girard, L. Fribourg, *Data-Driven Models of Monotone Systems*, IEEE TAC 69,
  2024: minimal simulating maps, grids, bounded derivatives (Sec. III-C), derivative bounds
  from data.
- L4DC 2021 and ADHS 2021 (same authors): data-driven abstractions of monotone systems,
  assume-guarantee contracts.
- A. Makdesi, *Data-driven abstractions for the safe learning of nonlinear systems*, PhD
  thesis, Université Paris-Saclay, 2023, Chapter 3: the five bounded-derivative models.
- A. Makdesi, M. Zamani (and S. Jafarpour), IFAC World Congress 2026: the gridless envelopes.
