---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 5. A learning agent inside a certified safety filter

**What you'll learn:** why reinforcement learning (RL) and safety constraints clash, how a
ddma controller turns into a *safety filter* (often called a *shield*) around any agent, what
exactly it guarantees, and what it doesn't. You need no RL background: the agent here is a
dozen lines of NumPy.

**Main calls:** {class}`ddma.rl.SafetyFilterWrapper`, {class}`ddma.rl.CruiseControlEnv`,
{meth}`ddma.Controller.filter`.

## The problem: learning by trial and error

An RL agent learns a policy (a map from observations to actions) by trying actions and
keeping what earns reward. That is its strength, since it needs no model, and its problem:
it must make mistakes to learn what not to do. In simulation a crash only costs a reset. On
a real car, robot or plant, a single crash during training can be unacceptable.

The fix studied here is to put a **filter** between the agent and the system:

```
agent --(wanted action)--> safety filter --(safe action)--> system
  ^                                                            |
  +--------------------- observation, reward ------------------+
```

The filter is the controller from tutorial 2: at each step it lets the agent's action through
if it is certified safe in the current state, and replaces it by the closest certified action
otherwise. The agent learns as usual. It never knows the filter is there, except that some of
its actions are softened.

## The task

{class}`~ddma.rl.CruiseControlEnv` is tutorial 2's problem as a
[Gymnasium](https://gymnasium.farama.org) environment. The reward per step is
$1 - ((v - 22)/10)^2$: drive close to 22 m/s, which is faster than the leader can go, so the
reward pushes the agent towards the leader. Leaving $h \in [10, 50]$, $v \in [2, 28]$ (a
crash or a lost leader) ends the episode with $-50$. The agent observes
$(h, v, \Delta h)$ but not the leader's speed.

```{code-cell} ipython3
import time

import matplotlib.pyplot as plt
import numpy as np

from ddma import (PartitionedSimulatingMap, RectangularPartition, SymbolicAbstraction,
                  affine_bounds, safety, transitions_to_data)
from ddma.rl import CruiseControlEnv, SafetyFilterWrapper

Env = CruiseControlEnv
env = Env()
print("actions:", env.action_space, "\nobservations:", env.observation_space)
```

A random agent shows the problem immediately:

```{code-cell} ipython3
def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, violations, done = 0.0, 0, False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total += r
        violations += info["violation"]
        done = terminated or truncated
    return total, violations

rand = np.random.default_rng(0)
random_policy = lambda obs: rand.uniform(-3, 3, 1).astype(np.float32)
res = [run_episode(Env(), random_policy, seed) for seed in range(20)]
print(f"random agent, 20 episodes: {sum(v for _, v in res)} episodes ended in a violation")
```

## The safety controller, from logged data

Exactly as in tutorial 2: 200,000 logged transitions of the black-box car (the environment's
dynamics, which ddma never sees as equations), a monotone model of the speed update, an
abstraction, and the maximal safe set.

```{code-cell} ipython3
T = Env.T
rng = np.random.default_rng(0)
K = 200_000
X = rng.uniform([5, 0], [60, 30], (K, 2))
U = rng.uniform(-3, 3, (K, 1))
Xn = np.column_stack([X[:, 0] - T * X[:, 1] + T * rng.uniform(*Env.VL, K),          # logged
                      Env.speed_update(X[:, 1], U[:, 0]) + rng.uniform(-Env.W, Env.W, K)])

A_G = np.array([[1.0, -T], [0.0, 0.0]])
Z, Y = transitions_to_data(X, U, Xn, lambda X, U: X @ A_G.T)
Xgrid = RectangularPartition.uniform([5, 0], [60, 30], [110, 60])
Ugrid = RectangularPartition.uniform([-3], [3], 24)
feats = [1, 2]
sigma = PartitionedSimulatingMap.from_data(
    Z[:, feats], Y, RectangularPartition.product(Xgrid, Ugrid).project(feats),
    [T * Env.VL[0], -Env.W], [T * Env.VL[1], Env.W])
abst = SymbolicAbstraction(Xgrid, Ugrid, sigma, g_bounds=affine_bounds(A_G), features=feats)
safe = abst.cells_inside(Env.SAFE_LO, Env.SAFE_HI)
ctrl = safety(abst, safe)
print(f"certified safe region: {100 * ctrl.domain.sum() / safe.sum():.1f} % of the specification")
```

## The wrapper

{class}`~ddma.rl.SafetyFilterWrapper` is a standard Gymnasium wrapper, so any agent from any
library can be trained on it. Three details matter:

- **It needs the true state**, not the observation. By default it reads
  `env.unwrapped.state`; for other environments pass `state_fn`. The guarantee is about the
  state the abstraction was built on.
- **It starts episodes inside the certified set** (`reset_in_domain=True`): it redraws the
  initial state until it lies in `ctrl.domain`. From a state outside the set no controller can
  promise anything, so the guarantee starts there.
- **It reports what it did**: `info["filtered"]`, `info["correction"]` (how far the action
  was moved) and `info["safe_action"]`, plus counters `n_steps`, `n_filtered`.

```{code-cell} ipython3
safe_env = SafetyFilterWrapper(Env(), ctrl)
res = [run_episode(safe_env, random_policy, seed) for seed in range(20)]
print(f"random agent inside the filter, 20 episodes: {sum(v for _, v in res)} violations; "
      f"{100 * safe_env.n_filtered / safe_env.n_steps:.0f} % of its actions were changed")
```

## A learning agent

To see the filter during *learning*, we need an agent that learns. A deep-RL library (PPO,
SAC) would work (see the end of this tutorial), but to keep this notebook fast and
dependency-free we use the simplest learner there is: the **cross-entropy method**.

- The policy is linear in a few features: $u = \mathrm{clip}(\theta^\top \phi(o), -3, 3)$ with
  $\phi(o) = (1,\ (h - 30)/20,\ (v - 15)/10,\ \Delta h)$.
- Each round, draw 16 parameter vectors $\theta$ from a Gaussian, run one episode with each,
  keep the best 4, and refit the Gaussian to them. Repeat.

Every episode is a real trial, crashes included, so we count violations over the whole
training, once without the filter and once with it.

```{code-cell} ipython3
def phi(obs):
    return np.array([1.0, (obs[0] - 30) / 20, (obs[1] - 15) / 10, obs[2]])

def linear_policy(theta):
    return lambda obs: np.clip([phi(obs) @ theta], -3, 3).astype(np.float32)

def train(env, rounds=15, pop=16, elite=4, seed=1):
    r = np.random.default_rng(seed)
    mu, sd = np.zeros(4), np.full(4, 1.5)
    curve, violations, episode = [], 0, 0
    for _ in range(rounds):
        thetas = mu + sd * r.standard_normal((pop, 4))
        returns = []
        for th in thetas:
            ret, v = run_episode(env, linear_policy(th), seed=episode)
            episode += 1
            returns.append(ret)
            violations += v
        best = thetas[np.argsort(returns)[-elite:]]
        mu, sd = best.mean(0), best.std(0) + 0.05
        curve.append((np.mean(returns), violations))
    return mu, np.array(curve)

t0 = time.perf_counter()
theta_raw, curve_raw = train(Env())
theta_safe, curve_safe = train(SafetyFilterWrapper(Env(), ctrl))
print(f"trained twice in {time.perf_counter() - t0:.1f} s ({15 * 16} episodes each)")
print(f"violations during training: without filter {curve_raw[-1, 1]:.0f}, "
      f"with filter {curve_safe[-1, 1]:.0f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(curve_raw[:, 1], color="#e76f51", label="without filter")
ax[0].plot(curve_safe[:, 1], color="#264653", label="with filter")
ax[0].set(xlabel="training round", ylabel="violations so far", title="Crashes while learning")
ax[1].plot(curve_raw[:, 0], color="#e76f51", label="without filter")
ax[1].plot(curve_safe[:, 0], color="#264653", label="with filter")
ax[1].set(xlabel="training round", ylabel="mean return", title="Learning progress")
for a in ax:
    a.legend(frameon=False, fontsize=8)
fig.tight_layout();
```

Without the filter the agent crashes or loses the leader throughout training: crashing is
how it learns where the edges are, and the reward keeps pulling it towards them. With the
filter it never does. It also learns faster: its very first round already scores well,
because the filter turns the worst random policies into safe ones and no episode is cut
short by a crash.

## What the guarantee covers, and what it doesn't

**Covered:** every step of every episode, during training and deployment, for any agent and
any action it proposes: the state stays in the certified set, so the specification holds.
This rests on the assumptions of tutorial 2 (monotone speed dynamics, the stated disturbance
and leader bounds) and on starting inside the set.

**Not covered: the agent on its own.** The agent learned with the filter in place. Take the
filter away and its policy may be unsafe, because it may have learned to rely on the filter's
corrections:

```{code-cell} ipython3
for name, theta in (("trained without filter", theta_raw), ("trained with filter", theta_safe)):
    pol = linear_policy(theta)
    alone = [run_episode(Env(), pol, 1000 + s) for s in range(20)]
    wrapped = SafetyFilterWrapper(Env(), ctrl)
    inside = [run_episode(wrapped, pol, 1000 + s) for s in range(20)]
    print(f"{name:23s}: alone {sum(v for _, v in alone):2d} violations, "
          f"mean return {np.mean([r for r, _ in alone]):6.1f} | in filter 0 violations: "
          f"{sum(v for _, v in inside) == 0}, return {np.mean([r for r, _ in inside]):6.1f}, "
          f"actions changed {100 * wrapped.n_filtered / wrapped.n_steps:4.1f} %")
```

The two agents learned different things. The one trained without the filter paid for its
crashes and ended up cautious: it is safe on its own and the filter barely touches it. The
one trained with the filter never paid for anything, so it learned the simplest
high-reward policy, roughly "full throttle", and lets the filter do the braking. Inside the
filter it earns the best return; alone it crashes in every episode.

Neither is wrong, but the lesson is clear: **the guarantee belongs to the agent and the
filter together.** Deploy the filter with the agent (a call costs well under a millisecond).
If you want an agent that needs the filter rarely, `SafetyFilterWrapper(env, ctrl, penalty=c)` subtracts
`c × correction` from the reward, which teaches the agent to propose safe actions itself:

```{code-cell} ipython3
theta_pen, _ = train(SafetyFilterWrapper(Env(), ctrl, penalty=1.0))
wrapped = SafetyFilterWrapper(Env(), ctrl)
_ = [run_episode(wrapped, linear_policy(theta_pen), 1000 + s) for s in range(20)]
print(f"trained with penalty: actions changed by the filter {100 * wrapped.n_filtered / wrapped.n_steps:.1f} %")
```

## With a deep-RL agent

`examples/rl_safety_filter.py` runs the same comparison with PPO (Stable-Baselines3, 100,000
steps, 3 seeds). It is too slow to run while this site builds, so here is its stored result:
without the filter PPO violates the constraints 266–319 times while it learns; inside the
filter, 0 times, in training and in evaluation, and it ends with a slightly higher return.

```{image} ../figures/rl_safety_filter.png
:alt: Constraint violations during PPO training, with and without the safety filter
:width: 640px
```

The code is the same as above. Only the agent changes:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", SafetyFilterWrapper(CruiseControlEnv(), ctrl)).learn(100_000)
```

## Summary

- A ddma safety controller becomes a **filter** around any agent: certified actions pass,
  others are replaced by the closest certified one.
- From a state in the certified set, **no action sequence can violate the specification**,
  during training as well as deployment.
- The filter needs the true state and must start in its domain; the guarantee is about the
  system with the filter, not the agent alone (use `penalty=` to make the agent need it
  less).

Next: {doc}`06-gridless-envelopes` computes safe sets without a grid.

## In MATLAB

`ddma.safe_step` and `ddma.safe_reset` wrap the step and reset functions of an environment
in the form used by the Reinforcement Learning Toolbox (`rlFunctionEnv`).
`matlab/examples/rl_safety_filter.m` runs a full example without the toolbox, with a random
and a full-throttle agent.

::::{tab-set}
:::{tab-item} MATLAB
```matlab
addpath examples                                     % cruise_control_env
[step0, reset0, P] = cruise_control_env(400);        % rlFunctionEnv-style step / reset
% the safety controller, as in tutorial 2
ddma.seed(0); K = 200000;
X = [5 + 55 * rand(K, 1), 30 * rand(K, 1)]; U = 6 * rand(K, 1) - 3;
Xn = [X(:, 1) - P.T * X(:, 2) + P.T * (13 + 4 * rand(K, 1)), ...
      X(:, 2) + P.T * (U - (P.f0 + P.f1 * X(:, 2) + P.f2 * X(:, 2).^2) / P.M) + P.W * (2 * rand(K, 1) - 1)];
A_G = [1 -P.T; 0 0];
Xedges = ddma.uniform_partition([5 0], [60 30], [110 60]);
Uedges = ddma.uniform_partition(-3, 3, 24);
S = ddma.fit([X(:, 2) U], Xn - X * A_G', {Xedges{2}, Uedges{1}}, [P.T * 13, -P.W], [P.T * 17, P.W]);
A = ddma.abstraction(Xedges, Uedges, S, ddma.affine_bounds(A_G), [2 3]);
[domain, allowed] = ddma.safety(A, ddma.cells_inside(Xedges, P.SAFE_LO, P.SAFE_HI));
safeStep  = ddma.safe_step(step0, A, allowed);       % filter every action
safeReset = ddma.safe_reset(reset0, A, allowed);     % start inside the certified set
[obs, L] = safeReset();
[obs, reward, done, L] = safeStep(3, L);             % L.Filtered, L.SafeAction, L.Correction
% with the Reinforcement Learning Toolbox:
%   env = rlFunctionEnv(obsInfo, actInfo, safeStep, safeReset);
```
:::
:::{tab-item} Python
```python
env = SafetyFilterWrapper(CruiseControlEnv(), ctrl, penalty=0.0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(np.array([3.0], np.float32))
```
:::
::::
