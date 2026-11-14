"""A reinforcement-learning agent that never violates its constraints, even while learning.

Adaptive cruise control (ddma.rl.CruiseControlEnv): drive close to 22 m/s behind a
leader whose speed switches between about 13 and 17 m/s, keeping the gap in [10, 50] m
and the speed in [2, 28] m/s. A crash or a lost leader ends the episode with -50.

1. A safety controller is learned from 200,000 logged transitions of the black-box car
   (examples/adaptive_cruise_control.py: monotone model, abstraction, maximal
   controlled invariant set), without the vehicle model.
2. PPO (Stable-Baselines3) is trained on the environment, once as is and once inside
   ddma.rl.SafetyFilterWrapper, for several seeds. Violations are counted during
   training and in evaluation.

Run:  python examples/rl_safety_filter.py [--steps 100000] [--seeds 3]
      (needs gymnasium, stable-baselines3; writes docs/figures/rl_safety_filter.png)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_cruise_control import learn_and_synthesize  # noqa: E402

from ddma.rl import CruiseControlEnv, SafetyFilterWrapper  # noqa: E402


class Recorder(gym.Wrapper):
    """Counts violations of the inner environment, with the step at which they occur."""

    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.violation_steps = []

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.steps += 1
        if info.get("violation", False):
            self.violation_steps.append(self.steps)
        return obs, r, term, trunc, info


def evaluate(model, ctrl, filtered, episodes=20):
    env = CruiseControlEnv()
    if filtered:
        env = SafetyFilterWrapper(env, ctrl)
    viol, speeds, returns = 0, [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, ret = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ret += r
            viol += info["violation"]
            speeds.append(float(obs[1]))
        returns.append(ret)
    return dict(violations=int(viol), mean_speed=float(np.mean(speeds)),
                mean_return=float(np.mean(returns)))


def main(steps=100_000, seeds=3):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(2)
    t0 = time.perf_counter()
    ctrl, _ = learn_and_synthesize(200_000)
    print(f"safety controller learned from 200,000 transitions in {time.perf_counter() - t0:.1f} s")
    results = {"steps": steps, "runs": []}
    for filtered in (False, True):
        for seed in range(seeds):
            rec = Recorder(CruiseControlEnv())
            env = SafetyFilterWrapper(rec, ctrl) if filtered else rec
            t0 = time.perf_counter()
            model = PPO("MlpPolicy", env, seed=seed, verbose=0)
            model.learn(steps)
            ev = evaluate(model, ctrl, filtered)
            run = dict(filtered=filtered, seed=seed, train_violations=len(rec.violation_steps),
                       violation_steps=rec.violation_steps, time=time.perf_counter() - t0,
                       filtered_actions=getattr(env, "n_filtered", 0), **ev)
            results["runs"].append(run)
            print(f"{'with filter' if filtered else 'no filter  '} seed {seed}: "
                  f"{run['train_violations']:4d} violations in training, "
                  f"{ev['violations']} in evaluation, mean speed {ev['mean_speed']:.1f} m/s, "
                  f"return {ev['mean_return']:.0f}  ({run['time']:.0f} s)")
    return results


def plot(results, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    steps = results["steps"]
    for run in results["runs"]:
        c = "#2a9d8f" if run["filtered"] else "#e76f51"
        v = np.array(run["violation_steps"], int)
        xs = np.concatenate([[0], v, [steps]])
        ys = np.concatenate([[0], np.arange(1, v.size + 1), [v.size]])
        ax.step(xs, ys, where="post", color=c, lw=1.4, alpha=0.85)
    nf = [r["train_violations"] for r in results["runs"] if not r["filtered"]]
    ax.plot([], [], color="#e76f51", label=f"PPO alone ({min(nf)}–{max(nf)} violations)")
    ax.plot([], [], color="#2a9d8f", label="PPO inside the ddma safety filter (0 violations)")
    ax.set(xlabel="training steps", ylabel="constraint violations so far", xlim=(0, steps))
    ax.set_title("Learning to drive: crashes and lost leaders during training", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    res = main(a.steps, a.seeds)
    root = Path(__file__).resolve().parents[1]
    (root / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    with open(root / "docs" / "rl_safety_filter.json", "w") as f:
        json.dump({k: v for k, v in res.items()}, f, indent=1)
    plot(res, root / "docs" / "figures" / "rl_safety_filter.png")
