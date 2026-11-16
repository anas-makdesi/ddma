"""Certified safety filters for reinforcement learning (Gymnasium).

:class:`SafetyFilterWrapper` puts a ddma controller (from :func:`ddma.safety`) around any
Gymnasium environment: every action of the agent goes through
:meth:`ddma.Controller.filter`, which returns it unchanged when it is safe and the nearest
safe action otherwise. As long as the state starts in the controller's domain, the system
stays in the safe set for ever, whatever the agent does, during training as well as
deployment.

:class:`CruiseControlEnv` is a small benchmark: the adaptive cruise control problem of
the TAC paper as an RL task (drive close to 22 m/s behind a leader whose speed changes).
An agent trained without the filter crashes or drops out of the speed range while it
learns; with the filter it never leaves the safe set.

Needs ``gymnasium`` (``pip install ddma[rl]``).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("ddma.rl needs gymnasium: pip install ddma[rl]") from exc

from .synthesis import Controller

__all__ = ["SafetyFilterWrapper", "CruiseControlEnv"]


class SafetyFilterWrapper(gym.Wrapper):
    """Certified safety filter around a Gymnasium environment.

    Parameters
    ----------
    env : the environment; its actions are the inputs u of the abstraction.
    controller : :class:`ddma.Controller` from :func:`ddma.safety` (maximal controlled
        invariant set of the safe set).
    state_fn : function env -> state x of the abstraction (default:
        ``env.unwrapped.state``); the filter needs the true state, not an observation.
    penalty : reward penalty per unit of correction ``||u_safe - u||`` (default 0), to
        teach the agent to propose safe actions itself.
    outside : what to do when the state is outside the controller domain (only possible
        if the episode starts there): ``"pass"`` the action unchanged (default) or
        ``"raise"``.
    reset_in_domain : redraw the initial state (up to 1000 resets) until it lies in the
        controller domain (default True), so that the guarantee holds from the start.

    ``info`` gains ``safe_action``, ``filtered`` (bool), ``correction`` (norm) and
    ``outside_domain``; :attr:`n_steps`, :attr:`n_filtered` count over the wrapper's life.
    """

    def __init__(self, env, controller: Controller,
                 state_fn: Optional[Callable] = None, penalty: float = 0.0,
                 outside: str = "pass", reset_in_domain: bool = True):
        super().__init__(env)
        if outside not in ("pass", "raise"):
            raise ValueError("outside must be 'pass' or 'raise'")
        self.controller = controller
        self.state_fn = state_fn or (lambda e: np.asarray(e.unwrapped.state, float))
        self.penalty = float(penalty)
        self.outside = outside
        self.reset_in_domain = bool(reset_in_domain)
        self.n_steps = 0
        self.n_filtered = 0

    def in_domain(self) -> bool:
        """Whether the current state lies in the controller domain."""
        return self.controller.input_cells(self.state_fn(self.env)).size > 0

    def reset(self, **kwargs):
        """Reset the environment; with ``reset_in_domain``, redraw until the state lies in
        the controller domain."""
        obs, info = self.env.reset(**kwargs)
        if self.reset_in_domain:
            kwargs.pop("seed", None)  # later draws continue the seeded generator
            for _ in range(1000):
                if self.in_domain():
                    break
                obs, info = self.env.reset(**kwargs)
            else:
                raise RuntimeError("no initial state in the controller domain after 1000 resets")
        return obs, info

    def safe_action(self, action):
        """(safe action, inside the domain?) for the current state."""
        u = np.asarray(action, dtype=float).ravel()
        x = self.state_fn(self.env)
        s = self.controller.filter(x, u)
        if s is None:
            if self.outside == "raise":
                raise RuntimeError(f"state {x} is outside the safety controller's domain")
            return u, False
        return np.asarray(s, dtype=float), True

    def step(self, action):
        """Filter the action (closest certified action), then step the environment."""
        u = np.asarray(action, dtype=float).ravel()
        s, inside = self.safe_action(u)
        corr = float(np.linalg.norm(s - u))
        filtered = corr > 0.0
        obs, reward, terminated, truncated, info = self.env.step(
            s.astype(np.asarray(action).dtype if hasattr(action, "dtype") else float))
        self.n_steps += 1
        self.n_filtered += int(filtered)
        info = dict(info, safe_action=s, filtered=filtered, correction=corr,
                    outside_domain=not inside)
        if self.penalty:
            reward = reward - self.penalty * corr
        return obs, reward, terminated, truncated, info


class CruiseControlEnv(gym.Env):
    """Adaptive cruise control as an RL task (TAC paper, Section V-B).

    State x = (h, v): gap to the leader [m], own speed [m/s]; action u in [-3, 3] m/s^2.

        h+ = h - T v + T v_l,
        v+ = v + T (u - (f0 + f1 v + f2 v^2) / M) + w,   |w| <= 0.05,

    with T = 0.5 s and a leader speed v_l in [13, 17] m/s that switches between fast and
    slow phases. Reward per step: 1 - ((v - 22) / 10)^2 (drive close to 22 m/s, faster
    than the leader can go, and keep driving). The episode ends with reward -50 when the
    state leaves the safe set h in [10, 50], v in [2, 28] (a crash or a lost leader),
    otherwise after ``horizon`` steps. Observation (h, v, h - h_prev): the leader's speed
    is not observed.
    ``info["violation"]`` is True on a violating step.
    """

    metadata = {"render_modes": []}
    T, M, F0, F1, F2 = 0.5, 1370.0, 51.0, 1.2567, 0.4342
    VL = (13.0, 17.0)
    W = 0.05
    SAFE_LO = np.array([10.0, 2.0])
    SAFE_HI = np.array([50.0, 28.0])
    V_REF = 22.0

    def __init__(self, horizon: int = 400, init_lo=(20.0, 12.0), init_hi=(40.0, 18.0)):
        self.horizon = int(horizon)
        self.init_lo, self.init_hi = np.array(init_lo, float), np.array(init_hi, float)
        self.action_space = spaces.Box(-3.0, 3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([0.0, 0.0, -10.0], np.float32), np.array([80.0, 40.0, 10.0], np.float32))
        self.state = np.zeros(2)
        self._k = 0
        self._dh = 0.0
        self._phase_len = 40
        self._vl = 15.0

    @classmethod
    def speed_update(cls, v, u):
        """The follower's speed update without disturbance (the dynamics the agent and
        ddma do not know)."""
        return v + cls.T * (u - (cls.F0 + cls.F1 * v + cls.F2 * v**2) / cls.M)

    def _obs(self):
        return np.array([self.state[0], self.state[1], self._dh], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Random initial state in [init_lo, init_hi] and a random leader phase."""
        super().reset(seed=seed)
        rng = self.np_random
        self.state = rng.uniform(self.init_lo, self.init_hi)
        self._k = 0
        self._dh = 0.0
        self._phase_len = int(rng.integers(20, 60))
        self._fast = bool(rng.integers(2))
        return self._obs(), {}

    def _leader_speed(self):
        if self._k % self._phase_len == 0 and self._k > 0:
            self._fast = not self._fast
        base = self.VL[1] if self._fast else self.VL[0]
        return float(np.clip(base + self.np_random.uniform(-0.5, 0.5), *self.VL))

    def step(self, action):
        """One step of 0.5 s; terminates with reward -50 when the state leaves the safe set."""
        u = float(np.clip(np.asarray(action, float).ravel()[0], -3.0, 3.0))
        h, v = self.state
        vl = self._leader_speed()
        h_new = h - self.T * v + self.T * vl
        v_new = self.speed_update(v, u) + self.np_random.uniform(-self.W, self.W)
        self.state = np.array([h_new, v_new])
        self._dh = h_new - h
        self._k += 1
        violation = bool(np.any(self.state < self.SAFE_LO) or np.any(self.state > self.SAFE_HI))
        reward = -50.0 if violation else 1.0 - ((v_new - self.V_REF) / 10.0) ** 2
        truncated = self._k >= self.horizon and not violation
        return self._obs(), reward, violation, truncated, {"violation": violation}
