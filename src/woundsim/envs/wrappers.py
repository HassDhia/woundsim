"""SB3-compatible wrappers for WoundSim environments."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeReward(gym.Wrapper):
    """Running reward normalization wrapper for stable RL training.

    Tracks a running mean and standard deviation of rewards and normalizes
    them to zero-mean, unit-variance. Compatible with SB3.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms_mean = 0.0
        self.return_rms_var = 1.0
        self.return_rms_count = 0
        self._ret = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret = self._ret * self.gamma + reward
        self._update_stats(self._ret)

        std = np.sqrt(self.return_rms_var) + self.epsilon
        normalized_reward = reward / std

        if terminated or truncated:
            self._ret = 0.0

        info["raw_reward"] = reward
        return obs, float(normalized_reward), terminated, truncated, info

    def _update_stats(self, val: float):
        self.return_rms_count += 1
        delta = val - self.return_rms_mean
        self.return_rms_mean += delta / self.return_rms_count
        delta2 = val - self.return_rms_mean
        self.return_rms_var += (delta * delta2 - self.return_rms_var) / self.return_rms_count

    def reset(self, **kwargs):
        self._ret = 0.0
        return self.env.reset(**kwargs)


class ClipAction(gym.ActionWrapper):
    """Clip actions to the environment's action space bounds.

    Useful when policy outputs may slightly exceed [0, 1] bounds.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)


class FlattenAction(gym.ActionWrapper):
    """Flatten multi-dimensional action spaces for compatibility.

    Some algorithms expect 1D action vectors. This ensures the action
    space is flat.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self.action_space = spaces.Box(
            low=env.action_space.low.flatten(),
            high=env.action_space.high.flatten(),
            dtype=env.action_space.dtype,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return action.reshape(self.env.action_space.shape)


class TimeLimit(gym.Wrapper):
    """Explicit time limit wrapper (backup for Gymnasium's built-in)."""

    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self._max_steps = max_steps
        self._step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)
