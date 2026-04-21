"""WoundMacrophage-v0: Gymnasium environment for macrophage polarization control.

Based on Zlobina et al. (2022) 5-variable ODE model. The agent controls
a polarization treatment signal to drive M1-to-M2 macrophage transition,
promoting wound healing through tissue regeneration.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from woundsim.models.zlobina import ZlobinaModel, ZlobinaParams


class WoundMacrophageEnv(gym.Env):
    """Macrophage polarization wound healing environment.

    Observation:
        5-dimensional normalized state [a, m1, m2, c, n].
    Action:
        1-dimensional continuous polarization treatment signal u in [0, 1].
    Reward:
        Penalizes debris and treatment cost; rewards tissue regeneration.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Normalization bounds for observation space
    _obs_high = np.array([1.0, 1e6, 1e6, 1.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        difficulty: str = "medium",
        dt: float = 6.0,
        max_steps: int = 100,
        render_mode: str | None = None,
        noise_scale: float = 0.0,
        w_a: float = 1.0,
        w_time: float = 0.01,
        w_n: float = 2.0,
        w_u: float = 0.1,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.dt = dt  # hours per step
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.noise_scale = noise_scale if difficulty == "hard" else 0.0

        # Reward weights
        self.w_a = w_a
        self.w_time = w_time
        self.w_n = w_n
        self.w_u = w_u

        self.model = ZlobinaModel(ZlobinaParams())

        # Observation space: normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        # Action space: polarization treatment signal
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._state: np.ndarray | None = None
        self._step_count = 0
        self._prev_n = 0.0

    def _normalize_obs(self, state: np.ndarray) -> np.ndarray:
        """Normalize raw state to [0, 1] observation."""
        return (state / self._obs_high).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._normalize_obs(self._state)

    def _get_info(self) -> dict[str, Any]:
        return {
            "raw_state": self._state.copy(),
            "step": self._step_count,
            "healed": bool(self._state[4] > 0.95),
            "debris": float(self._state[0]),
            "new_tissue": float(self._state[4]),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.model.get_default_initial_state(self.difficulty).copy()

        # Add noise for hard difficulty
        if self.noise_scale > 0 and self.np_random is not None:
            noise = self.np_random.normal(0, self.noise_scale, size=self._state.shape)
            self._state += noise
            self._state = np.clip(
                self._state,
                self.model.STATE_BOUNDS_LOW,
                self.model.STATE_BOUNDS_HIGH,
            )

        self._step_count = 0
        self._prev_n = self._state[4]
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"

        u = float(np.clip(action[0], 0.0, 1.0))
        self._state = self.model.step(self._state, u=u, dt=self.dt)
        self._step_count += 1

        a, m1, m2, c, n = self._state
        dn = n - self._prev_n
        self._prev_n = n

        # Reward: penalize debris, time, treatment cost; reward tissue growth
        reward = (
            -self.w_a * a
            - self.w_time * self.dt
            + self.w_n * dn
            - self.w_u * u * u
        )

        # Termination: healed or max steps
        terminated = bool(n > 0.95)
        truncated = bool(self._step_count >= self.max_steps)

        if terminated:
            reward += 10.0  # bonus for successful healing

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
