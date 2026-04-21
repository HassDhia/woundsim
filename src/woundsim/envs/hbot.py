"""WoundHBOT-v0: Gymnasium environment for hyperbaric oxygen therapy.

Based on Flegg et al. (2009, 2015) HBOT angiogenesis model. The agent
controls HBOT session intensity and duration to promote wound healing
through vascularization.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from woundsim.models.flegg import FleggModel, FleggParams


class WoundHBOTEnv(gym.Env):
    """HBOT wound healing environment.

    Observation:
        4-dimensional normalized state [b, n_cap, O, w].
    Action:
        2-dimensional continuous:
        - action[0]: HBOT session intensity (maps to 1-3 atm) [0, 1]
        - action[1]: session duration fraction (maps to 0-120 min) [0, 1]
    Reward:
        Rewards healing rate; penalizes treatment cost and hyperoxygenation.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    _obs_high = np.array([1.0, 1.0, 300.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        difficulty: str = "chronic",
        dt: float = 4.0,
        max_steps: int = 240,
        render_mode: str | None = None,
        w_heal: float = 5.0,
        w_cost: float = 0.2,
        w_hyperox: float = 0.5,
        O_hyperox_thresh: float = 250.0,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.w_heal = w_heal
        self.w_cost = w_cost
        self.w_hyperox = w_hyperox
        self.O_hyperox_thresh = O_hyperox_thresh

        self.model = FleggModel(FleggParams())

        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        self._state: np.ndarray | None = None
        self._step_count = 0
        self._prev_w = 0.0

    def _normalize_obs(self, state: np.ndarray) -> np.ndarray:
        return (state / self._obs_high).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._normalize_obs(self._state)

    def _get_info(self) -> dict[str, Any]:
        return {
            "raw_state": self._state.copy(),
            "step": self._step_count,
            "healed": bool(self._state[3] < 0.05),
            "wound_area": float(self._state[3]),
            "oxygen": float(self._state[2]),
            "capillary_tips": float(self._state[0]),
            "capillary_sprouts": float(self._state[1]),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.model = FleggModel(FleggParams())
        self._state = self.model.get_default_initial_state(self.difficulty).copy()
        self._step_count = 0
        self._prev_w = self._state[3]
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"

        action = np.clip(action, 0.0, 1.0)
        self._state = self.model.step(self._state, action=action, dt=self.dt)
        self._step_count += 1

        b, n_cap, O, w = self._state
        healing_rate = self._prev_w - w  # positive when wound shrinks
        self._prev_w = w

        treatment_cost = float(np.sum(action**2))
        hyperox_penalty = max(0.0, O - self.O_hyperox_thresh) / 100.0

        reward = (
            self.w_heal * healing_rate * 10.0
            - self.w_cost * treatment_cost
            - self.w_hyperox * hyperox_penalty
            - 0.01  # small time penalty per step
        )

        terminated = bool(w < 0.05)
        truncated = bool(self._step_count >= self.max_steps)

        if terminated:
            reward += 2.0  # moderate healing bonus

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
