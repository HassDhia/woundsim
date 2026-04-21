"""WoundDiabetic-v0: Gymnasium environment for diabetic wound treatment.

Extended inflammation model incorporating glucose-insulin dynamics.
The agent must manage wound treatment, growth factors, AND glycemic
control simultaneously.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from woundsim.models.inflammation import InflammationModel, InflammationParams


class WoundDiabeticEnv(gym.Env):
    """Diabetic wound healing environment.

    Observation:
        7-dimensional normalized state [w, a, m1, m2, G, I, E].
    Action:
        3-dimensional continuous:
        - action[0]: polarization treatment signal [0, 1]
        - action[1]: topical growth factor dose [0, 1]
        - action[2]: insulin dose adjustment [0, 1]
    Reward:
        Penalizes wound area, glucose deviation, treatment cost;
        rewards healing rate.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    _obs_high = np.array(
        [1.0, 1.0, 1e6, 1e6, 400.0, 200.0, 1.0], dtype=np.float32
    )
    _obs_low = np.array(
        [0.0, 0.0, 0.0, 0.0, 70.0, 0.0, 0.0], dtype=np.float32
    )

    def __init__(
        self,
        difficulty: str = "moderate",
        dt: float = 8.0,
        max_steps: int = 150,
        render_mode: str | None = None,
        w_wound: float = 1.0,
        w_glucose: float = 0.5,
        w_treat: float = 0.1,
        w_heal: float = 2.0,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.w_wound = w_wound
        self.w_glucose = w_glucose
        self.w_treat = w_treat
        self.w_heal = w_heal

        self.model = InflammationModel(InflammationParams())

        # Normalize observation to [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

        self._state: np.ndarray | None = None
        self._step_count = 0
        self._prev_w = 0.0

    def _normalize_obs(self, state: np.ndarray) -> np.ndarray:
        obs_range = self._obs_high - self._obs_low
        obs_range[obs_range == 0] = 1.0
        normalized = (state - self._obs_low) / obs_range
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._normalize_obs(self._state)

    def _get_info(self) -> dict[str, Any]:
        return {
            "raw_state": self._state.copy(),
            "step": self._step_count,
            "healed": bool(self._state[0] < 0.05),
            "wound_area": float(self._state[0]),
            "debris": float(self._state[1]),
            "glucose": float(self._state[4]),
            "insulin": float(self._state[5]),
            "ecm": float(self._state[6]),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.model = InflammationModel(InflammationParams())
        self._state = self.model.get_default_initial_state(self.difficulty).copy()
        self._step_count = 0
        self._prev_w = self._state[0]
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"

        action = np.clip(action, 0.0, 1.0)
        self._state = self.model.step(self._state, action=action, dt=self.dt)
        self._step_count += 1

        w, a, m1, m2, G, I, E = self._state
        healing_rate = self._prev_w - w
        self._prev_w = w

        # Glucose penalty: deviation from target (100 mg/dL)
        glucose_penalty = abs(G - 100.0) / 300.0

        treatment_cost = float(np.sum(action**2))

        reward = (
            -self.w_wound * w
            - self.w_glucose * glucose_penalty
            - self.w_treat * treatment_cost
            + self.w_heal * healing_rate
        )

        terminated = bool(w < 0.05)
        truncated = bool(self._step_count >= self.max_steps)

        if terminated:
            reward += 10.0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
