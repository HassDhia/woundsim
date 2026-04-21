"""WoundIschemic-v0: Gymnasium environment for ischemic wound treatment.

Based on simplified Xue & Friedman (2009) 6-variable ODE model. The agent
controls revascularization treatment and growth factor application to
promote healing in ischemic wounds.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from woundsim.models.xue_friedman import XueFriedmanModel, XueFriedmanParams


class WoundIschemicEnv(gym.Env):
    """Ischemic wound healing environment.

    Observation:
        6-dimensional normalized state [w, O, V, M, F, E].
    Action:
        2-dimensional continuous:
        - action[0]: ischemia treatment (revascularization scaling) [0, 1]
        - action[1]: growth factor application [0, 1]
    Reward:
        Penalizes wound area and time; rewards ECM formation.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    _obs_high = np.array([1.0, 100.0, 100.0, 1e6, 1e6, 1.0], dtype=np.float32)

    def __init__(
        self,
        difficulty: str = "moderate",
        dt: float = 12.0,
        max_steps: int = 200,
        render_mode: str | None = None,
        w_wound: float = 1.0,
        w_time: float = 0.005,
        w_ecm: float = 2.0,
        w_treat: float = 0.1,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.w_wound = w_wound
        self.w_time = w_time
        self.w_ecm = w_ecm
        self.w_treat = w_treat

        self.model = XueFriedmanModel(XueFriedmanParams())

        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        self._state: np.ndarray | None = None
        self._step_count = 0
        self._prev_E = 0.0

    def _normalize_obs(self, state: np.ndarray) -> np.ndarray:
        return (state / self._obs_high).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._normalize_obs(self._state)

    def _get_info(self) -> dict[str, Any]:
        return {
            "raw_state": self._state.copy(),
            "step": self._step_count,
            "healed": bool(self._state[0] < 0.05),
            "wound_area": float(self._state[0]),
            "oxygen": float(self._state[1]),
            "ecm": float(self._state[5]),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.model = XueFriedmanModel(XueFriedmanParams())
        self._state = self.model.get_default_initial_state(self.difficulty).copy()
        self._step_count = 0
        self._prev_E = self._state[5]
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"

        action = np.clip(action, 0.0, 1.0)
        self._state = self.model.step(self._state, action=action, dt=self.dt)
        self._step_count += 1

        w, O, V, M, F, E = self._state
        dE = E - self._prev_E
        self._prev_E = E

        treatment_cost = float(np.sum(action**2))
        reward = (
            -self.w_wound * w
            - self.w_time * self.dt
            + self.w_ecm * dE
            - self.w_treat * treatment_cost
        )

        terminated = bool(w < 0.05)
        truncated = bool(self._step_count >= self.max_steps)

        if terminated:
            reward += 10.0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
