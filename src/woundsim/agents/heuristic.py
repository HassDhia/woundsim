"""Clinical heuristic baseline agents for WoundSim environments.

Each heuristic implements a simplified version of standard clinical
protocols for the corresponding wound type.
"""

import gymnasium as gym
import numpy as np


class HeuristicAgent:
    """Clinical heuristic baseline agent.

    Implements fixed or simple rule-based treatment protocols that
    approximate standard clinical practice for each wound type.
    """

    # Registry of heuristic strategies per environment
    _strategies = {}

    def __init__(self, env: gym.Env, env_id: str | None = None):
        self.env = env
        self.env_id = env_id or ""
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        """Select heuristic strategy based on environment ID."""
        if "Macrophage" in self.env_id:
            return self._macrophage_heuristic
        elif "Ischemic" in self.env_id:
            return self._ischemic_heuristic
        elif "HBOT" in self.env_id:
            return self._hbot_heuristic
        elif "Diabetic" in self.env_id:
            return self._diabetic_heuristic
        else:
            return self._default_heuristic

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Predict action using clinical heuristic.

        Args:
            observation: Current normalized observation.
            deterministic: Ignored (heuristics are deterministic).

        Returns:
            Tuple of (action, None) matching SB3 interface.
        """
        return self._strategy(observation), None

    def _macrophage_heuristic(self, obs: np.ndarray) -> np.ndarray:
        """Constant polarization signal u=0.5 (moderate M1-to-M2 conversion).

        Standard approach: maintain steady polarization to balance
        debris clearance (M1) with tissue repair (M2).
        """
        return np.array([0.5], dtype=np.float32)

    def _ischemic_heuristic(self, obs: np.ndarray) -> np.ndarray:
        """Full revascularization + moderate growth factor.

        Clinical protocol: maximize blood flow restoration,
        apply moderate growth factor supplementation.
        """
        return np.array([1.0, 0.5], dtype=np.float32)

    def _hbot_heuristic(self, obs: np.ndarray) -> np.ndarray:
        """Moderate HBOT protocol: balanced to sustain angiogenesis.

        Moderate intensity keeps oxygen below the O_thresh that
        suppresses capillary tip sprouting, preserving the
        vascularization pathway needed for wound closure.
        """
        return np.array([0.3, 0.4], dtype=np.float32)

    def _diabetic_heuristic(self, obs: np.ndarray) -> np.ndarray:
        """Standard wound care + insulin sliding scale.

        Moderate polarization treatment (0.4), low growth factor (0.3),
        and glucose-responsive insulin dosing.
        """
        # obs[4] is normalized glucose: (G - 70) / 330
        # Estimate glucose from normalized observation
        glucose_norm = obs[4] if len(obs) > 4 else 0.5
        glucose_est = glucose_norm * 330.0 + 70.0

        # Insulin sliding scale
        if glucose_est > 250:
            insulin_dose = 0.8
        elif glucose_est > 180:
            insulin_dose = 0.5
        elif glucose_est > 130:
            insulin_dose = 0.3
        else:
            insulin_dose = 0.1

        return np.array([0.4, 0.3, insulin_dose], dtype=np.float32)

    def _default_heuristic(self, obs: np.ndarray) -> np.ndarray:
        """Default: mid-range action for unknown environments."""
        action_dim = self.env.action_space.shape[0]
        return np.full(action_dim, 0.5, dtype=np.float32)

    def evaluate(self, n_episodes: int = 50) -> dict[str, float]:
        """Evaluate the heuristic agent over multiple episodes."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = self.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "n_episodes": n_episodes,
        }
