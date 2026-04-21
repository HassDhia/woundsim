"""Random baseline agent for WoundSim environments."""

import gymnasium as gym
import numpy as np


class RandomAgent:
    """Agent that samples random actions from the environment's action space.

    Serves as a lower-bound baseline for comparing learned policies.
    """

    def __init__(self, env: gym.Env, seed: int | None = None):
        self.env = env
        self.action_space = env.action_space
        if seed is not None:
            self.action_space.seed(seed)

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        """Sample a random action.

        Args:
            observation: Current observation (ignored).
            deterministic: Ignored for random agent.

        Returns:
            Tuple of (action, None) matching SB3 interface.
        """
        return self.action_space.sample(), None

    def evaluate(self, n_episodes: int = 50) -> dict[str, float]:
        """Evaluate the random agent over multiple episodes.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Dictionary with mean_reward, std_reward, n_episodes.
        """
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
