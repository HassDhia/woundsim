"""Evaluation utilities for WoundSim agents."""

import gymnasium as gym
import numpy as np


def evaluate_agent(
    agent,
    env: gym.Env,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> dict[str, float]:
    """Evaluate any agent with a predict() method on an environment.

    Args:
        agent: Agent with predict(obs, deterministic=...) method.
        env: Gymnasium environment.
        n_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.

    Returns:
        Dictionary with evaluation statistics.
    """
    rewards = []
    episode_lengths = []
    healed_count = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        episode_lengths.append(steps)
        if info.get("healed", False):
            healed_count += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "heal_rate": healed_count / n_episodes,
        "n_episodes": n_episodes,
    }
