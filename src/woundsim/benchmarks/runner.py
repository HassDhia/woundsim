"""Benchmark runner for systematic evaluation of agents across environments."""

import json
from pathlib import Path

import gymnasium as gym

import woundsim  # noqa: F401
from woundsim.agents.heuristic import HeuristicAgent
from woundsim.agents.random_agent import RandomAgent
from woundsim.benchmarks.environments import BENCHMARK_ENVS
from woundsim.training.evaluate import evaluate_agent


class BenchmarkRunner:
    """Run benchmarks across all WoundSim environments.

    Evaluates random and heuristic baselines on each environment
    at each difficulty level.
    """

    def __init__(self, n_episodes: int = 10, seed: int = 42):
        self.n_episodes = n_episodes
        self.seed = seed

    def run_baselines(self) -> dict:
        """Run random and heuristic baselines on all environments.

        Returns:
            Dictionary of results keyed by environment ID.
        """
        results = {}

        for env_config in BENCHMARK_ENVS:
            env_id = env_config["env_id"]
            difficulty = env_config["default_difficulty"]

            env = gym.make(env_id, difficulty=difficulty)
            env.reset(seed=self.seed)

            # Random baseline
            random_agent = RandomAgent(env, seed=self.seed)
            random_results = evaluate_agent(
                random_agent, env, n_episodes=self.n_episodes
            )

            # Heuristic baseline
            heuristic_agent = HeuristicAgent(env, env_id=env_id)
            heuristic_results = evaluate_agent(
                heuristic_agent, env, n_episodes=self.n_episodes
            )

            results[env_id] = {
                "name": env_config["name"],
                "difficulty": difficulty,
                "random": random_results,
                "heuristic": heuristic_results,
                "heuristic_vs_random_ratio": (
                    heuristic_results["mean_reward"] / random_results["mean_reward"]
                    if random_results["mean_reward"] != 0
                    else float("inf")
                ),
            }

            env.close()

        return results

    def save_results(self, results: dict, path: str = "results/benchmark_results.json"):
        """Save benchmark results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
