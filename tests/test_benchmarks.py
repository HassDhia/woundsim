"""Tests for benchmark runner."""


import woundsim  # noqa: F401
from woundsim.benchmarks.environments import BENCHMARK_ENVS
from woundsim.benchmarks.runner import BenchmarkRunner


class TestBenchmarkEnvironments:
    def test_benchmark_envs_not_empty(self):
        assert len(BENCHMARK_ENVS) == 4

    def test_all_envs_have_required_fields(self):
        for env_config in BENCHMARK_ENVS:
            assert "env_id" in env_config
            assert "name" in env_config
            assert "difficulties" in env_config
            assert "default_difficulty" in env_config

    def test_all_env_ids_are_registered(self):
        import gymnasium as gym
        for env_config in BENCHMARK_ENVS:
            env = gym.make(env_config["env_id"])
            assert env is not None
            env.close()


class TestBenchmarkRunner:
    def test_runner_init(self):
        runner = BenchmarkRunner(n_episodes=2, seed=42)
        assert runner.n_episodes == 2
        assert runner.seed == 42

    def test_run_baselines(self):
        runner = BenchmarkRunner(n_episodes=2, seed=42)
        results = runner.run_baselines()
        assert len(results) == 4
        for _env_id, res in results.items():
            assert "random" in res
            assert "heuristic" in res
            assert "mean_reward" in res["random"]
            assert "mean_reward" in res["heuristic"]

    def test_save_results(self, tmp_path):
        runner = BenchmarkRunner(n_episodes=2, seed=42)
        results = runner.run_baselines()
        save_path = str(tmp_path / "test_results.json")
        runner.save_results(results, path=save_path)
        import json
        with open(save_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 4
