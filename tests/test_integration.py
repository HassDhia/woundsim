"""Integration tests for full episode runs across all environments."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded

import woundsim  # noqa: F401
from woundsim.agents.heuristic import HeuristicAgent
from woundsim.agents.random_agent import RandomAgent
from woundsim.training.configs import ENV_CONFIGS
from woundsim.training.evaluate import evaluate_agent

ENV_IDS = [
    "woundsim/WoundMacrophage-v0",
    "woundsim/WoundIschemic-v0",
    "woundsim/WoundHBOT-v0",
    "woundsim/WoundDiabetic-v0",
]


class TestFullEpisodeRuns:
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_full_episode_random(self, env_id):
        """Full episode with random agent completes without error."""
        env = gym.make(env_id)
        agent = RandomAgent(env, seed=42)
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        assert steps > 0
        assert np.isfinite(total_reward)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_full_episode_heuristic(self, env_id):
        """Full episode with heuristic agent completes without error."""
        env = gym.make(env_id)
        agent = HeuristicAgent(env, env_id=env_id)
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        assert steps > 0
        assert np.isfinite(total_reward)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_observations_always_valid(self, env_id):
        """All observations stay within bounds throughout an episode."""
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            assert np.all(obs >= 0.0), f"Negative obs in {env_id}: {obs}"
            assert np.all(obs <= 1.0), f"Obs > 1 in {env_id}: {obs}"
            assert np.all(np.isfinite(obs)), f"Non-finite obs in {env_id}: {obs}"
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_info_always_has_healed(self, env_id):
        """Info dict always contains healed field."""
        env = gym.make(env_id)
        _, info = env.reset(seed=42)
        assert "healed" in info
        done = False
        obs = env.observation_space.sample()
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            assert "healed" in info
            done = terminated or truncated
        env.close()


class TestEvaluationInfrastructure:
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_evaluate_agent_works(self, env_id):
        """evaluate_agent returns valid results."""
        env = gym.make(env_id)
        agent = RandomAgent(env, seed=42)
        results = evaluate_agent(agent, env, n_episodes=3)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "heal_rate" in results
        assert "mean_length" in results
        assert results["n_episodes"] == 3
        assert np.isfinite(results["mean_reward"])
        env.close()


class TestEnvConfigs:
    def test_all_envs_have_configs(self):
        """Every registered env has a training config."""
        for env_id in ENV_IDS:
            assert env_id in ENV_CONFIGS

    def test_configs_have_required_keys(self):
        """All configs have the required hyperparameter keys."""
        required = [
            "total_timesteps",
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "difficulty",
        ]
        for env_id, config in ENV_CONFIGS.items():
            for key in required:
                assert key in config, f"{env_id} missing {key}"

    def test_configs_have_valid_values(self):
        """Config values are within reasonable ranges."""
        for _env_id, config in ENV_CONFIGS.items():
            assert config["total_timesteps"] > 0
            assert 0 < config["learning_rate"] < 1
            assert config["n_steps"] > 0
            assert config["batch_size"] > 0
            assert config["n_epochs"] > 0
            assert 0 < config["gamma"] <= 1


class TestEdgeCases:
    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_zero_action(self, env_id):
        """Zero action does not crash."""
        env = gym.make(env_id)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_max_action(self, env_id):
        """Maximum action does not crash."""
        env = gym.make(env_id)
        env.reset(seed=42)
        action = np.ones(env.action_space.shape, dtype=np.float32)
        obs, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_repeated_same_action(self, env_id):
        """Repeated constant action produces stable trajectory."""
        env = gym.make(env_id)
        env.reset(seed=42)
        action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        for _ in range(30):
            obs, reward, terminated, truncated, _ = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_multiple_resets(self, env_id):
        """Multiple resets produce valid observations."""
        env = gym.make(env_id)
        for seed in [0, 42, 100]:
            obs, info = env.reset(seed=seed)
            assert np.all(np.isfinite(obs))
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
        env.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_step_without_reset_raises(self, env_id):
        """Stepping without reset raises an error."""
        env = gym.make(env_id)
        # Gymnasium wraps and raises ResetNeeded before our AssertionError
        action = env.action_space.sample()
        with pytest.raises((AssertionError, ResetNeeded)):
            env.step(action)
        env.close()
