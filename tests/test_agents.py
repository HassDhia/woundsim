"""Tests for agent implementations."""

import gymnasium as gym
import numpy as np
import pytest

import woundsim  # noqa: F401
from woundsim.agents.heuristic import HeuristicAgent
from woundsim.agents.random_agent import RandomAgent


class TestRandomAgent:
    @pytest.fixture
    def env(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        yield env
        env.close()

    def test_predict_returns_valid_action(self, env):
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action, state = agent.predict(obs)
        assert action.shape == env.action_space.shape
        assert state is None

    def test_predict_in_bounds(self, env):
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action, _ = agent.predict(obs)
            assert np.all(action >= env.action_space.low)
            assert np.all(action <= env.action_space.high)

    def test_evaluate_returns_dict(self, env):
        agent = RandomAgent(env, seed=42)
        results = agent.evaluate(n_episodes=3)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "n_episodes" in results
        assert results["n_episodes"] == 3

    def test_evaluate_finite_rewards(self, env):
        agent = RandomAgent(env, seed=42)
        results = agent.evaluate(n_episodes=3)
        assert np.isfinite(results["mean_reward"])
        assert np.isfinite(results["std_reward"])


class TestHeuristicAgent:
    @pytest.fixture(params=[
        "woundsim/WoundMacrophage-v0",
        "woundsim/WoundIschemic-v0",
        "woundsim/WoundHBOT-v0",
        "woundsim/WoundDiabetic-v0",
    ])
    def env_and_id(self, request):
        env_id = request.param
        env = gym.make(env_id)
        yield env, env_id
        env.close()

    def test_predict_returns_valid_action(self, env_and_id):
        env, env_id = env_and_id
        agent = HeuristicAgent(env, env_id=env_id)
        obs, _ = env.reset(seed=42)
        action, state = agent.predict(obs)
        assert action.shape == env.action_space.shape
        assert state is None

    def test_predict_in_bounds(self, env_and_id):
        env, env_id = env_and_id
        agent = HeuristicAgent(env, env_id=env_id)
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs)
        assert np.all(action >= env.action_space.low)
        assert np.all(action <= env.action_space.high)

    def test_evaluate_returns_dict(self, env_and_id):
        env, env_id = env_and_id
        agent = HeuristicAgent(env, env_id=env_id)
        results = agent.evaluate(n_episodes=3)
        assert "mean_reward" in results
        assert results["n_episodes"] == 3

    def test_macrophage_heuristic_value(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        agent = HeuristicAgent(env, env_id="woundsim/WoundMacrophage-v0")
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs)
        assert action[0] == pytest.approx(0.5)
        env.close()

    def test_ischemic_heuristic_values(self):
        env = gym.make("woundsim/WoundIschemic-v0")
        agent = HeuristicAgent(env, env_id="woundsim/WoundIschemic-v0")
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs)
        assert action[0] == pytest.approx(1.0)
        assert action[1] == pytest.approx(0.5)
        env.close()

    def test_hbot_heuristic_values(self):
        env = gym.make("woundsim/WoundHBOT-v0")
        agent = HeuristicAgent(env, env_id="woundsim/WoundHBOT-v0")
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs)
        assert action[0] == pytest.approx(0.7)
        assert action[1] == pytest.approx(0.75)
        env.close()

    def test_diabetic_heuristic_insulin_scaling(self):
        env = gym.make("woundsim/WoundDiabetic-v0")
        agent = HeuristicAgent(env, env_id="woundsim/WoundDiabetic-v0")
        # Low glucose obs -> low insulin
        obs_low = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.1])  # G~103
        action_low, _ = agent.predict(obs_low)
        # High glucose obs -> high insulin
        obs_high = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.1])  # G~367
        action_high, _ = agent.predict(obs_high)
        assert action_high[2] > action_low[2]
        env.close()
