"""Tests for WoundIschemic-v0 environment."""

import gymnasium as gym
import numpy as np
import pytest

import woundsim  # noqa: F401


class TestWoundIschemicEnv:
    @pytest.fixture
    def env(self):
        env = gym.make("woundsim/WoundIschemic-v0", difficulty="moderate")
        yield env
        env.close()

    def test_make_env(self):
        env = gym.make("woundsim/WoundIschemic-v0")
        assert env is not None
        env.close()

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (6,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (2,)

    def test_action_space_bounds(self, env):
        np.testing.assert_array_almost_equal(env.action_space.low, [0.0, 0.0])
        np.testing.assert_array_almost_equal(env.action_space.high, [1.0, 1.0])

    def test_reset_returns_obs_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,)
        assert isinstance(info, dict)

    def test_reset_obs_in_bounds(self, env):
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

    def test_info_contains_wound_area(self, env):
        _, info = env.reset(seed=42)
        assert "wound_area" in info
        assert "oxygen" in info
        assert "ecm" in info

    def test_reward_is_finite(self, env):
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)

    def test_seed_reproducibility(self):
        env1 = gym.make("woundsim/WoundIschemic-v0", difficulty="moderate")
        env2 = gym.make("woundsim/WoundIschemic-v0", difficulty="moderate")
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env1.close()
        env2.close()

    def test_difficulty_mild(self):
        env = gym.make("woundsim/WoundIschemic-v0", difficulty="mild")
        _, info = env.reset(seed=42)
        assert info["wound_area"] == pytest.approx(0.3, abs=0.05)
        env.close()

    def test_difficulty_severe(self):
        env = gym.make("woundsim/WoundIschemic-v0", difficulty="severe")
        _, info = env.reset(seed=42)
        assert info["wound_area"] == pytest.approx(0.8, abs=0.05)
        env.close()
