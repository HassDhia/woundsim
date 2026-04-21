"""Tests for WoundDiabetic-v0 environment."""

import gymnasium as gym
import numpy as np
import pytest

import woundsim  # noqa: F401


class TestWoundDiabeticEnv:
    @pytest.fixture
    def env(self):
        env = gym.make("woundsim/WoundDiabetic-v0", difficulty="moderate")
        yield env
        env.close()

    def test_make_env(self):
        env = gym.make("woundsim/WoundDiabetic-v0")
        assert env is not None
        env.close()

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (7,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (3,)

    def test_action_space_bounds(self, env):
        np.testing.assert_array_almost_equal(env.action_space.low, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(env.action_space.high, [1.0, 1.0, 1.0])

    def test_reset_returns_obs_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,)
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

    def test_info_contains_fields(self, env):
        _, info = env.reset(seed=42)
        assert "wound_area" in info
        assert "glucose" in info
        assert "insulin" in info
        assert "ecm" in info
        assert "debris" in info

    def test_reward_is_finite(self, env):
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)

    def test_seed_reproducibility(self):
        env1 = gym.make("woundsim/WoundDiabetic-v0", difficulty="moderate")
        env2 = gym.make("woundsim/WoundDiabetic-v0", difficulty="moderate")
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env1.close()
        env2.close()

    def test_difficulty_well_controlled(self):
        env = gym.make("woundsim/WoundDiabetic-v0", difficulty="well-controlled")
        _, info = env.reset(seed=42)
        assert info["glucose"] == pytest.approx(120.0, abs=5.0)
        env.close()

    def test_difficulty_uncontrolled(self):
        env = gym.make("woundsim/WoundDiabetic-v0", difficulty="uncontrolled")
        _, info = env.reset(seed=42)
        assert info["glucose"] == pytest.approx(350.0, abs=5.0)
        env.close()
