"""Tests for WoundMacrophage-v0 environment."""

import gymnasium as gym
import numpy as np
import pytest

import woundsim  # noqa: F401


class TestWoundMacrophageEnv:
    @pytest.fixture
    def env(self):
        env = gym.make("woundsim/WoundMacrophage-v0", difficulty="medium")
        yield env
        env.close()

    def test_make_env(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        assert env is not None
        env.close()

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (5,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (1,)

    def test_action_space_bounds(self, env):
        assert env.action_space.low[0] == pytest.approx(0.0)
        assert env.action_space.high[0] == pytest.approx(1.0)

    def test_reset_returns_obs_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,)
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
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_bounds(self, env):
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_info_contains_raw_state(self, env):
        _, info = env.reset(seed=42)
        assert "raw_state" in info
        assert info["raw_state"].shape == (5,)

    def test_info_contains_healed(self, env):
        _, info = env.reset(seed=42)
        assert "healed" in info
        assert isinstance(info["healed"], bool)

    def test_reward_is_finite(self, env):
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)

    def test_seed_reproducibility(self):
        env1 = gym.make("woundsim/WoundMacrophage-v0", difficulty="medium")
        env2 = gym.make("woundsim/WoundMacrophage-v0", difficulty="medium")
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        action = np.array([0.5])
        obs1_next, r1, _, _, _ = env1.step(action)
        obs2_next, r2, _, _, _ = env2.step(action)
        np.testing.assert_array_equal(obs1_next, obs2_next)
        assert r1 == pytest.approx(r2)
        env1.close()
        env2.close()

    def test_difficulty_easy(self):
        env = gym.make("woundsim/WoundMacrophage-v0", difficulty="easy")
        _, info = env.reset(seed=42)
        assert info["debris"] == pytest.approx(0.3, abs=0.1)
        env.close()

    def test_difficulty_hard(self):
        env = gym.make("woundsim/WoundMacrophage-v0", difficulty="hard")
        _, info = env.reset(seed=42)
        assert info["debris"] == pytest.approx(0.9, abs=0.1)
        env.close()
