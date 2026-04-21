"""Tests for environment wrappers."""

import gymnasium as gym
import numpy as np

import woundsim  # noqa: F401
from woundsim.envs.wrappers import ClipAction, FlattenAction, NormalizeReward, TimeLimit


class TestNormalizeReward:
    def test_normalized_reward_is_finite(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        wrapped = NormalizeReward(env)
        wrapped.reset(seed=42)
        for _ in range(20):
            action = wrapped.action_space.sample()
            _, reward, terminated, truncated, info = wrapped.step(action)
            assert np.isfinite(reward)
            assert "raw_reward" in info
            if terminated or truncated:
                wrapped.reset(seed=42)
        wrapped.close()

    def test_raw_reward_preserved(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        wrapped = NormalizeReward(env)
        wrapped.reset(seed=42)
        action = wrapped.action_space.sample()
        _, _, _, _, info = wrapped.step(action)
        assert "raw_reward" in info
        assert np.isfinite(info["raw_reward"])
        wrapped.close()


class TestClipAction:
    def test_clips_out_of_bound_actions(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        wrapped = ClipAction(env)
        wrapped.reset(seed=42)
        # Action out of bounds
        action = np.array([1.5], dtype=np.float32)
        obs, reward, _, _, _ = wrapped.step(action)
        assert np.isfinite(reward)
        wrapped.close()

    def test_preserves_valid_actions(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        wrapped = ClipAction(env)
        wrapped.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, _, _, _ = wrapped.step(action)
        assert np.isfinite(reward)
        wrapped.close()


class TestFlattenAction:
    def test_action_space_is_1d(self):
        env = gym.make("woundsim/WoundIschemic-v0")
        wrapped = FlattenAction(env)
        assert len(wrapped.action_space.shape) == 1
        wrapped.close()

    def test_step_with_flat_action(self):
        env = gym.make("woundsim/WoundIschemic-v0")
        wrapped = FlattenAction(env)
        wrapped.reset(seed=42)
        action = wrapped.action_space.sample()
        obs, reward, _, _, _ = wrapped.step(action)
        assert np.isfinite(reward)
        wrapped.close()


class TestTimeLimit:
    def test_truncates_at_limit(self):
        env = gym.make("woundsim/WoundMacrophage-v0")
        wrapped = TimeLimit(env, max_steps=5)
        wrapped.reset(seed=42)
        for i in range(5):
            action = wrapped.action_space.sample()
            _, _, terminated, truncated, _ = wrapped.step(action)
            if i < 4:
                assert not truncated or terminated
        # The 5th step should be truncated
        assert truncated or terminated
        wrapped.close()
