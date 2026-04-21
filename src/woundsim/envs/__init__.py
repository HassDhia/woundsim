"""Gymnasium environment registration for WoundSim."""

import gymnasium as gym


def register_envs():
    """Register all WoundSim environments with Gymnasium."""
    gym.register(
        id="woundsim/WoundMacrophage-v0",
        entry_point="woundsim.envs.macrophage:WoundMacrophageEnv",
        max_episode_steps=200,
    )
    gym.register(
        id="woundsim/WoundIschemic-v0",
        entry_point="woundsim.envs.ischemic:WoundIschemicEnv",
        max_episode_steps=300,
    )
    gym.register(
        id="woundsim/WoundHBOT-v0",
        entry_point="woundsim.envs.hbot:WoundHBOTEnv",
        max_episode_steps=240,
    )
    gym.register(
        id="woundsim/WoundDiabetic-v0",
        entry_point="woundsim.envs.diabetic:WoundDiabeticEnv",
        max_episode_steps=300,
    )
