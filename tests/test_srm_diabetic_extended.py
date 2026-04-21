"""SR-M claim: diabetic_extended

Extended diabetic wound model: 7 state variables coupling macrophage
polarization with glucose–insulin physiology.
"""

from __future__ import annotations

import gymnasium as gym
import woundsim  # noqa: F401


def test_diabetic_observation_is_7_dim():
    env = gym.make("woundsim/WoundDiabetic-v0")
    assert env.observation_space.shape == (7,), (
        f"paper claims 7-state coupled macrophage+glucose-insulin model; "
        f"env observes {env.observation_space.shape}"
    )
