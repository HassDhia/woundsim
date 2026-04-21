"""SR-M claim: ischemic_xue_friedman

Xue-Friedman 2009-type ischemic wound as a 6-variable ODE.
"""

from __future__ import annotations

import gymnasium as gym
import woundsim  # noqa: F401


def test_ischemic_observation_is_6_dim():
    env = gym.make("woundsim/WoundIschemic-v0")
    assert env.observation_space.shape == (6,), (
        f"paper claims Xue-Friedman 6-state model; "
        f"env observes {env.observation_space.shape}"
    )
