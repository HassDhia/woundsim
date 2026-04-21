"""SR-M claim: macrophage_zlobina

Zlobina 2022-type macrophage polarization as a 5-variable ODE, with
M1→M2 transition responsive to anti-inflammatory input.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import woundsim  # noqa: F401  — registers envs


def test_macrophage_observation_is_5_dim():
    env = gym.make("woundsim/WoundMacrophage-v0")
    assert env.observation_space.shape == (5,), (
        f"paper claims Zlobina-type 5-state model; "
        f"env observes {env.observation_space.shape}"
    )


def test_m1_declines_under_anti_inflammatory_input():
    """Under sustained anti-inflammatory action (u → high M2-promoting),
    M1 at the end of the episode should be ≤ M1 at the start."""
    env = gym.make("woundsim/WoundMacrophage-v0")
    obs, _ = env.reset(seed=0)
    m1_initial = float(obs[0])
    high_anti_inflammatory = np.array([1.0], dtype=np.float32)
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(high_anti_inflammatory)
        if term or trunc:
            break
    m1_final = float(obs[0])
    assert m1_final <= m1_initial + 1e-6, (
        f"M1 should decline under sustained anti-inflammatory input; "
        f"observed {m1_initial:.3f} → {m1_final:.3f}"
    )
