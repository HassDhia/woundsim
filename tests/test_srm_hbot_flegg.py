"""SR-M claim: hbot_flegg

Flegg 2009/2010-type HBOT + angiogenesis as a 4-variable ODE, with
the Flegg O_thresh = 40 mmHg angiogenesis-suppression threshold.
"""

from __future__ import annotations

import gymnasium as gym
import woundsim  # noqa: F401
from woundsim.models.flegg import FleggParams


def test_hbot_observation_is_4_dim():
    env = gym.make("woundsim/WoundHBOT-v0")
    assert env.observation_space.shape == (4,), (
        f"paper claims Flegg 4-state model; "
        f"env observes {env.observation_space.shape}"
    )


def test_flegg_o_thresh_matches_published_value():
    """Flegg 2009 reports O_thresh ≈ 40 mmHg as the oxygen threshold above
    which angiogenesis is suppressed. Default parameters must match."""
    p = FleggParams()
    assert abs(p.O_thresh - 40.0) < 1e-6, (
        f"paper §6.1 / Flegg 2009 target: O_thresh = 40 mmHg; "
        f"got {p.O_thresh} mmHg"
    )
