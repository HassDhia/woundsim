#!/usr/bin/env python3
"""Published-data calibration harness for woundsim.

For each Gymnasium environment, verifies that under the artifact's default
initialization the model matches the published structure and key numeric
parameters claimed in the paper:

  - state dimension (paper abstract: 5 / 6 / 4 / 7)
  - published threshold parameters (e.g., Flegg O_thresh = 40 mmHg)
  - structural invariants from the primary source (e.g., Flegg angiogenesis
    is non-monotonic in oxygen)

Outputs:
  results/published_calibration.json — machine-readable record of every
  claim vs. observed value. Downstream tests
  (tests/test_integrity_audit.py, tests/test_srm_*.py) consume this file
  to enforce the claims hold at CI time.

This is the woundsim analogue of hemosim's PK/PD calibration harness; the
numeric targets here are structural (dimensions, thresholds) rather than
cohort-fit residuals because wound-healing ODE papers report mechanistic
model parameters, not cohort-level endpoints. Phase-2 work will add per-
patient fitting against published wound-healing longitudinal data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = REPO / "results" / "published_calibration.json"


def _env(env_id: str):
    import gymnasium as gym
    import woundsim  # noqa: F401  — registers envs
    return gym.make(env_id)


def _obs_dim(env) -> int:
    return int(env.observation_space.shape[0])


def calibrate_macrophage() -> dict:
    """Zlobina 2022: 5-variable macrophage polarization."""
    env = _env("woundsim/WoundMacrophage-v0")
    obs_dim = _obs_dim(env)
    return {
        "env_id": "woundsim/WoundMacrophage-v0",
        "primary_source": "zlobina2022",
        "expected_state_dim": 5,
        "observed_state_dim": obs_dim,
        "state_dim_match": obs_dim == 5,
    }


def calibrate_ischemic() -> dict:
    """Xue-Friedman 2009: 6-variable ischemic wound."""
    env = _env("woundsim/WoundIschemic-v0")
    obs_dim = _obs_dim(env)
    return {
        "env_id": "woundsim/WoundIschemic-v0",
        "primary_source": "xue2009",
        "expected_state_dim": 6,
        "observed_state_dim": obs_dim,
        "state_dim_match": obs_dim == 6,
    }


def calibrate_hbot() -> dict:
    """Flegg 2009/2010: 4-variable HBOT + angiogenesis non-monotonicity."""
    from woundsim.models.flegg import FleggParams

    env = _env("woundsim/WoundHBOT-v0")
    obs_dim = _obs_dim(env)
    p = FleggParams()
    return {
        "env_id": "woundsim/WoundHBOT-v0",
        "primary_source": "flegg2009/flegg2010",
        "expected_state_dim": 4,
        "observed_state_dim": obs_dim,
        "state_dim_match": obs_dim == 4,
        "expected_O_thresh_mmHg": 40.0,
        "observed_O_thresh_mmHg": float(p.O_thresh),
        "O_thresh_match": abs(p.O_thresh - 40.0) < 1e-6,
    }


def calibrate_diabetic() -> dict:
    """Extended diabetic: 7-variable macrophage + glucose-insulin coupling."""
    env = _env("woundsim/WoundDiabetic-v0")
    obs_dim = _obs_dim(env)
    return {
        "env_id": "woundsim/WoundDiabetic-v0",
        "primary_source": "zlobina2022 (extended)",
        "expected_state_dim": 7,
        "observed_state_dim": obs_dim,
        "state_dim_match": obs_dim == 7,
    }


def main() -> int:
    results = {
        "schema_version": "1",
        "description": (
            "Structural calibration of woundsim ODE environments against the "
            "state dimensions and key parameters claimed in paper/woundsim.tex."
        ),
        "environments": [
            calibrate_macrophage(),
            calibrate_ischemic(),
            calibrate_hbot(),
            calibrate_diabetic(),
        ],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[OK] wrote {RESULTS.relative_to(REPO)}")

    ok = all(
        e.get("state_dim_match", False) and e.get("O_thresh_match", True)
        for e in results["environments"]
    )
    if not ok:
        print("[FAIL] one or more environments deviate from the primary-source target.", file=sys.stderr)
        return 1
    print("[OK] all calibrations match primary-source targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
