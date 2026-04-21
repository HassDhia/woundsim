# WoundSim

[![Tests](https://img.shields.io/badge/tests-173%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Gymnasium](https://img.shields.io/badge/gymnasium-compatible-orange)]()

Gymnasium-compatible reinforcement learning environments for wound healing treatment optimization.

## Overview

WoundSim provides four RL environments, each wrapping a validated ODE model from the wound healing literature:

| Environment | Model | State Dim | Action Dim | Source |
|---|---|---|---|---|
| `WoundMacrophage-v0` | Zlobina macrophage polarization | 5 | 1 | Zlobina & Gomez (2022) |
| `WoundIschemic-v0` | Simplified Xue-Friedman ischemic | 6 | 2 | Xue, Friedman & Sen (2009) |
| `WoundHBOT-v0` | Flegg HBOT angiogenesis | 4 | 2 | Flegg et al. (2009, 2015) |
| `WoundDiabetic-v0` | Extended diabetic inflammation | 7 | 3 | Waugh & Sherratt (2006) |

All parameters are sourced from peer-reviewed publications with inline `# SOURCE:` comments.

## Installation

```bash
# Core (environments only)
pip install woundsim

# With RL training support
pip install woundsim[rl]

# Development
pip install woundsim[all]
```

## Quick Start

```python
import gymnasium as gym
import woundsim

# Create environment
env = gym.make("woundsim/WoundMacrophage-v0", difficulty="medium")
obs, info = env.reset(seed=42)

# Run episode
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
import gymnasium as gym
import woundsim

env = gym.make("woundsim/WoundMacrophage-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
```

## Environments

### WoundMacrophage-v0

Five-variable ODE model of macrophage M1/M2 polarization. The agent controls a polarization treatment signal to balance debris clearance (M1) with tissue regeneration (M2).

- **Observation:** `[debris, M1, M2, granulation, new_tissue]` (normalized)
- **Action:** `[polarization_signal]` in [0, 1]
- **Reward:** Penalizes debris and treatment cost; rewards tissue growth
- **Difficulties:** easy (a0=0.3), medium (a0=0.6), hard (a0=0.9)

### WoundIschemic-v0

Six-variable model of ischemic wound healing. The agent controls revascularization and growth factor application.

- **Observation:** `[wound_area, oxygen, VEGF, macrophages, fibroblasts, ECM]`
- **Action:** `[revascularization, growth_factor]` in [0, 1]
- **Difficulties:** mild, moderate, severe

### WoundHBOT-v0

Four-variable HBOT angiogenesis model. The agent controls hyperbaric oxygen session parameters.

- **Observation:** `[capillary_tips, sprouts, oxygen, wound_area]`
- **Action:** `[intensity, duration_fraction]` in [0, 1]
- **Difficulties:** acute, chronic, non-healing

### WoundDiabetic-v0

Seven-variable diabetic wound model with glucose-insulin dynamics. The agent must manage wound treatment AND glycemic control.

- **Observation:** `[wound, debris, M1, M2, glucose, insulin, ECM]`
- **Action:** `[polarization, growth_factor, insulin_dose]` in [0, 1]
- **Difficulties:** well-controlled, moderate, uncontrolled

## Baselines

Three baseline agents are included:

1. **Random:** Uniform random actions (lower bound)
2. **Heuristic:** Clinical protocol approximations
3. **PPO:** Trained with Stable-Baselines3

## Training All Environments

```bash
python train_all.py
```

This trains PPO on all four environments, evaluates against baselines, and saves results to `results/training_results.json`.

## Generating Figures

```bash
python generate_figures.py
```

## Project Structure

```
woundsim/
  src/woundsim/
    envs/          # Gymnasium environments
    models/        # ODE wound healing models
    agents/        # Random, heuristic, PPO agents
    training/      # Training infrastructure
    benchmarks/    # Benchmark runner
  tests/           # 100+ tests
  paper/           # LaTeX paper and figures
  results/         # Training results (JSON)
```

## Citation

```bibtex
@software{dhia2026woundsim,
  title={WoundSim: Gymnasium-Compatible RL Environments for Wound Healing},
  author={Dhia, Hass},
  year={2026},
  url={https://github.com/smarttechinvest/woundsim}
}
```

## License

MIT
