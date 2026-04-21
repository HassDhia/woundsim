# WoundSim

**Gymnasium-compatible reinforcement learning environments for wound healing treatment optimization**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-190%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/woundsim.svg)](https://pypi.org/project/woundsim/)
![version](https://img.shields.io/badge/version-0.1.1-blue.svg)

---

WoundSim provides four Gymnasium-compatible RL environments for wound healing treatment optimization, each wrapping a validated ODE model from the wound healing literature. All model parameters are sourced from peer-reviewed publications with inline `# SOURCE:` comments. The package includes random, clinical heuristic, and PPO baselines with full Stable-Baselines3 compatibility.

## Installation

```bash
pip install woundsim              # Core (numpy, scipy, gymnasium)
pip install woundsim[rl]          # + SB3, PyTorch for RL training
pip install woundsim[all]         # Everything
```

Development install:

```bash
git clone https://github.com/HassDhia/woundsim.git
cd woundsim
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
import woundsim
from woundsim.agents.heuristic import HeuristicAgent

env = gym.make("woundsim/WoundMacrophage-v0", difficulty="medium")
agent = HeuristicAgent(env, env_id="woundsim/WoundMacrophage-v0")
obs, info = env.reset(seed=42)

done = False
total_reward = 0.0
while not done:
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode reward: {total_reward:.2f}")
env.close()
```

## Environments

| Environment | Model | State Dim | Action Dim | Source |
|---|---|---|---|---|
| `WoundMacrophage-v0` | Zlobina macrophage polarization | 5 | 1 | Zlobina & Gomez (2022) |
| `WoundIschemic-v0` | Simplified Xue-Friedman ischemic | 6 | 2 | Xue, Friedman & Sen (2009) |
| `WoundHBOT-v0` | Flegg HBOT angiogenesis | 4 | 2 | Flegg et al. (2009, 2015) |
| `WoundDiabetic-v0` | Extended diabetic inflammation | 7 | 3 | Waugh & Sherratt (2006) |

### WoundMacrophage-v0

Five-variable ODE model of macrophage M1/M2 polarization from Zlobina et al. (2022). The agent controls a polarization treatment signal to balance debris clearance (M1) with tissue regeneration (M2).

- **Observation:** `[debris, M1, M2, granulation, new_tissue]` (normalized to [0,1])
- **Action:** `[polarization_signal]` in [0, 1]
- **Reward:** Penalizes debris and treatment cost; rewards tissue growth
- **Difficulties:** easy (a0=0.3), medium (a0=0.6), hard (a0=0.9 + noise)

### WoundIschemic-v0

Six-variable model simplified from Xue & Friedman (2009). The agent controls revascularization intensity and growth factor application to heal ischemic wounds.

- **Observation:** `[wound_area, oxygen, VEGF, macrophages, fibroblasts, ECM]`
- **Action:** `[revascularization, growth_factor]` in [0, 1]
- **Difficulties:** mild, moderate, severe

### WoundHBOT-v0

Four-variable HBOT angiogenesis model from Flegg et al. (2009, 2015). The agent controls hyperbaric oxygen session parameters to promote vascularization and wound closure.

- **Observation:** `[capillary_tips, sprouts, oxygen, wound_area]`
- **Action:** `[intensity, duration_fraction]` in [0, 1]
- **Difficulties:** acute, chronic, non-healing

### WoundDiabetic-v0

Seven-variable diabetic wound model incorporating glucose-insulin dynamics from Waugh & Sherratt (2006) with macrophage polarization from Zlobina et al. (2022). The agent must simultaneously manage wound treatment and glycemic control.

- **Observation:** `[wound, debris, M1, M2, glucose, insulin, ECM]`
- **Action:** `[polarization, growth_factor, insulin_dose]` in [0, 1]
- **Difficulties:** well-controlled, moderate, uncontrolled

## Architecture

```
woundsim/
  src/woundsim/
    models/        # ODE wound healing models (Zlobina, Xue-Friedman, Flegg)
    envs/          # Gymnasium environments wrapping each model
    agents/        # Random, heuristic, PPO agent implementations
    training/      # Training infrastructure with shared configs
    benchmarks/    # Systematic benchmark evaluation runner
  tests/           # 173 unit and integration tests
  paper/           # LaTeX paper and publication figures
  results/         # Training results (JSON)
```

## Training

```bash
# Train PPO on all environments and evaluate against baselines
python train_all.py

# Generate publication-quality figures from training results
python generate_figures.py
```

## Paper

The accompanying paper is available at:
- [PDF (GitHub)](https://github.com/HassDhia/woundsim/blob/main/paper/woundsim.pdf)

## Citation

If you use WoundSim in your research, please cite the software:

```bibtex
@software{dhia2026woundsim_software,
  author = {Dhia, Hass},
  title = {WoundSim: Gymnasium-Compatible Reinforcement Learning Environments for Wound Healing Treatment Optimization},
  year = {2026},
  publisher = {Smart Technology Investments Research Institute},
  url = {https://github.com/HassDhia/woundsim}
}
```

To cite the accompanying paper:

```bibtex
@misc{dhia2026woundsim,
  author = {Dhia, Hass},
  title = {WoundSim: Gymnasium-Compatible Reinforcement Learning Environments for Wound Healing Treatment Optimization},
  year = {2026},
  howpublished = {\url{https://github.com/HassDhia/woundsim/blob/main/paper/woundsim.pdf}},
  institution = {Smart Technology Investments Research Institute}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Hass Dhia - Smart Technology Investments Research Institute
- Email: hass@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
