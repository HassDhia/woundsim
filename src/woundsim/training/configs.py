"""Shared training configurations for all WoundSim environments.

This is the single source of truth for hyperparameters used by both
the ppo.py CLI and train_all.py.
"""

ENV_CONFIGS: dict[str, dict] = {
    "woundsim/WoundMacrophage-v0": {
        "total_timesteps": 200_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "difficulty": "medium",
    },
    "woundsim/WoundIschemic-v0": {
        "total_timesteps": 300_000,
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "difficulty": "moderate",
    },
    "woundsim/WoundHBOT-v0": {
        "total_timesteps": 200_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "difficulty": "chronic",
    },
    "woundsim/WoundDiabetic-v0": {
        "total_timesteps": 500_000,
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "difficulty": "moderate",
    },
}
