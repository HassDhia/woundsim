"""Training utilities for WoundSim environments."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


def train_single_env(
    env_id: str,
    config: dict,
    seed: int = 42,
    save_dir: str = "results/models",
    verbose: int = 1,
) -> tuple:
    """Train PPO on a single environment and return model + training curve.

    Args:
        env_id: Gymnasium environment ID.
        config: Training configuration dict from ENV_CONFIGS.
        seed: Random seed.
        save_dir: Directory to save trained models.
        verbose: Verbosity level.

    Returns:
        Tuple of (trained_model, training_curve) where training_curve
        is a list of [step, mean_reward, std_reward].
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("stable-baselines3 required. Install with: pip install woundsim[rl]")
        sys.exit(1)

    import woundsim  # noqa: F401

    difficulty = config.get("difficulty")
    kwargs = {"difficulty": difficulty} if difficulty else {}
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)

    # Callback to record training curve
    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.current_rewards = []
            self.training_curve = []

        def _on_step(self) -> bool:
            # Collect rewards from info buffer
            if len(self.model.ep_info_buffer) > 0:
                recent = [ep["r"] for ep in self.model.ep_info_buffer]
                if len(recent) >= 5:
                    self.training_curve.append([
                        self.num_timesteps,
                        float(np.mean(recent)),
                        float(np.std(recent)),
                    ])
            return True

    callback = RewardCallback()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        seed=seed,
        verbose=verbose,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
    )

    # Save model
    save_path = Path(save_dir) / env_id.replace("/", "_")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    env.close()
    return model, callback.training_curve
