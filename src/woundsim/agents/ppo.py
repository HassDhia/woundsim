"""PPO agent with CLI entrypoint for WoundSim environments.

Wraps Stable-Baselines3 PPO for training and evaluation on wound
healing environments.
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


def make_env(env_id: str, difficulty: str | None = None, seed: int = 42) -> gym.Env:
    """Create a WoundSim environment with optional difficulty override."""
    import woundsim  # noqa: F401 - triggers registration

    kwargs = {}
    if difficulty is not None:
        kwargs["difficulty"] = difficulty
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    return env


def train_ppo(
    env_id: str,
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    difficulty: str | None = None,
    save_path: str | None = None,
    verbose: int = 1,
):
    """Train a PPO agent on a WoundSim environment.

    Args:
        env_id: Gymnasium environment ID.
        total_timesteps: Total training steps.
        learning_rate: PPO learning rate.
        n_steps: Steps per rollout.
        batch_size: Minibatch size.
        n_epochs: Optimization epochs per rollout.
        gamma: Discount factor.
        seed: Random seed.
        difficulty: Environment difficulty tier.
        save_path: Path to save trained model.
        verbose: Verbosity level.

    Returns:
        Trained PPO model.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 required. Install with: pip install woundsim[rl]")
        sys.exit(1)

    env = make_env(env_id, difficulty=difficulty, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
    )

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    env.close()
    return model


def evaluate_model(model, env: gym.Env, n_episodes: int = 50) -> dict[str, float]:
    """Evaluate a trained model on an environment.

    Args:
        model: Trained SB3 model (or any object with predict(obs)).
        env: Gymnasium environment.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dictionary with mean_reward, std_reward, n_episodes.
    """
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "n_episodes": n_episodes,
    }


def main():
    """CLI entrypoint for PPO training and evaluation."""
    parser = argparse.ArgumentParser(description="Train PPO on WoundSim environments")
    parser.add_argument(
        "--env",
        type=str,
        default="woundsim/WoundMacrophage-v0",
        help="Environment ID",
    )
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", type=str, default=None, help="Difficulty tier")
    parser.add_argument("--save", type=str, default=None, help="Model save path")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Eval episodes")
    parser.add_argument("--output", type=str, default=None, help="Results JSON path")

    args = parser.parse_args()

    print(f"Training PPO on {args.env} for {args.timesteps} steps (seed={args.seed})")

    model = train_ppo(
        env_id=args.env,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        difficulty=args.difficulty,
        save_path=args.save,
    )

    env = make_env(args.env, difficulty=args.difficulty, seed=args.seed)
    results = evaluate_model(model, env, n_episodes=args.eval_episodes)
    env.close()

    print(f"Results: mean={results['mean_reward']:.2f}, std={results['std_reward']:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
