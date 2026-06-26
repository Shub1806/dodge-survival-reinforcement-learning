import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from dodge_env import DodgeEnv


def parse_args():
    p = argparse.ArgumentParser(description="Train a PPO agent on the dodge game.")
    p.add_argument("--timesteps", type=int, default=300_000,
                   help="total environment steps to train for")
    p.add_argument("--envs", type=int, default=4,
                   help="number of parallel environments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="models/dodge_ppo",
                   help="output path for the saved model (no extension)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vec_env = make_vec_env(DodgeEnv, n_envs=args.envs, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved trained model to {args.out}.zip")


if __name__ == "__main__":
    main()
