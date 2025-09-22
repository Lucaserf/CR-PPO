import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.atari_wrappers import AtariWrapper
from cdpo_stablebaselines.CDPO import CDPO
import argparse

parser = argparse.ArgumentParser(description='CDPO')
parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--entropy_value', type=str, default='1e-1', help='Entropy value')
parser.add_argument('--only_entropy', action="store_true", help='Only entropy')
parser.add_argument("--timesteps", type=int, default=1000000, help='Total timesteps')
parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy')
parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate for policy')
parser.add_argument('--clip_range', type=float, default=0.1, help='Clip range for PPO')
parser.add_argument("--epochs", type=int, default=4, help='PPO epochs')
parser.add_argument("--n_steps", type=int, default=128, help='Number of steps between updates')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma for advantage computation')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda for GAE')
parser.add_argument("--batch_size", type=int, default=256, help='Batch size for PPO')
# parser.add_argument("--env_kwargs", type=dict,default={})
args = parser.parse_args()

env_name = args.env

if env_name in ["CarRacing-v2", "CartPole-v1"]:
    env_kwargs = {'continuous': False} if 'CarRacing' in env_name else {}
    env = make_vec_env(env_name, n_envs=1, seed=args.seed, env_kwargs=env_kwargs)
else:
    env = make_atari_env(env_name, n_envs=8, seed=args.seed)
    env = VecFrameStack(env, n_stack=4)

entropy_string = args.entropy_value
entropy_value = float(args.entropy_value)

# Set th seed
th.manual_seed(args.seed)


if args.only_entropy:
    log_name = f"logs/{env_name}_{args.seed}_{entropy_string}_entropy"
else:
    log_name = f"logs/{env_name}_{args.seed}_{entropy_string}_complexity"


# Create logger
new_logger = configure(log_name, ["csv", "tensorboard"])

# Instantiate the model

if args.only_entropy:
    model = PPO(args.policy, env, verbose=1, seed=args.seed, ent_coef=entropy_value, n_steps=args.n_steps, n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, clip_range=args.clip_range, vf_coef=0.5, gamma=args.gamma, gae_lambda=args.gae_lambda)
else:
    model = CDPO(args.policy, env, verbose=1, seed=args.seed, ent_coef=entropy_value, n_steps=args.n_steps, n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, clip_range=args.clip_range, vf_coef=0.5, gamma=args.gamma, gae_lambda=args.gae_lambda)


model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=args.timesteps)