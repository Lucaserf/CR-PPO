import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import TimeLimit
# from stable_baselines3.common.atari_wrappers import AtariWrapper
from crppo_stablebaselines.CRPPO import CRPPO
import argparse
from CARTerpillar.CARTerpillar import CARTerpillarEnv

# CartPole-v1:
#   n_envs: 8
#   n_timesteps: !!float 1e5
#   policy: 'MlpPolicy'
#   n_steps: 32
#   batch_size: 256
#   gae_lambda: 0.8
#   gamma: 0.98
#   n_epochs: 20
#   ent_coef: 0.0
#   learning_rate: lin_0.001
#   clip_range: lin_0.2

parser = argparse.ArgumentParser(description='CRPPO')
parser.add_argument('--env', type=str, default='Pendulum-v1', help='Environment name')
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--entropy_value', type=str, default='0.0', help='Entropy value')
parser.add_argument('--only_entropy', action="store_true", help='Only entropy')
parser.add_argument("--timesteps", type=int, default=100000, help='Total timesteps')
parser.add_argument('--policy', type=str, default='MlpPolicy', help='Policy')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for policy')
parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range for PPO')
parser.add_argument("--epochs", type=int, default=20, help='PPO epochs')
parser.add_argument("--n_steps", type=int, default=32, help='Number of steps between updates')
parser.add_argument('--gamma', type=float, default=0.98, help='Gamma for advantage computation')
parser.add_argument('--gae_lambda', type=float, default=0.8, help='Lambda for GAE')
parser.add_argument("--n_carts", type=int, default=1, help='Number of carts for NCartpole environment')
parser.add_argument("--gravity", type=float, default=9.8, help='Gravity for NCartpole environment')
parser.add_argument("--batch_size", type=int, default=256, help='Batch size for PPO')

args = parser.parse_args()

env_name = args.env


def make_cartpole_env():
    env = CARTerpillarEnv(n_poles=args.n_carts, gravity=args.gravity)
    env = TimeLimit(env, max_episode_steps=500)
    return env

env = make_vec_env(make_cartpole_env, n_envs=8, seed=args.seed)

env_name = f"CARTerpillar{args.gravity}_{args.n_carts}"

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
    model = PPO(args.policy, env, verbose=1, seed=args.seed, ent_coef=entropy_value, n_steps=args.n_steps, n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, clip_range=args.clip_range,  gamma=args.gamma, gae_lambda=args.gae_lambda)
else:
    model = CRPPO(args.policy, env, verbose=1, seed=args.seed, ent_coef=entropy_value, n_steps=args.n_steps, n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, clip_range=args.clip_range,  gamma=args.gamma, gae_lambda=args.gae_lambda)

model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=args.timesteps)