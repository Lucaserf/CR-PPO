import torch as th
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import sys
sys.modules['gym'] = gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack
#import crafter


import ale_py

# 1. Tell gymnasium to discover and register all ale-py environments internally
gym.register_envs(ale_py)

# 2. Manually map your exact string to the newly registered official environment
gym.register(
    id="AsteroidsNoFrameskip-v4",
    entry_point="ale_py.env:AtariEnv",  # Modern entry point
    kwargs={
        "game": "asteroids", 
        "obs_type": "rgb", 
        "frameskip": 1, 
        "repeat_action_probability": 0.0
    },
)
class CrafterGymnasiumEnv(gym.Env):
    def __init__(self, reward=True, seed=None):
        self.reward = reward
        self.env = crafter.Env(reward=reward, seed=seed)
        
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = Discrete(17)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env = crafter.Env(reward=self.reward, seed=seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

def make_crafter_env():
    return CrafterGymnasiumEnv(reward=True)

# from stable_baselines3.common.atari_wrappers import AtariWrapper
from crppo_stablebaselines.CRPPO import CRPPO
import argparse

parser = argparse.ArgumentParser(description='CRPPO')
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
elif env_name in ["CrafterReward-v1"]:
    env = make_vec_env(make_crafter_env, n_envs=8, seed=args.seed)
    env = VecFrameStack(env, n_stack=4)
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
    model = CRPPO(args.policy, env, verbose=1, seed=args.seed, ent_coef=entropy_value, n_steps=args.n_steps, n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, clip_range=args.clip_range, vf_coef=0.5, gamma=args.gamma, gae_lambda=args.gae_lambda)


model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=args.timesteps)