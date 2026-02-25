# CRPPO - Curiosity-Driven Policy Optimization

## In Summary

Several policy gradient methods, such as Proximal Policy Optimization (PPO), often include entropy regularization. This can help with exploration by encouraging stochastic policies and preventing deterministic behavior, and in certain scenarios, it can also facilitate the optimization process. However, especially for PPO, its real utility is disputed, and finding the correct scaling factor for the entropy loss is not trivial; in addition, maximizing entropy pushes the policy towards a uniform random distribution regardless of the current nature of the policy and the actual need for exploration, which might result in a less efficient learning strategy.

In this work, we propose to replace entropy with a new regularization term based on complexity, defined as the product of Shannon entropy and disequilibrium. More specifically, we define Complexity-Driven Policy Optimization (CRPPO), a new learning algorithm based on PPO that replaces its entropy bonus with our complexity term. By scaling the policy entropy with its disequilibrium, CRPPO is capable of promoting divergence when the policy becomes too deterministic, while also promoting convergence when the policy is too random.

## Execution Instruction

This repo contains an implementation of CRPPO based on `stable-baselines3` and `pytorch`, as well as one based on `baselines` and `tensorflow` 1.x, and all the code necessary to reproduce the results reported in the paper. In particular, the experiments concerned three types of environments: classic and Atari gym environments; one ProcGen environment, i.e., CoinRun; and a brand new environment, namely CARTerpillar, which extends CartPole to multiple carts and provides a simple and scalable way to evaluate agents under different levels of difficulty. Below you can find the instructions to repeat all our experiments.

### Gym environments

To run the experiments on gym environments, you can set up a virtual environment with Python >=3.10 and install the required packages by executing:

```
python -m pip install -r requirements_stablebaselines.txt
```

Then, you can launch the training with:

```
python3 main_gymnasium.py --env <env_name> --entropy_value <entropy_value> --seed <seed> --timesteps <timesteps> --policy <policy> --lr <learning_rate> --clip_range <clip_range> --epochs <epochs> --n_steps <n_steps> --gamma <gamma> --gae_lambda <gae_lambda> --batch_size <batch_size>
```

For example, to train a CRPPO agent on `CartPole-v1` you can run:

```
python3 main_gymnasium.py --env CartPole-v1 --entropy_value 1e-1 --seed 0 --timesteps 1000000 --policy MlpPolicy --lr 2.5e-4 --clip_range 0.1 --epochs 4 --n_steps 128 --gamma 0.99 --gae_lambda 0.95 --batch_size 128
```

To train on an Atari environment like `AirRaidNoFrameskip-v4`:

```
python3 main_gymnasium.py --env AirRaidNoFrameskip-v4 --policy CnnPolicy --entropy_value 1e-2 --seed 0 --timesteps 1000000 --lr 2.5e-4 --clip_range 0.1 --epochs 4 --n_steps 128 --gamma 0.99 --gae_lambda 0.95 --batch_size 256
```


You can play with the `--entropy_value` parameter, which corresponds to the scaling factor for either entropy or complexity regularization. To switch from CRPPO to PPO, just pass the argument `--only_entropy`., you can use the Dockerfile `Dockerfile.procgen` to build a Docker container in which running the code. Alternatively, you can set up a virtual environment with Python >=3.6,<=3.8 by executing:

```
python -m pip install -r requirements_procgen.txt
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
```

Then, you can launch the training with:

```
python3 main_procgen.py --env_name coinrun --num_envs 256 --distribution_mode hard --num_levels 500 --start_level 0 --timesteps_per_proc 50000000 --entropy_value 1e-2
```

and play with the `--entropy_value` parameter, which corresponds to the scaling factor for either entropy or complexity regularization. To rollback from CRPPO to PPO, just pass the argument `--only_entropy`.


Please note that, following the original paper, we used parallel training through MPI:

```
mpiexec -np 2 python3 main_procgen.py --env_name coinrun --num_envs 256 --distribution_mode hard --num_levels 500 --start_level 0 --test_worker_interval 2 --timesteps_per_proc 50000000 --entropy_value 1e-2
```

### CARTerpillar enironment

To run the experiments on the CARTerpillar environment, you can set up a virtual environment with Python >=3.10 and install the required packages by executing:

```
python -m pip install -r requirements_stablebaselines.txt
```

Then, you can launch the training with:

```
python3 main_CARTerpillar.py --n_carts <n_carts> --gravity <gravity> --entropy_value <entropy_value> --seed <seed> --timesteps <timesteps> --policy <policy> --lr <learning_rate> --clip_range <clip_range> --epochs <epochs> --n_steps <n_steps> --gamma <gamma> --gae_lambda <gae_lambda> --batch_size <batch_size>
```

For example, to train a CRPPO agent on `CARTerpillar` with 2 poles and default gravity you can run:

```
python3 main_CARTerpillar.py --n_carts 2 --entropy_value 1e-1 --seed 0 --timesteps 100000
```

You can play with the `--entropy_value` parameter, which corresponds to the scaling factor for either entropy or complexity regularization. To switch from CRPPO to PPO, just pass the argument `--only_entropy`. The `--n_carts` and `--gravity` arguments allow you to change the difficulty of the environment.