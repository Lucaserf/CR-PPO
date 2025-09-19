# CDPO - Curiosity-Driven Policy Optimization

## In Summary

A brief summary of the method.

## Execution Instruction

This repo contains an implementation of CDPO based on `stable-baselines3` and `pytorch`, as well as one based on `baselines` and `tensorflow` 1.x, and all the code necessary to reproduce the results reported in the paper.

### Gym environments

### CoinRun environment

To train a CDPO agent on CoinRun, you can use the Dockerfile `Dockerfile.procgen` to build a Docker container in which running the code. Alternatively, you can set up a virtual environment with Python >=3.6,<=3.8 by executing:

```
python -m pip install -r requirements_procgen.txt
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
```

Then, you can launch the training with:

```
python3 main_procgen.py --env_name coinrun --num_envs 256 --distribution_mode hard --num_levels 500 --start_level 0 --timesteps_per_proc 50000000 --entropy_value 1e-2
```

and play with the `--entropy_value` parameter, which corresponds to the scaling factor for either entropy or complexity regularization. To rollback from CDPO to PPO, just pass the argument `--only_entropy`.


Please note that, following the original paper, we used parallel training through MPI:

```
mpiexec -np 2 python3 main_procgen.py --env_name coinrun --num_envs 256 --distribution_mode hard --num_levels 500 --start_level 0 --test_worker_interval 2 --timesteps_per_proc 50000000 --entropy_value 1e-2
```

### CARTerpillar enironment
