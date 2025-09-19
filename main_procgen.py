import numpy as np

np.bool = np.bool_  # workaround to make all packages working with the same numpy version
import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse
from cdpo_baselines import CDPO


def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, is_test_worker=False,
             log_dir='/tmp/procgen', comm=None, ent_coef=0.01, only_entropy=True):
    learning_rate = 5e-4
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    logger.info("training")

    if only_entropy:
        ppo2.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=0,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    else:
        CDPO.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=0,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )


def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000)
    parser.add_argument('--entropy_value', type=str, default='1e-1', help='Entropy value')
    parser.add_argument('--only_entropy', action='store_true')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    ent_type = 'entropy' if args.only_entropy else 'complexity'

    # Set the random seed
    tf.set_random_seed(args.start_level)

    train_fn(args.env_name,
             args.num_envs,
             args.distribution_mode,
             args.num_levels,
             args.start_level,
             args.timesteps_per_proc,
             is_test_worker=is_test_worker,
             log_dir='/CDPO/logs/' + args.env_name + '_' + str(args.start_level) + '_' + args.entropy_value + '_' + ent_type + '/',
             comm=comm,
             ent_coef=float(args.entropy_value),
             only_entropy=args.only_entropy)


if __name__ == '__main__':
    main()
