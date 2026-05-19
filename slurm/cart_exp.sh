#!/bin/bash
#SBATCH --job-name=CR-PPO
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/%N_%j.log
#SBATCH -e ./slurm/slurm.%N.%j.err
#SBATCH --chdir=/scratch.hpc/luca.serfilippi3/CR-PPO
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.serfilippi@unibo.it

# PYTHON_SCRIPT=$1
PYTHON_SCRIPT="main_CARTerpillar.py"

export APPTAINER_CACHEDIR=/scratch.hpc/luca.serfilippi3

WORKDIR=/scratch.hpc/luca.serfilippi3/CR-PPO
SLURM_DIR=${WORKDIR}/slurm
CONTAINER_NAME="${SLURM_DIR}/exp_container.sif"

# Load tokens from file
# source ${SLURM_DIR}/tokens.txt

# Build container if it doesn't exist yet, otherwise skip
if [ ! -f $CONTAINER_NAME ]; then
    echo "Container not found, building ${CONTAINER_NAME} from ${SLURM_DIR}/exp_container.def ..."
    apptainer build --ignore-fakeroot-command $CONTAINER_NAME ${SLURM_DIR}/exp_container.def
    echo "Container built successfully."
else
    echo "Container ${CONTAINER_NAME} already exists, skipping build."
fi

echo "Running ${PYTHON_SCRIPT} (pretrained=False) inside container in parallel..."

seeds=("3" "4")
entropy_values=("1e-1" "3e-2" "1e-2" "3e-3" "1e-3" "0")
n_carts="6"

env="CartPole-v1"
policy="MlpPolicy"

timesteps="4000000"



for seed in "${seeds[@]}"; do
    for entropy_value in "${entropy_values[@]}"; do
        apptainer exec --nv --bind $WORKDIR:$WORKDIR $CONTAINER_NAME python3 ${PYTHON_SCRIPT} \
        --env $env \
        --seed $seed \
        --entropy_value $entropy_value \
        --timesteps $timesteps \
        --policy $policy \
        --n_carts $n_carts \
        --only_entropy &
    done
    
    for entropy_value in "${entropy_values[@]}"; do
        apptainer exec --nv --bind $WORKDIR:$WORKDIR $CONTAINER_NAME python3 ${PYTHON_SCRIPT} \
        --env $env \
        --seed $seed \
        --entropy_value $entropy_value \
        --timesteps $timesteps \
        --policy $policy \
        --n_carts $n_carts &
    done
done


wait
echo "Training completed."

