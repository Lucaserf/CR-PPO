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
PYTHON_SCRIPT="main_gymnasium.py"

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

seeds=("0" "1" "2" "3" "4")
entropy_values=("1e-1" "5e-2" "1e-2" "5e-3" "1e-3" "0")

env= "CrafterReward-v1"
policy= "CnnPolicy"

timesteps= "2000000"

for seed in "${seeds[@]}"; do
    for entropy_value in "${entropy_values[@]}"; do
        apptainer exec --nv --bind $WORKDIR:$WORKDIR $CONTAINER_NAME python3 ${PYTHON_SCRIPT} \
        --env $env \
        --seed $seed \
        --entropy_value $entropy_value \
        --timesteps $timesteps \
        --policy $policy \
        --only_entropy &

        apptainer exec --nv --bind $WORKDIR:$WORKDIR $CONTAINER_NAME python3 ${PYTHON_SCRIPT} \
        --env $env \
        --seed $seed \
        --entropy_value $entropy_value \
        --timesteps $timesteps \
        --policy $policy &
    done
done


wait
echo "Training completed."

