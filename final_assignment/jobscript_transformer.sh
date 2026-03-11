#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00

# Uncomment ONE of the lines below to choose training mode:
srun apptainer exec --nv --env-file .env container.sif /bin/bash main_transformer_scratch.sh
# srun apptainer exec --nv --env-file .env container.sif /bin/bash main_transformer_pretrained.sh
