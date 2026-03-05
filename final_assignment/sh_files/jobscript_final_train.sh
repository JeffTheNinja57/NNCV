#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --job-name=final-unet
#SBATCH --output=slurm-final-train-%j.out

# Full training of a fixed architecture with checkpoint-resume support.
# If the job times out, just re-submit — it picks up where it left off.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main_final_train.sh

