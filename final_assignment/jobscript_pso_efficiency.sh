#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --job-name=pso-unet-eff
#SBATCH --output=slurm-pso-efficiency-%j.out

# Efficiency mode searches over smaller architectures (max_depth=4,
# max_channels=256), so PSO evaluations are faster than performance mode.
# 8 hours should be sufficient on a single A100.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main_pso_efficiency.sh
