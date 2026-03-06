#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --job-name=pso-unet-eff
#SBATCH --output=slurm-pso-efficiency-%j.out
#SBATCH --exclusive

# Efficiency mode searches over smaller architectures (max_depth=4,
# max_channels=256), so PSO evaluations are faster than performance mode.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main_pso_efficiency.sh
