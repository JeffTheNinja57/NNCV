#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --job-name=pso-unet-perf
#SBATCH --output=slurm-pso-performance-%j.out

# PSO search trains many short-lived models (pop=5, iter=10, 2 epochs each)
# followed by a full 100-epoch training of the best architecture.
# 12 hours should be sufficient on a single A100.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main_pso_performance.sh
