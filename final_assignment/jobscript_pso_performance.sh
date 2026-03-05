#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=4
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --job-name=pso-unet-perf
#SBATCH --output=slurm-pso-performance-%j.out
#SBATCH --exclusive

# PSO search trains many short-lived models (pop=10, iter=10, 20 epochs each)
# followed by a full 100-epoch training of the best architecture.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main_pso_performance.sh
