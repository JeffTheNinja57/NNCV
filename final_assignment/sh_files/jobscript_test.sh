#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --job-name=pso-unet-test
#SBATCH --output=slurm-pso-test-%j.out

# Quick smoke test: 1 PSO iteration, 1 particle, 1 epoch.
# Should finish in a few minutes — enough to verify that the container,
# data loading, GPU, wandb logging, and the full code path all work.

srun apptainer exec --nv --env-file .env container.sif /bin/bash main_test.sh

