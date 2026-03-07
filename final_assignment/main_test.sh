#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# PSO U-Net — Quick Smoke Test
# Runs a minimal PSO search (1 iteration, 1 particle, 1 epoch per eval)
# followed by a 1-epoch full training of the "best" architecture.
# Purpose: verify container, data, GPU, wandb, and code work end-to-end
# before submitting the real efficiency / performance jobs.
# ──────────────────────────────────────────────────────────────────────

# Export WANDB_API_KEY from .env for non-interactive login
if [ -f .env ]; then
    export WANDB_API_KEY=$(grep '^WANDB_API_KEY' .env | cut -d= -f2)
fi

wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 1 \
    --lr 0.001 \
    --num-workers 4 \
    --seed 42 \
    --experiment-id "pso-unet-test" \
    --mode efficiency \
    --pso-iterations 1 \
    --pso-population 1 \
    --pso-epochs 1 \
    --cg 0.5 \
    --lambda-efficiency 0.3 \
    --max-params 5000000 \
    --max-depth 3 \
    --max-channels 128 \
    --pso-batch-size 8 \
    --full-training-epochs 1

