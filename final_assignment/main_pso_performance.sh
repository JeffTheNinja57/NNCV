#!/bin/bash
# PSO U-Net Architecture Search — Performance Mode
# Fitness = -val_loss (best segmentation accuracy regardless of model size)

wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "pso-unet-performance" \
    --mode performance \
    --pso-iterations 10 \
    --pso-population 20 \
    --pso-epochs 20 \
    --cg 0.5 \
    --max-depth 5 \
    --max-channels 1024 \
    --full-training-epochs 100
