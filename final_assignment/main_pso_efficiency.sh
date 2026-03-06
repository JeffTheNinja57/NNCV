#!/bin/bash
# PSO U-Net Architecture Search — Efficiency Mode
# Fitness = val_mIoU - lambda * (n_params / max_params)
# Favours shallower, narrower architectures with good accuracy.

wandb login
python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 1 \
    --seed 42 \
    --experiment-id "pso-unet-efficiency" \
    --mode efficiency \
    --pso-iterations 10 \
    --pso-population 20 \
    --pso-epochs 1 \
    --cg 0.5 \
    --lambda-efficiency 0.3 \
    --max-params 5000000 \
    --max-depth 4 \
    --max-channels 256 \
    --full-training-epochs 100\
