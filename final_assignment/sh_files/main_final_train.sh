#!/bin/bash
# Final training — fixed architecture, full checkpoint-resume support.
# Edit --depth and --channels below to match the best architecture
# found by your PSO search, or point --arch-file at a JSON file.

wandb login

python3 final_train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "final-unet-v1" \
    --depth 4 \
    --channels 64 128 256 512 512

