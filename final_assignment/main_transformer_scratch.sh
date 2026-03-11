wandb login

python3 train_transformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 5 \
    --lr 0.001 \
    --patch-size 16 \
    --num-workers 1 \
    --seed 42 \
    --experiment-id "vit-scratch"
