wandb login

python3 train_transformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0001 \
    --patch-size 16 \
    --num-workers 1 \
    --seed 42 \
    --experiment-id "vit-scratch"
