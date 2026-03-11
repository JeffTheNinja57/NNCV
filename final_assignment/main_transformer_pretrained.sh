wandb login

python3 train_transformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 5 \
    --lr 0.00001 \
    --patch-size 8 \
    --pretrained \
    --num-workers 1 \
    --seed 42 \
    --experiment-id "vit-dino-finetune"
