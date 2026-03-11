"""
Training script for the ViT segmentation model on Cityscapes.

Based on the existing train.py but uses the transformer Model instead of U-Net.
Supports two modes:
  - From scratch:  python train_transformer.py --patch-size 16
  - With DINO:     python train_transformer.py --patch-size 8 --pretrained

Hyperparameters are set via command-line args (called from main_transformer_*.sh).
Progress is logged to Weights & Biases since training runs on a remote cluster.
"""

import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
)

from transformer import Model


# ---------------------------------------------------------------------------
# Cityscapes label helpers (same as train.py)
# ---------------------------------------------------------------------------

# Map original Cityscapes class IDs → train IDs (19 classes + 255 = void)
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Map train IDs → RGB colours for visualisation in W&B
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # black for void / ignored

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = ArgumentParser("Training script for ViT segmentation model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes",
                        help="Path to the Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of data-loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="vit-training",
                        help="Experiment name for Weights & Biases")

    # --- Transformer-specific args ---
    parser.add_argument("--patch-size", type=int, default=8,
                        help="ViT patch size (8 for DINO pretrained, 16 for scratch)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Load DINO self-supervised pretrained weights into the backbone")
    return parser


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

DINO_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"


def main(args):
    # Initialise W&B
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    # Create checkpoint directory
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device — works on CUDA (cluster), MPS (Mac), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- Data transforms (same as train.py) -----
    img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    # ----- Datasets & loaders -----
    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # ----- Model -----
    model = Model(
        in_channels=3,
        n_classes=19,
        patch_size=args.patch_size,
    ).to(device)

    # Optionally load DINO pretrained weights into the backbone
    if args.pretrained:
        print("Loading DINO pretrained weights …")
        state_dict = torch.hub.load_state_dict_from_url(
            url=DINO_URL, map_location=device,
        )
        model.backbone.load_state_dict(state_dict, strict=True)
        print("DINO weights loaded successfully.")

    # ----- Loss, optimiser -----
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # ----- Training loop -----
    best_valid_loss = float("inf")
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        # ---- Train ----
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # ---- Validate ----
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                # Log visual predictions for the first batch
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels_vis = labels.unsqueeze(1)

                    pred_color = convert_train_id_to_color(predictions)
                    label_color = convert_train_id_to_color(labels_vis)

                    pred_img = make_grid(pred_color.cpu(), nrow=8).permute(1, 2, 0).numpy()
                    label_img = make_grid(label_color.cpu(), nrow=8).permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(pred_img)],
                        "labels": [wandb.Image(label_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)
            print(f"  val_loss = {valid_loss:.4f}")
            wandb.log({"valid_loss": valid_loss},
                       step=(epoch + 1) * len(train_dataloader) - 1)

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
                )
                torch.save({
                    "state_dict": model.state_dict(),
                    "patch_size": args.patch_size,
                }, current_best_model_path)

    print("Training complete!")

    # Save final model
    torch.save({
        "state_dict": model.state_dict(),
        "patch_size": args.patch_size,
    }, os.path.join(
        output_dir,
        f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
    ))
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
