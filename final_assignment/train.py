"""Baseline U-Net training on Cityscapes (5LSM0 final assignment).

Trains the canonical depth-4 Ronneberger U-Net (provided by the course staff)
with AdamW + CrossEntropyLoss, logging to Weights & Biases and saving the
best-validation checkpoint.
"""
import os
from argparse import ArgumentParser

import numpy as np
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

from model import Model


# Cityscapes ships 34 raw label IDs but only 19 are evaluated; remap so that
# unused IDs collapse to 255 (ignored by the loss).
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


# Per-class colours for qualitative wandb visualisations.
train_id_to_color = {
    cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255
}
train_id_to_color[255] = (0, 0, 0)


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


def get_args_parser():
    parser = ArgumentParser("Baseline U-Net training on Cityscapes")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes",
                        help="Path to the Cityscapes dataset root")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="unet-training",
                        help="Run name for Weights & Biases")
    return parser


def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Deterministic seeds so re-runs of the baseline are comparable.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # 256x256 keeps memory low enough for batch_size=64 on a single A100.
    img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # NEAREST interpolation — bilinear would invent non-existent class IDs.
    target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine",
        target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine",
        target_type="semantic",
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

    # Model() with no args uses DEFAULT_ARCH (depth=4, [64,128,256,512,512]) —
    # the canonical course-provided baseline; predict.py / CodaLab rely on it.
    model = Model().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ignore_index=255 skips the unlabelled / void pixels produced by the
    # train-id remapping above.
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_valid_loss = float("inf")
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

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

                # Log a single batch of qualitative predictions per epoch.
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels_vis = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels_vis = convert_train_id_to_color(labels_vis)

                    predictions_img = make_grid(
                        predictions.cpu(), nrow=8,
                    ).permute(1, 2, 0).numpy()
                    labels_img = make_grid(
                        labels_vis.cpu(), nrow=8,
                    ).permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss},
                      step=(epoch + 1) * len(train_dataloader) - 1)

            # Keep only the single best checkpoint to bound disk usage.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt",
                )
                torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")

    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt",
        ),
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
