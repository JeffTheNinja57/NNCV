"""
Final training script — trains a fixed U-Net architecture with full
checkpoint-resume support.  No PSO search is performed.

Use this **after** you have found a good architecture with train.py,
or pass the architecture directly via --depth and --channels.

Resume behaviour
----------------
Every epoch the full training state is saved to
``<output_dir>/training_checkpoint.pth``.  If the script is launched
again with the same --experiment-id it will automatically detect the
checkpoint and continue from where it left off.  This makes it safe
to re-submit the Slurm job if it times out.

Example
-------
python3 final_train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 --epochs 200 --lr 0.001 \
    --num-workers 10 --seed 42 \
    --experiment-id "final-unet-v1" \
    --depth 4 --channels 64 128 256 512 512
"""

import os
import json
from argparse import ArgumentParser

import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


# ---- Cityscapes label remapping --------------------------------------------
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


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


# ---- Argument parser -------------------------------------------------------
def get_args_parser():
    parser = ArgumentParser(
        "Final training script — fixed architecture with checkpoint resume"
    )
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="final-unet")

    # Architecture specification (manual)
    parser.add_argument("--depth", type=int, default=4,
                        help="Encoder depth")
    parser.add_argument("--channels", type=int, nargs="+",
                        default=[64, 128, 256, 512, 512],
                        help="Channel counts per encoder level "
                             "(length must be depth + 1)")

    # Alternatively, load the architecture from a PSO results JSON file
    parser.add_argument("--arch-file", type=str, default=None,
                        help="Path to a JSON file with keys 'depth' and "
                             "'channels'. Overrides --depth/--channels.")

    return parser


# ---- Main ------------------------------------------------------------------
def main(args):
    # ---- Resolve architecture ---------------------------------------------
    if args.arch_file is not None:
        with open(args.arch_file) as f:
            arch = json.load(f)
        print(f"Loaded architecture from {args.arch_file}: {arch}")
    else:
        arch = {"depth": args.depth, "channels": args.channels}

    assert len(arch["channels"]) == arch["depth"] + 1, (
        f"channels length ({len(arch['channels'])}) must equal "
        f"depth + 1 ({arch['depth'] + 1})"
    )

    # ---- Output / checkpoint paths ----------------------------------------
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "training_checkpoint.pth")

    # ---- Reproducibility --------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ---- Data loaders -----------------------------------------------------
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

    # ---- Model, optimizer, scheduler --------------------------------------
    model = Model(in_channels=3, n_classes=19, arch=arch).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: {arch}")
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Resume from checkpoint if it exists ------------------------------
    start_epoch = 0
    best_valid_loss = float("inf")
    current_best_model_path = None
    wandb_id = None

    if os.path.exists(checkpoint_path):
        print(f"\n>>> Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_valid_loss = ckpt["best_valid_loss"]
        current_best_model_path = ckpt.get("best_model_path")
        wandb_id = ckpt.get("wandb_id")
        print(f"    Resuming at epoch {start_epoch}, "
              f"best_valid_loss={best_valid_loss:.6f}\n")

    # ---- Initialize wandb (resume-aware) ----------------------------------
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        id=wandb_id,
        resume="allow",
        config={**vars(args), "arch": arch, "n_params": n_params},
    )

    wandb.log({"model/n_params": n_params})

    # ---- Training loop ----------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        # -- Train ----------------------------------------------------------
        model.train()
        train_losses = []
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        avg_train_loss = sum(train_losses) / len(train_losses)

        # -- Validate -------------------------------------------------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                # Log visual predictions for the first batch
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels_vis = labels.unsqueeze(1)
                    pred_color = convert_train_id_to_color(predictions)
                    lab_color = convert_train_id_to_color(labels_vis)
                    wandb.log({
                        "predictions": [wandb.Image(
                            make_grid(pred_color.cpu(), nrow=8)
                            .permute(1, 2, 0).numpy()
                        )],
                        "labels": [wandb.Image(
                            make_grid(lab_color.cpu(), nrow=8)
                            .permute(1, 2, 0).numpy()
                        )],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

        valid_loss = sum(val_losses) / len(val_losses)
        wandb.log({
            "valid_loss": valid_loss,
        }, step=(epoch + 1) * len(train_dataloader) - 1)

        print(f"  train_loss={avg_train_loss:.4f}  "
              f"valid_loss={valid_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # -- Save best model ------------------------------------------------
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if current_best_model_path and os.path.exists(current_best_model_path):
                os.remove(current_best_model_path)
            current_best_model_path = os.path.join(
                output_dir,
                f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
            )
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save(state, current_best_model_path)
            print(f"  ★ New best model saved → {current_best_model_path}")

        # -- Save full training checkpoint (every epoch) --------------------
        state = (model.module.state_dict()
                 if isinstance(model, nn.DataParallel)
                 else model.state_dict())
        torch.save({
            "epoch": epoch,
            "model_state_dict": state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_valid_loss": best_valid_loss,
            "best_model_path": current_best_model_path,
            "arch": arch,
            "wandb_id": wandb_id,
        }, checkpoint_path)

        # Step the scheduler
        scheduler.step()

    # ---- Save final model -------------------------------------------------
    print("\nTraining complete!")
    final_state = (model.module.state_dict()
                   if isinstance(model, nn.DataParallel)
                   else model.state_dict())
    torch.save(
        final_state,
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
        ),
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

