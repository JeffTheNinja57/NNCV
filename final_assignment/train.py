"""
Training script for a PSO-optimised PyTorch U-Net on Cityscapes.

Execution flow:
1. Parse arguments (including PSO and benchmark-mode settings).
2. Set up data loaders and wandb.
3. Run the PSO architecture search (``PSOUNet.search``) to find the
   best U-Net configuration.
4. Log the found architecture and fitness history to wandb.
5. Fully train the best architecture using the existing training loop
   (AdamW, CrossEntropyLoss, checkpoint saving — all unchanged).

Two benchmark modes are supported via ``--mode``:
- **performance** — maximise segmentation accuracy (fitness = −val_loss).
- **efficiency** — best accuracy at lowest parameter count
  (fitness = val_mIoU − λ·n_params/max_params).
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
)

from model import Model
from pso_unet import PSOUNet


# ---- Cityscapes label remapping (unchanged) --------------------------------
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
        "Training script for a PSO-optimised PyTorch U-Net model"
    )

    # Existing arguments (unchanged defaults)
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes",
                        help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (full training)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="pso-unet",
                        help="Experiment ID for Weights & Biases")

    # ---- PSO / benchmark-mode arguments -----------------------------------
    parser.add_argument("--mode", type=str, required=True,
                        choices=["efficiency", "performance"],
                        help="Benchmark mode: 'efficiency' or 'performance'")
    parser.add_argument("--pso-iterations", type=int, default=10,
                        help="Number of PSO iterations")
    parser.add_argument("--pso-population", type=int, default=5,
                        help="PSO swarm size")
    parser.add_argument("--pso-epochs", type=int, default=2,
                        help="Short training epochs per particle evaluation")
    parser.add_argument("--cg", type=float, default=0.5,
                        help="PSO global best pull weight (Cg)")
    parser.add_argument("--lambda-efficiency", type=float, default=0.3,
                        help="Parameter penalty weight (efficiency mode)")
    parser.add_argument("--max-params", type=int, default=5_000_000,
                        help="Parameter count normaliser (efficiency mode)")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Max encoder depth PSO can explore")
    parser.add_argument("--max-channels", type=int, default=512,
                        help="Max channels at any encoder level")
    parser.add_argument("--full-training-epochs", type=int, default=None,
                        help="Epochs for final gBest training "
                             "(defaults to --epochs)")

    return parser


# ---- Main ------------------------------------------------------------------
def main(args):
    # Resolve full-training epochs
    full_epochs = (args.full_training_epochs
                   if args.full_training_epochs is not None
                   else args.epochs)

    # Initialize wandb
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    # Output directory
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data loaders (unchanged) -----------------------------------------
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine",
        target_type="semantic", transforms=transform,
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine",
        target_type="semantic", transforms=transform,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # ====================================================================
    # Phase 1: PSO Architecture Search
    # ====================================================================
    pso = PSOUNet(
        pop_size=args.pso_population,
        n_iter=args.pso_iterations,
        pso_epochs=args.pso_epochs,
        lr=args.lr,
        min_depth=1,
        max_depth=args.max_depth,
        min_channels=16,
        max_channels=args.max_channels,
        in_channels=3,
        n_classes=19,
        mode=args.mode,
        Cg=args.cg,
        lambda_=args.lambda_efficiency,
        max_params=args.max_params,
    )

    best_arch = pso.search(train_dataloader, valid_dataloader, device)

    # Log PSO results to wandb
    wandb.log({
        "pso/gBest_arch": str(best_arch),
        "pso/gBest_depth": best_arch["depth"],
        "pso/gBest_channels": best_arch["channels"],
        "pso/gBest_fitness": pso.gBest_fitness,
    })
    # Log fitness history as a wandb table
    for step, fitness in enumerate(pso.get_fitness_history()):
        wandb.log({"pso/fitness_history": fitness, "pso_step": step})

    # ====================================================================
    # Phase 2: Full Training of gBest Architecture
    # ====================================================================
    print(f"\nFull training of gBest architecture for {full_epochs} epochs...")
    print(f"Architecture: {best_arch}")

    model = Model(
        in_channels=3,
        n_classes=19,
        arch=best_arch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # ---- Training loop (unchanged internals) ------------------------------
    best_valid_loss = float('inf')
    current_best_model_path = None

    for epoch in range(full_epochs):
        print(f"Epoch {epoch + 1:04}/{full_epochs:04}")

        # Training
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
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # Validation
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

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)
                    predictions = predictions.unsqueeze(1)
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
            wandb.log({
                "valid_loss": valid_loss,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

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

    # Save final model
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
