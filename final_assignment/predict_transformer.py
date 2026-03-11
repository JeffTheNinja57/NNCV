"""
Prediction script for the ViT segmentation model.

Loads a trained checkpoint, runs inference on all PNG images in IMAGE_DIR,
and saves the predicted segmentation masks to OUTPUT_DIR.

Works for both Peak Performance and Robustness benchmark submissions —
just submit to the corresponding challenge server endpoint.

For Docker submission, copy this file as predict.py and transformer.py as
model.py (or update the Dockerfile COPY lines).
"""

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from transformer import Model

# Fixed paths inside the participant container — do NOT change these.
# For local testing you can override them.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"


def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize, normalise, and add a batch dimension."""
    transform = Compose([
        ToImage(),
        Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # match training
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """Argmax → resize back to original resolution → numpy array."""
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # remove batch & channel dims
    return prediction_numpy


def main():
    # Device — works on CUDA (challenge server), MPS (Mac), or CPU (Docker)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load checkpoint (contains state_dict + patch_size)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    patch_size = checkpoint["patch_size"]

    model = Model(patch_size=patch_size)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Save predicted mask
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
