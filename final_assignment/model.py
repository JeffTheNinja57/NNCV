from typing import Dict, List, Optional

import torch
import torch.nn as nn


# Default architecture matching the original hardcoded U-Net.
# predict.py calls Model() with no args, so this must always produce
# the same network as the original implementation.
DEFAULT_ARCH: Dict[str, object] = {
    "depth": 4,
    "channels": [64, 128, 256, 512, 512],
}


class Model(nn.Module):
    """
    A dynamic U-Net architecture for image segmentation.

    The encoder/decoder structure is built programmatically from an
    architecture config dict with keys ``depth`` and ``channels``.
    The decoder is always the exact mirror of the encoder — its channel
    sizes are derived from the encoder channels, and skip connections
    between mirrored levels are always preserved.

    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks
    for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    The CodaLab server requires the model class to be named "Model"
    and will use the default constructor values, so ``DEFAULT_ARCH``
    is used when no ``arch`` argument is provided.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 19,
        arch: Optional[Dict] = None,
    ):
        """
        Args:
            in_channels: Number of input channels. Default is 3 for RGB.
            n_classes: Number of output classes. Default is 19 for Cityscapes.
            arch: Architecture config dict with keys ``depth`` (int) and
                ``channels`` (list of int). ``channels`` has length
                ``depth + 1``, giving the output channel count at each
                encoder level (inc, down1, …, down_depth).  If *None*,
                ``DEFAULT_ARCH`` is used.
        """
        super().__init__()

        if arch is None:
            arch = DEFAULT_ARCH
        depth: int = arch["depth"]
        channels: List[int] = list(arch["channels"])
        assert len(channels) == depth + 1, (
            f"channels length ({len(channels)}) must equal depth + 1 ({depth + 1})"
        )

        self.in_channels = in_channels
        self.depth = depth
        self.channels = channels

        # ---- Encoder --------------------------------------------------------
        self.inc = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(channels[i], channels[i + 1]))

        # ---- Decoder (derived from encoder — symmetry enforced) -------------
        # Up block i receives:
        #   - upsampled features from the previous decoder stage (or bottleneck)
        #   - skip connection from encoder stage (depth - 1 - i)
        # Its output channels mirror the encoder:
        #   out_ch = channels[max(0, depth - 2 - i)]
        self.ups = nn.ModuleList()
        prev_ch = channels[depth]  # bottleneck output channels
        for i in range(depth):
            skip_ch = channels[depth - 1 - i]
            out_ch = channels[max(0, depth - 2 - i)]
            self.ups.append(Up(prev_ch + skip_ch, out_ch))
            prev_ch = out_ch

        self.outc = OutConv(prev_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            Logits tensor of shape ``(batch, n_classes, H, W)``.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        # Encoder — store each level's output for skip connections
        skips: List[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Decoder — pop skips in reverse (skip the bottleneck itself)
        x = skips.pop()  # bottleneck
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
