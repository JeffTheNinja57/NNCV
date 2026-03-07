"""
UNetParticle — a single particle in the PSO search over U-Net architectures.

Each particle represents one U-Net architecture parameterised by a structured
dict ``{"depth": int, "channels": [int, ...]}``.  The decoder is always the
exact mirror of the encoder (enforced in ``update_position``), so only the
encoder description is stored.

The PSO velocity logic is adapted from the flat layer-list approach in the
original thesis PSO-CNN codebase (``utils.py``).  Instead of operating on a
variable-length list of heterogeneous layer dicts, velocity here operates on
two components:

* **depth** — an integer selected probabilistically from gBest or pBest.
* **channels** — a per-level value selected probabilistically from gBest or
  pBest, with a small random perturbation for exploration.

This is a *discrete crossover* PSO: the velocity is not accumulated across
iterations but recomputed each time from the current pBest and gBest.
"""
from __future__ import annotations

import copy
import gc
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import Model


# Label remapping needed for loss / mIoU during PSO evaluation.
# Imported lazily in compute_fitness to avoid circular deps with train.py.
_convert_to_train_id = None


def _get_convert_fn():
    """Lazy import of the label-remapping function from train.py."""
    global _convert_to_train_id
    if _convert_to_train_id is None:
        from train import convert_to_train_id
        _convert_to_train_id = convert_to_train_id
    return _convert_to_train_id


class UNetParticle:
    """One particle in the PSO swarm — represents a single U-Net architecture.

    Attributes:
        arch: Architecture dict ``{"depth": int, "channels": [int, ...]}``.
        velocity: Target architecture produced by the last velocity update.
        pBest_arch: Personal best architecture dict.
        pBest_fitness: Fitness of the personal best.
        fitness: Fitness of the current architecture (after last evaluation).
    """

    def __init__(
        self,
        min_depth: int,
        max_depth: int,
        min_channels: int,
        max_channels: int,
        in_channels: int = 3,
        n_classes: int = 19,
        channel_step: int = 16,
    ):
        """
        Args:
            min_depth: Minimum number of encoder Down blocks.
            max_depth: Maximum number of encoder Down blocks.
            min_channels: Minimum channel count at any encoder level.
            max_channels: Maximum channel count at any encoder level.
            in_channels: Number of input image channels (e.g. 3 for RGB).
            n_classes: Number of segmentation classes.
            channel_step: Channels are rounded to multiples of this value.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.channel_step = channel_step

        # Random initial architecture
        self.arch: Dict = self._random_arch()
        # Velocity starts as a copy of the initial arch (i.e. "keep")
        self.velocity: Dict = copy.deepcopy(self.arch)

        # Personal best = initial arch (no fitness yet)
        self.pBest_arch: Dict = copy.deepcopy(self.arch)
        self.pBest_fitness: Optional[float] = None
        self.fitness: Optional[float] = None

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    def _random_arch(self) -> Dict:
        """Generate a random valid U-Net architecture.

        Produces a non-decreasing channel list so that deeper levels have
        at least as many channels as shallower ones, which is the standard
        U-Net convention and helps gradient flow.
        """
        depth = int(np.random.randint(self.min_depth, self.max_depth + 1))
        channels: List[int] = []
        prev = self.min_channels
        for _ in range(depth + 1):
            lo = max(self.min_channels, prev)
            hi = self.max_channels
            if lo > hi:
                lo = hi
            raw = np.random.randint(lo // self.channel_step,
                                    hi // self.channel_step + 1)
            ch = int(raw * self.channel_step)
            ch = max(self.min_channels, min(self.max_channels, ch))
            channels.append(ch)
            prev = ch
        return {"depth": depth, "channels": channels}

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def build_model(self) -> Model:
        """Construct a PyTorch U-Net from the current architecture dict."""
        return Model(
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            arch=self.arch,
        )

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def compute_fitness(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        mode: str,
        epochs: int,
        lr: float,
        lambda_: float = 0.3,
        max_params: int = 5_000_000,
    ) -> float:
        """Train the architecture for a few epochs and compute fitness.

        Args:
            train_loader: Training data loader (Cityscapes).
            val_loader: Validation data loader (Cityscapes).
            device: CUDA / CPU device.
            mode: ``"performance"`` or ``"efficiency"``.
            epochs: Number of short training epochs for this evaluation.
            lr: Learning rate for AdamW.
            lambda_: Parameter penalty weight (efficiency mode only).
            max_params: Parameter count normaliser (efficiency mode only).

        Returns:
            Fitness score (higher is better for both modes).
        """
        convert = _get_convert_fn()
        model = self.build_model().to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = AdamW(model.parameters(), lr=lr)

        # --- Short training ---
        model.train()
        for _ in range(epochs):
            for images, labels in train_loader:
                labels = convert(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Free training-loop GPU tensors before evaluation — Python keeps
        # loop variables alive in function scope, so they would still be
        # live when empty_cache() is called at the end if not deleted here.
        del images, labels, outputs, loss
        torch.cuda.empty_cache()

        # --- Evaluation ---
        if mode == "performance":
            val_loss = self._compute_val_loss(model, val_loader, device,
                                              criterion, convert)
            # Negate so PSO can maximise (lower loss → higher fitness)
            fitness = -val_loss
        elif mode == "efficiency":
            miou = self._compute_miou(model, val_loader, device, convert)
            n_params = self.count_parameters(model)
            fitness = miou - lambda_ * (n_params / max_params)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        self.fitness = fitness

        # Free all GPU objects; gc.collect() breaks any reference cycles
        # before releasing cached memory back to CUDA.
        del model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()

        return fitness

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_val_loss(
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        convert_fn,
    ) -> float:
        """Mean validation CrossEntropyLoss (ignore_index=255)."""
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for images, labels in val_loader:
                labels = convert_fn(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
        return sum(losses) / len(losses) if losses else float("inf")

    @staticmethod
    def _compute_miou(
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        convert_fn,
        n_classes: int = 19,
    ) -> float:
        """Mean Intersection-over-Union across *n_classes*, ignoring 255.

        Classes with zero ground-truth pixels in the entire validation set
        are excluded from the mean.
        """
        model.eval()
        # Confusion accumulators per class
        intersection = torch.zeros(n_classes, dtype=torch.long, device=device)
        union = torch.zeros(n_classes, dtype=torch.long, device=device)

        with torch.no_grad():
            for images, labels in val_loader:
                labels = convert_fn(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                preds = model(images).argmax(dim=1)  # (B, H, W)

                valid_mask = labels != 255
                preds = preds[valid_mask]
                labels = labels[valid_mask]

                for c in range(n_classes):
                    pred_c = preds == c
                    label_c = labels == c
                    intersection[c] += (pred_c & label_c).sum()
                    union[c] += (pred_c | label_c).sum()

        # Per-class IoU, skipping classes with 0 union
        ious: List[float] = []
        for c in range(n_classes):
            if union[c] > 0:
                ious.append((intersection[c].float() / union[c].float()).item())

        return sum(ious) / len(ious) if ious else 0.0

    # ------------------------------------------------------------------
    # PSO velocity & position updates
    # ------------------------------------------------------------------

    def update_velocity(self, gBest_arch: Dict, Cg: float) -> None:
        """Compute a new target architecture by mixing pBest and gBest.

        This mirrors the discrete-crossover approach from the thesis PSO-CNN
        (``utils.computeVelocity``): for each architectural dimension, with
        probability *Cg* we adopt the gBest value, otherwise the pBest value.

        After selection, a small random perturbation (±20%) is applied to each
        channel value to encourage exploration of the search space.

        Args:
            gBest_arch: Global best architecture dict.
            Cg: Probability of choosing gBest over pBest for each dimension.
        """
        # --- Depth ---
        if np.random.rand() < Cg:
            vel_depth = gBest_arch["depth"]
        else:
            vel_depth = self.pBest_arch["depth"]

        # --- Channels ---
        # Pad shorter list by repeating its last value so both lists have
        # the same length (max of gBest, pBest channel lengths).
        g_ch = list(gBest_arch["channels"])
        p_ch = list(self.pBest_arch["channels"])
        max_len = max(len(g_ch), len(p_ch))
        while len(g_ch) < max_len:
            g_ch.append(g_ch[-1])
        while len(p_ch) < max_len:
            p_ch.append(p_ch[-1])

        vel_channels: List[float] = []
        for g_val, p_val in zip(g_ch, p_ch):
            if np.random.rand() < Cg:
                base = g_val
            else:
                base = p_val
            # Random perturbation for exploration
            perturbed = base * np.random.uniform(0.8, 1.2)
            vel_channels.append(perturbed)

        self.velocity = {"depth": vel_depth, "channels": vel_channels}

    def update_position(self) -> None:
        """Apply the velocity to produce a new valid architecture.

        After setting depth and channels from the velocity, the following
        constraints are enforced:

        1. ``depth`` is clamped to ``[min_depth, max_depth]``.
        2. ``channels`` is resized to length ``depth + 1`` (truncated or
           extended by repeating the last value).
        3. Each channel value is clamped to ``[min_channels, max_channels]``
           and rounded to the nearest multiple of ``channel_step``.
        4. Channels are forced non-decreasing (each level >= the previous).

        Because the decoder is always derived from the encoder channels in
        ``Model.__init__``, symmetry is structurally guaranteed — no
        asymmetric architecture can ever be stored.
        """
        depth = int(self.velocity["depth"])
        depth = max(self.min_depth, min(self.max_depth, depth))

        raw_channels = list(self.velocity["channels"])

        # Resize to depth + 1
        target_len = depth + 1
        if len(raw_channels) > target_len:
            raw_channels = raw_channels[:target_len]
        while len(raw_channels) < target_len:
            raw_channels.append(raw_channels[-1])

        # Clamp, round to step, enforce non-decreasing
        channels: List[int] = []
        prev = self.min_channels
        for val in raw_channels:
            rounded = int(round(val / self.channel_step)) * self.channel_step
            rounded = max(self.min_channels, min(self.max_channels, rounded))
            rounded = max(prev, rounded)
            channels.append(rounded)
            prev = rounded

        self.arch = {"depth": depth, "channels": channels}

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------

    def clone(self) -> UNetParticle:
        """Create an independent copy of this particle.

        The PyTorch model is **not** cloned — it is rebuilt from the
        architecture dict when needed (via ``build_model``).  This avoids
        deep-copying GPU tensors.
        """
        p = UNetParticle.__new__(UNetParticle)
        p.min_depth = self.min_depth
        p.max_depth = self.max_depth
        p.min_channels = self.min_channels
        p.max_channels = self.max_channels
        p.in_channels = self.in_channels
        p.n_classes = self.n_classes
        p.channel_step = self.channel_step

        p.arch = copy.deepcopy(self.arch)
        p.velocity = copy.deepcopy(self.velocity)
        p.pBest_arch = copy.deepcopy(self.pBest_arch)
        p.pBest_fitness = self.pBest_fitness
        p.fitness = self.fitness
        return p

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ch_str = ", ".join(str(c) for c in self.arch["channels"])
        return (
            f"UNetParticle(depth={self.arch['depth']}, "
            f"channels=[{ch_str}], fitness={self.fitness})"
        )
