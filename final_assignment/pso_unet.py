"""
PSOUNet — Particle Swarm Optimisation loop for U-Net architecture search.

This module is the direct analogue of ``psoCNN.py`` from the thesis PSO-CNN
codebase, ported to PyTorch and adapted to the structured ``(depth, channels)``
architecture space of U-Net.

Execution flow (mirrors ``psoCNN.__init__`` + ``psoCNN.fit``):

1. Initialise a population of ``UNetParticle`` objects with random valid
   architectures.
2. Evaluate every particle (short training) to establish initial pBest and
   gBest.
3. For ``n_iter`` iterations:
   a. For each particle — velocity update → position update → evaluate →
      update pBest → update gBest.
   b. Record gBest fitness for this iteration.
4. Return the gBest architecture dict for full training.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from unet_particle import UNetParticle


class PSOUNet:
    """PSO-based Neural Architecture Search for U-Net.

    Attributes:
        gBest_arch: Architecture dict of the global best particle.
        gBest_fitness: Fitness of the global best.
        fitness_history: gBest fitness after each iteration (for logging).
    """

    def __init__(
        self,
        pop_size: int,
        n_iter: int,
        pso_epochs: int,
        lr: float,
        min_depth: int,
        max_depth: int,
        min_channels: int,
        max_channels: int,
        in_channels: int,
        n_classes: int,
        mode: str,
        Cg: float,
        lambda_: float = 0.3,
        max_params: int = 5_000_000,
    ):
        """
        Args:
            pop_size: Number of particles in the swarm.
            n_iter: Number of PSO iterations.
            pso_epochs: Short training epochs per particle evaluation.
            lr: Learning rate used during short training.
            min_depth: Minimum encoder depth a particle can have.
            max_depth: Maximum encoder depth a particle can have.
            min_channels: Minimum channel count at any encoder level.
            max_channels: Maximum channel count at any encoder level.
            in_channels: Number of input image channels (e.g. 3).
            n_classes: Number of segmentation classes (e.g. 19).
            mode: ``"performance"`` or ``"efficiency"``.
            Cg: Probability of adopting gBest trait during velocity update.
            lambda_: Parameter penalty weight (efficiency mode only).
            max_params: Parameter count normaliser (efficiency mode only).
        """
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.pso_epochs = pso_epochs
        self.lr = lr
        self.mode = mode
        self.Cg = Cg
        self.lambda_ = lambda_
        self.max_params = max_params

        # Create the swarm
        self.population: List[UNetParticle] = []
        for _ in range(pop_size):
            self.population.append(
                UNetParticle(
                    min_depth=min_depth,
                    max_depth=max_depth,
                    min_channels=min_channels,
                    max_channels=max_channels,
                    in_channels=in_channels,
                    n_classes=n_classes,
                )
            )

        # Will be set during search()
        self.gBest_arch: Optional[Dict] = None
        self.gBest_fitness: Optional[float] = None
        self.fitness_history: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_particle(
        self,
        particle: UNetParticle,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Evaluate a particle and update its personal best if improved."""
        fitness = particle.compute_fitness(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            mode=self.mode,
            epochs=self.pso_epochs,
            lr=self.lr,
            lambda_=self.lambda_,
            max_params=self.max_params,
        )

        # Update pBest (mirrors psoCNN.fit pBest logic)
        if particle.pBest_fitness is None or fitness >= particle.pBest_fitness:
            particle.pBest_arch = copy.deepcopy(particle.arch)
            particle.pBest_fitness = fitness

        return fitness

    def _update_gbest(self, particle: UNetParticle) -> None:
        """Update global best if this particle's pBest beats it."""
        if (self.gBest_fitness is None
                or particle.pBest_fitness >= self.gBest_fitness):
            self.gBest_arch = copy.deepcopy(particle.pBest_arch)
            self.gBest_fitness = particle.pBest_fitness

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict:
        """Run the full PSO architecture search.

        This method mirrors ``psoCNN.__init__`` (initial evaluation) followed
        by ``psoCNN.fit`` (iterative search).

        Args:
            train_loader: Cityscapes training data loader.
            val_loader: Cityscapes validation data loader.
            device: CUDA / CPU device.

        Returns:
            The global best architecture dict
            ``{"depth": int, "channels": [int, ...]}``.
        """
        # ---- Phase 1: Initial evaluation of all particles ---------------
        # (mirrors psoCNN.__init__ lines 312-355)
        print("=" * 60)
        print("PSO U-Net Architecture Search")
        print(f"  Mode: {self.mode}")
        print(f"  Population: {self.pop_size}")
        print(f"  Iterations: {self.n_iter}")
        print(f"  PSO epochs/particle: {self.pso_epochs}")
        print(f"  Cg: {self.Cg}")
        print("=" * 60)

        print("\nPhase 1: Evaluating initial population...")
        for idx, particle in enumerate(self.population):
            print(f"  Particle {idx + 1}/{self.pop_size}: {particle}")
            fitness = self._evaluate_particle(
                particle, train_loader, val_loader, device,
            )
            self._update_gbest(particle)
            print(f"    Fitness: {fitness:.6f}")

        print(f"\nInitial gBest fitness: {self.gBest_fitness:.6f}")
        print(f"Initial gBest arch:   {self.gBest_arch}")
        self.fitness_history.append(self.gBest_fitness)

        # ---- Phase 2: PSO iterations ------------------------------------
        # (mirrors psoCNN.fit)
        for iteration in range(self.n_iter):
            print(f"\n--- Iteration {iteration + 1}/{self.n_iter} ---")

            for idx, particle in enumerate(self.population):
                # Velocity update: mix pBest and gBest with probability Cg
                particle.update_velocity(self.gBest_arch, self.Cg)

                # Position update: apply velocity, clamp, enforce constraints
                particle.update_position()

                print(f"  Particle {idx + 1}/{self.pop_size}: {particle}")

                # Evaluate new architecture
                fitness = self._evaluate_particle(
                    particle, train_loader, val_loader, device,
                )
                print(f"    Fitness: {fitness:.6f}")

                # Update gBest if this particle improved
                prev_gBest = self.gBest_fitness
                self._update_gbest(particle)
                if self.gBest_fitness > prev_gBest:
                    print(f"    ** New gBest! {prev_gBest:.6f} -> "
                          f"{self.gBest_fitness:.6f}")
                    print(f"       Arch: {self.gBest_arch}")

            self.fitness_history.append(self.gBest_fitness)
            print(f"  gBest fitness after iteration {iteration + 1}: "
                  f"{self.gBest_fitness:.6f}")

        print("\n" + "=" * 60)
        print("PSO Search Complete")
        print(f"  Best architecture: {self.gBest_arch}")
        print(f"  Best fitness:      {self.gBest_fitness:.6f}")
        print("=" * 60)

        return self.gBest_arch

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_gbest_arch(self) -> Dict:
        """Return the global best architecture dict."""
        return self.gBest_arch

    def get_fitness_history(self) -> List[float]:
        """Return gBest fitness values recorded after each iteration."""
        return self.fitness_history
