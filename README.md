# 5LSM0 Final Assignment — Baseline U-Net

This branch contains the **baseline submission**: the canonical depth-4
Ronneberger U-Net provided by the 5LSM0 course staff, trained end-to-end on
Cityscapes (19 classes) and packaged for the CodaLab challenge server.

For the Particle Swarm Optimisation (PSO) architecture-search submission, see
the [`pso`](https://github.com/JeffTheNinja57/NNCV/tree/pso) branch.

## What this branch does

- Trains the baseline U-Net (`depth=4`, channels `[64,128,256,512,512]`,
  ~17.3 M parameters) on Cityscapes at 256×256 resolution.
- Logs training curves and qualitative segmentations to Weights & Biases.
- Saves the best-validation-loss checkpoint to `checkpoints/<experiment-id>/`.
- Provides a Docker recipe + scripts for submitting the trained model to the
  course CodaLab leaderboard.

## Repository layout

```
final_assignment/
├── train.py                  # baseline training loop (this is the entry point)
├── model.py                  # U-Net definition (uses DEFAULT_ARCH)
├── predict.py                # inference wrapper used by the CodaLab container
├── main.sh                   # invokes train.py with the reported hyperparameters
├── jobscript_slurm.sh        # SLURM submission for the TU/e cluster
├── Dockerfile                # CodaLab submission image
├── download_docker_and_data.sh
└── readmes/                  # course-provided installation / submission notes
```

## Quick start

1. **Install** — follow [`final_assignment/readmes/README-Installation.md`](final_assignment/readmes/README-Installation.md).
2. **Get the data** — `bash final_assignment/download_docker_and_data.sh`
   (requires `HF_TOKEN` in `.env`).
3. **Login to wandb** — `wandb login` (or export `WANDB_API_KEY` in `.env`).
4. **Train locally** — from `final_assignment/`:
   ```bash
   bash main.sh
   ```
   Or on the TU/e SLURM cluster:
   ```bash
   sbatch jobscript_slurm.sh
   ```
5. **Submit to CodaLab** — see [`final_assignment/readmes/README-Submission.md`](final_assignment/readmes/README-Submission.md).

## Reproducing the reported numbers

The baseline run reported in the paper used:

| Hyperparameter   | Value           |
|------------------|-----------------|
| Optimiser        | AdamW, lr=1e-3  |
| Batch size       | 64              |
| Resolution       | 256×256         |
| Epochs           | 100 (planned)   |
| Loss             | CrossEntropy, ignore_index=255 |
| Seed             | 42              |

Best validation loss: **0.4363** at epoch 38.
Test set (CodaLab Peak leaderboard): **0.4105 mDice / 0.3340 mIoU**.

## See also

- [`final_assignment/readmes/README-Installation.md`](final_assignment/readmes/README-Installation.md) — environment setup
- [`final_assignment/readmes/README-Slurm.md`](final_assignment/readmes/README-Slurm.md) — running on the cluster
- [`final_assignment/readmes/README-Submission.md`](final_assignment/readmes/README-Submission.md) — building the Docker image and submitting to CodaLab
- [`final_assignment/readmes/README-Report.md`](final_assignment/readmes/README-Report.md) — paper requirements
