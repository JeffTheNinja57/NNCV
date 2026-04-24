# PSO-UNet — Particle Swarm Architecture Search for Cityscapes

This branch contains the PSO-driven architecture search and the
corresponding final-training pipeline that produces the submission for
the Cityscapes **Efficiency** benchmark of the 5LSM0 final assignment.

The baseline U-Net (peak-performance submission) lives on the
[`main`](https://github.com/JeffTheNinja57/NNCV/tree/main) branch.

---

## What this branch does

Two phases run in sequence:

1. **Search** (`pso_unet.py` + `unet_particle.py`) — a swarm of
   `UNetParticle`s explores a structured `(depth, channels)` U-Net
   space. Each particle is briefly trained, scored, and used to update
   personal/global bests via a discrete-crossover velocity operator.
2. **Final training** (`final_train.py`) — the global best
   architecture (`gBest`) is retrained from scratch with checkpoint
   resume, and the best-validation-loss checkpoint is exported for
   submission.

The fitness function switches between two modes via `--mode`:

- `efficiency` → `mIoU − λ · n_params / max_params`
- `performance` → `−val_loss`

---

## Repository layout (PSO-relevant files)

```
final_assignment/
├── pso_unet.py              # PSO main loop (swarm, iterations, gBest)
├── unet_particle.py         # One particle: arch + velocity + fitness
├── model.py                 # Dynamic U-Net built from {depth, channels}
├── train.py                 # Entry: runs PSO search → full training
├── final_train.py           # Standalone full training of a fixed arch
├── predict.py               # Submission-server inference entrypoint
├── Dockerfile
├── download_docker_and_data.sh
├── main_pso_efficiency.sh   # Local run: efficiency-mode search
├── main_pso_performance.sh  # Local run: performance-mode search
├── main_final_train.sh      # Local run: final training of gBest
├── main_test.sh             # Local 1-iter / 1-epoch smoke test
├── jobscript_pso_efficiency.sh   # SLURM: full efficiency search
├── jobscript_pso_performance.sh  # SLURM: full performance search
├── jobscript_final_train.sh      # SLURM: final training of gBest
├── jobscript_test.sh             # SLURM: smoke test
└── readmes/                 # Installation, SLURM, submission, report
```

---

## Quick start

### 1. Install and pull the data

```bash
cd final_assignment
bash download_docker_and_data.sh    # builds the docker image and pulls Cityscapes
```

See [`readmes/README-Installation.md`](final_assignment/readmes/README-Installation.md)
for prerequisites and the `.env` setup (you need `HF_TOKEN` for the
data download and `WANDB_API_KEY` for logging).

### 2. Smoke test

Before kicking off a multi-hour search, verify the full pipeline runs
end-to-end (1 iteration, 1 particle, 1 epoch):

```bash
bash main_test.sh
```

### 3. Run the PSO search

Locally:

```bash
bash main_pso_efficiency.sh        # or main_pso_performance.sh
```

On SLURM (TU/e cluster):

```bash
sbatch jobscript_pso_efficiency.sh
```

The default efficiency-mode hyper-parameters reproduce the report:

| Flag                    | Value | Meaning                                    |
|-------------------------|-------|--------------------------------------------|
| `--pso-iterations`      | 10    | PSO outer iterations                       |
| `--pso-population`      | 20    | Particles in the swarm                     |
| `--pso-epochs`          | 20    | Short-training epochs per particle eval    |
| `--cg`                  | 0.5   | Probability of adopting the gBest trait    |
| `--lambda-efficiency`   | 0.3   | Parameter-penalty weight in the fitness    |
| `--max-params`          | 5e6   | Parameter-count normaliser                 |
| `--max-depth`           | 4     | Upper bound on encoder depth               |
| `--max-channels`        | 256   | Upper bound on channels per encoder level  |
| `--full-training-epochs`| 100   | Final-training budget for the gBest        |

`train.py` runs the search **and** the final training in one go. It
prints the discovered architecture and writes the best-validation-loss
checkpoint to `checkpoints/<experiment-id>/`.

### 4. Final training of a known gBest

If you already know the architecture (e.g. from a prior search), skip
the PSO and train it directly with `final_train.py`:

```bash
bash main_final_train.sh
# or:
sbatch jobscript_final_train.sh
```

Edit `--depth` and `--channels` in those scripts to match your gBest.

### 5. Submit to the challenge server

Pick the best checkpoint, copy it to `final_assignment/model.pt`, then
follow [`readmes/README-Submission.md`](final_assignment/readmes/README-Submission.md):

```bash
docker build -t nncv-submission:efficiency -f final_assignment/Dockerfile final_assignment
docker save -o submission_efficiency.tar nncv-submission:efficiency
```

`predict.py` constructs `Model` with the gBest arch baked in, so the
container is self-contained.

---

## Reproducing the reported numbers

The report describes the gBest discovered on this branch:

- `depth = 1`, `channels = [48, 48]` (~127 k parameters, 0.50 MB)
- Peak-leaderboard mean Dice: **0.3393**
- Efficiency leaderboard: 24.2 FPS, 400.6 GFLOPs, rank 39

To reproduce: run `main_pso_efficiency.sh` followed by
`main_final_train.sh` with the gBest plugged in.

---

## See also

- [`final_assignment/readmes/README-Installation.md`](final_assignment/readmes/README-Installation.md) — environment setup
- [`final_assignment/readmes/README-Slurm.md`](final_assignment/readmes/README-Slurm.md) — SLURM workflow
- [`final_assignment/readmes/README-Submission.md`](final_assignment/readmes/README-Submission.md) — Docker + tar submission
- [`final_assignment/readmes/README-Report.md`](final_assignment/readmes/README-Report.md) — report rubric
