# Spot Training — Design Specification

## Goal

Run nanochat training (up to full speedrun pipeline) on AWS spot GPU instances with automatic preemption recovery and multi-region failover, using SkyPilot for orchestration and S3 for checkpoint persistence.

## Architecture Overview

A single SkyPilot YAML file (`runs/spot_train.yaml`) defines the managed spot job. SkyPilot handles provisioning, preemption recovery, and multi-region failover. Checkpoints persist on an S3 bucket mounted into the instance filesystem. The user's Mac acts as the controller — launching jobs and checking status via `sky jobs` commands.

```
Mac (controller)                SkyPilot                      Spot Instance (8x H100)
 │                                 │                                │
 ├── sky jobs launch ──────────────┤                                │
 │                                 ├── find cheapest spot ──────────┤
 │                                 │                                ├── setup (clone, venv, deps)
 │                                 │                                ├── check S3 for checkpoint
 │   sky jobs queue                │                                ├── resume or start fresh
 │   sky jobs logs                 │                                ├── training (save every 500 steps)
 │                                 │                                │
 │                                 │   preemption detected          │
 │                                 │                                ├── checkpoint already on S3
 │                                 ├── relaunch in next region ─────┤
 │                                 │                                ├── resume from last checkpoint
 │                                 │                                └── training complete
 │                                 │                                │
 ├── download model from S3 ◄──────┤                                │
```

**Mac sleep behavior:** SkyPilot's managed jobs controller runs as a local process. If the Mac sleeps during a preemption, recovery is delayed until the Mac wakes. Checkpoints are safe on S3 regardless — when the Mac wakes, `sky jobs queue` shows the job status and SkyPilot resumes the recovery process. If the Mac reboots or the controller process is killed, the spot instance may become orphaned. In this case, run `sky jobs queue` to check status and `sky jobs launch` to restart the pipeline (it will resume from the last S3 checkpoint).

## Components

### 1. SkyPilot YAML (`runs/spot_train.yaml`)

Single file defining the entire job.

**Resources:**
- Instance: p5.48xlarge (8x H100 80GB)
- Spot pricing, with ordered region preference:
  1. eu-west-2 (London) — cheapest at ~$9-12/h
  2. us-west-2 (Oregon) — high availability at ~$18/h
  3. us-east-1 (Virginia) — high availability at ~$18/h

**File mounts:**
- S3 bucket `nanochat-checkpoints` mounted at `/checkpoints`
- Used for checkpoint persistence only (not dataset)

**Storage strategy — two directories:**
- `/checkpoints/nanochat/` (S3 mount) — checkpoints and tokenizer only. Persists across preemptions.
- `/local/nanochat/` (instance NVMe) — dataset downloads, temporary training files. Lost on preemption but re-downloadable.

Training writes checkpoints to local NVMe first, then syncs to S3 after each save. This avoids FUSE filesystem issues with `torch.save` (which does random-access writes incompatible with S3's object storage semantics). A post-save `rsync` or `aws s3 sync` copies completed checkpoints to the S3 mount.

**Environment variables (configurable at launch):**

| Env var | Default | Purpose |
|---------|---------|---------|
| `DEPTH` | 24 | Transformer depth (determines model size) |
| `SKIP_TOKENIZER` | 0 | Skip tokenizer training (set 1 if already done) |
| `SKIP_SFT` | 0 | Skip SFT phase |
| `SAVE_EVERY` | 500 | Checkpoint frequency (steps). 500 steps ≈ 17 min on H100 |
| `TARGET_RATIO` | 8 | target-param-data-ratio for training horizon |

### 2. Setup Phase

Runs once per instance launch:

1. Clone nanochat repo from user's fork (`youest/nanochat`, branch `cellmem/v2`)
2. Install uv and create venv with GPU extras
3. Dependencies are installed fresh each time (fast with uv, ~30 seconds)

### 3. Run Phase — Configurable Pipeline

The run script executes up to 4 stages, each skippable:

**Stage 1: Tokenizer** (skip if `SKIP_TOKENIZER=1` or tokenizer already exists on S3)
- Download first 8 data shards to local NVMe for tokenizer training
- Kick off background download of 170 shards to local NVMe
- Train tokenizer (vocab_size=32768)
- Evaluate tokenizer
- Copy tokenizer artifacts to S3 mount

**Stage 2: Base Training** (always runs, with automatic resume)
- Sync latest checkpoint from S3 to local NVMe
- Detect last checkpoint step
- If checkpoint exists: `--resume-from-step=$LAST_STEP`
- If no checkpoint: start fresh
- `NANOCHAT_BASE_DIR=/local/nanochat` — training writes to local NVMe
- After each checkpoint save, `aws s3 sync` copies to S3 bucket
- `--device-batch-size=16` (fits d24 on H100 80GB)
- `--fp8` enabled (H100 native support)
- `--save-every=$SAVE_EVERY`
- `OMP_NUM_THREADS=1` for optimal multi-GPU performance

**Stage 3: Base Eval** (runs after training completes)
- CORE metric evaluation
- Results saved to S3

**Stage 4: SFT** (skip if `SKIP_SFT=1`)
- Download synthetic identity conversations
- Run SFT training and evaluation
- Copy results to S3

### 4. Resume Logic

The resume mechanism leverages nanochat's existing `--resume-from-step` which restores:
- Model weights (`model_<step>.pt`)
- Optimizer state (`optim_<step>_rank0.pt` through `rank7.pt`)
- Dataloader position (parquet file index, row group, epoch)
- Training loop state (step, smooth_train_loss, total_training_time)

On launch, the run script:
1. Syncs checkpoints from S3 to local NVMe: `aws s3 sync s3://nanochat-checkpoints/nanochat/base_checkpoints/ /local/nanochat/base_checkpoints/`
2. Detects the last checkpoint step:
```bash
LAST_STEP=$(ls /local/nanochat/base_checkpoints/d${DEPTH}/meta_*.json 2>/dev/null \
  | sed 's/.*meta_0*\([0-9]*\)\.json/\1/' | sort -n | tail -1)
```
3. Passes `--resume-from-step=$LAST_STEP` if a checkpoint exists

### 5. Checkpoint Sync Strategy

To avoid FUSE filesystem issues with `torch.save`, checkpoints are written to local NVMe and synced to S3:

```bash
# Wrapper: train with periodic S3 sync
# base_train.py saves checkpoints to local disk (NANOCHAT_BASE_DIR=/local/nanochat)
# A background sync loop copies new checkpoints to S3

sync_checkpoints() {
  while true; do
    sleep 60
    aws s3 sync /local/nanochat/base_checkpoints/d${DEPTH}/ \
      s3://nanochat-checkpoints/nanochat/base_checkpoints/d${DEPTH}/ \
      --exclude "optim_*" --include "*.json" --include "model_*.pt"
    # Sync optimizer separately (large files, only latest)
    LATEST=$(ls /local/nanochat/base_checkpoints/d${DEPTH}/meta_*.json 2>/dev/null | sort | tail -1 | sed 's/meta/optim/' | sed 's/.json/_rank*.pt/')
    for f in $LATEST; do
      aws s3 cp "$f" s3://nanochat-checkpoints/nanochat/base_checkpoints/d${DEPTH}/$(basename $f)
    done
  done
}

sync_checkpoints &
SYNC_PID=$!

# Run training
torchrun ...

# Final sync
kill $SYNC_PID 2>/dev/null
aws s3 sync /local/nanochat/base_checkpoints/d${DEPTH}/ \
  s3://nanochat-checkpoints/nanochat/base_checkpoints/d${DEPTH}/
```

### 6. Checkpoint Economics

With `SAVE_EVERY=500` on p5.48xlarge:
- Each step takes ~2 seconds on 8xH100
- 500 steps ≈ 17 minutes of training
- At $18/h spot price, each checkpoint protects ~$5 of compute
- At $10/h (London), each checkpoint protects ~$2.80 of compute
- Total checkpoints for d24 training (~5000 steps): 10 files
- Each checkpoint: ~2GB (model) + ~2.8GB (optimizer x8 ranks) ≈ 5GB
- Total S3 storage: ~50GB (can clean old checkpoints)

## Usage

```bash
# Install SkyPilot (one time)
pip install "skypilot[aws]"
sky check

# Full speedrun pipeline
sky jobs launch runs/spot_train.yaml

# Just base training (skip tokenizer and SFT)
sky jobs launch runs/spot_train.yaml \
  --env SKIP_TOKENIZER=1 --env SKIP_SFT=1

# Smaller model for testing
sky jobs launch runs/spot_train.yaml --env DEPTH=12

# Monitor
sky jobs queue
sky jobs logs

# Download results when done
aws s3 sync s3://nanochat-checkpoints/nanochat/base_checkpoints/ \
  ~/.cache/nanochat/base_checkpoints/

# Clean up old checkpoints (keep only latest)
aws s3 rm s3://nanochat-checkpoints/nanochat/base_checkpoints/d24/ \
  --recursive --exclude "*/meta_005000*" --exclude "*/model_005000*" --exclude "*/optim_005000*"
```

## Prerequisites

- AWS CLI configured with credentials
- SkyPilot installed (`pip install "skypilot[aws]"`)
- S3 bucket `nanochat-checkpoints` created (any region — S3 is globally accessible)
- Spot instance quota for p5.48xlarge in target regions (request via AWS console if needed)

## Cost Estimate

| Scenario | Region | Price/h | Training time | Total |
|----------|--------|---------|---------------|-------|
| Best case (no preemption) | eu-west-2-az3 | $9.60 | ~3h | ~$29 |
| Typical (1 preemption) | mixed | ~$12 | ~3.5h | ~$42 |
| Worst case (3 preemptions) | mixed | ~$15 | ~4.5h | ~$68 |

Setup overhead (clone, deps, dataset download) adds ~15-30 minutes per instance launch.
S3 storage: ~$1.15/month for 50GB.

## Non-Goals

- Telegram/Slack notifications (future enhancement)
- Multi-node distributed training (single 8-GPU node is sufficient)
- Custom infrastructure code (SkyPilot handles everything)

## Files to Create

| File | Purpose |
|------|---------|
| `runs/spot_train.yaml` | SkyPilot job definition |
| `docs/superpowers/specs/2026-03-23-spot-training-design.md` | This spec |
