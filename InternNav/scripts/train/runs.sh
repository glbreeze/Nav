#!/bin/bash
# Smoke-test run for LoGoPlanner training on the small sample shard.
#
# Usage (from InternNav/):
#   bash scripts/train/runs.sh
#
# What this does:
#   - Single GPU (cuda:0), batch_size=2, num_workers=0 to keep errors readable.
#   - Points at the small InternData-N1 shard at traj_data_navdp.
#   - Writes dataset path cache to /tmp so repeated runs reuse the scan.
#   - Runs 1 epoch (set EPOCHS env var to override).
#
# Override any of these env vars on the command line, e.g.:
#   BATCH_SIZE=4 EPOCHS=3 bash scripts/train/runs.sh

set -eo pipefail

NAME=${NAME:-logoplanner_smoke}
MODEL=logoplanner
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
ROOT_DIR=${ROOT_DIR:-/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data_navdp}
DATASET_CACHE=${DATASET_CACHE:-/tmp/logoplanner_dataset_lerobot.json}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_SHOW_CPP_STACKTRACES=1

# Make the vendored diffusion_policy submodule importable as `diffusion_policy`.
# Run `git submodule update --init` first if src/diffusion-policy is empty.
export PYTHONPATH="$PWD/src/diffusion-policy:${PYTHONPATH:-}"

python scripts/train/train.py \
    --name "$NAME" \
    --model-name "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --root-dir "$ROOT_DIR" \
    --dataset-navdp "$DATASET_CACHE"
