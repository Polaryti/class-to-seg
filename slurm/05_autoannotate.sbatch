#!/bin/bash
#SBATCH --job-name=ws-autoann
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# One array task per shard (0..PIPE_SHARDS-1): BiRefNet + classifier on the classification
# corpus, logging per-image confidences and contour statistics. Resumable.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"
read -r -a EXTRA <<< "$EXTRA_AUTOANN_ARGS"

python autoannotate.py --src "$CLS_SRC" --names "$NAMES_YAML" --classifier "$CLASSIFIER" \
    --birefnet "$BIREFNET" --mask-thr "$PIPE_MASK_THR" --eps-grid "$PIPE_EPS_GRID" \
    --out "$WORK/pipeline" --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$PIPE_SHARDS" --device "$DEVICE" "${EXTRA[@]}"
