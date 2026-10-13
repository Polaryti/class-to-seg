#!/bin/bash
#SBATCH --job-name=ws-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
# One array task per line of $TASKS/train_tasks.txt ("<id> <fold> <seed> <flags>").
# Safe to resubmit: a finished run exits at once, an interrupted one resumes from its
# last checkpoint.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"
TASKFILE="$TASKS/train_tasks.txt"
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$TASKFILE")
if [ -z "$LINE" ]; then echo "no task at index ${SLURM_ARRAY_TASK_ID:-0} of $TASKFILE"; exit 0; fi
read -r ID FOLD SEED _FLAGS <<< "$LINE"
KIND=$(python -c "from common import read_manifest; print(read_manifest('$MANIFEST')['$ID']['kind'])")
if [ "$KIND" = "maskrcnn" ]; then read -r -a EXTRA <<< "$EXTRA_MASKRCNN_ARGS"; else read -r -a EXTRA <<< "$EXTRA_TRAIN_ARGS"; fi
echo "training $ID $FOLD seed $SEED ($KIND)"
python train_run.py --id "$ID" --fold "$FOLD" --seed "$SEED" --manifest "$MANIFEST" \
    --protocol "$PROTOCOL" --runs "$RUNS" --weights-dir "$WEIGHTS_DIR" --device "$DEVICE" \
    --workers "$(( ${SLURM_CPUS_PER_TASK:-16} / 2 ))" "${EXTRA[@]}"
