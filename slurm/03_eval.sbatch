#!/bin/bash
#SBATCH --job-name=ws-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# One array task per line of $TASKS/eval_tasks.txt: predicts and scores that run on the
# common test set of its fold, writing one JSON in $RESULTS/per_run. Re-running a task
# simply overwrites its JSON.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"
read -r -a EXTRA <<< "$EXTRA_EVAL_ARGS"

TASKFILE="$TASKS/eval_tasks.txt"
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$TASKFILE")
if [ -z "$LINE" ]; then echo "no task at index ${SLURM_ARRAY_TASK_ID:-0} of $TASKFILE"; exit 0; fi
read -r ID FOLD SEED FLAGS <<< "$LINE"
case ",$FLAGS," in *,gflops,*) EXTRA+=(--gflops) ;; esac
case ",$FLAGS," in *,savedets,*) EXTRA+=(--save-dets) ;; esac
echo "scoring $ID $FOLD seed $SEED (flags: $FLAGS)"
python eval_run.py --id "$ID" --fold "$FOLD" --seed "$SEED" --manifest "$MANIFEST" \
    --runs "$RUNS" --results "$RESULTS" --report "$FOLDS_REPORT" --device "$DEVICE" "${EXTRA[@]}"
