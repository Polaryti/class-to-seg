#!/bin/bash
#SBATCH --job-name=ws-prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# CPU only. Reads the cross-validation folds, works out the common test set of each fold,
# builds the control corpora and the COCO ground truth, and writes the task lists that the
# training and evaluation arrays iterate over.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"

python check_env.py --manifest "$MANIFEST" --data "$NAMES_YAML" --weights-dir "$WEIGHTS_DIR"

# 1. partitions of every corpus and fold, and the images no model of a fold has seen
if [ ! -f "$FOLDS_REPORT/folds_summary.json" ]; then
    python check_folds.py --root "$FOLDS_ROOT" --out "$FOLDS_REPORT"
else
    echo "check_folds : skipped ($FOLDS_REPORT/folds_summary.json exists)"
fi

# 2. per-fold data YAMLs, control corpora, ground truth, corpus statistics.
#    Never rebuilt once it exists (running jobs use it); delete $PROTOCOL by hand to start over.
python prepare_folds.py --folds-root "$FOLDS_ROOT" --report "$FOLDS_REPORT" --out "$PROTOCOL" \
    --results "$RESULTS" --workers "${SLURM_CPUS_PER_TASK:-8}"

# 3. one line per run to train and per run to score
python list_tasks.py --manifest "$MANIFEST" --runs "$RUNS" --report "$FOLDS_REPORT" --out "$TASKS" \
    --seeds "$SEEDS" --maskrcnn-seeds "$MASKRCNN_SEEDS" --figure-id "$FIGURE_ID"

python make_tables.py --results "$RESULTS" --protocol "$PROTOCOL" --out "$PAPER_TABLES"
