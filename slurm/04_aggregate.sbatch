#!/bin/bash
#SBATCH --job-name=ws-aggregate
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# CPU only. Averages the per-run results over seeds and folds, runs the paired tests,
# and writes the tables and figures of the paper.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"

python aggregate.py --results "$RESULTS" --out "$RESULTS"

FOLD0=$(python -c "from folds import folds_available; print(folds_available('$FOLDS_REPORT')[0])")
PRED="$RESULTS/predictions/${FIGURE_ID}__${FOLD0}_seed0.json.gz"
if [ -f "$PRED" ]; then
    python make_figures.py --data "$PROTOCOL/v3_$FOLD0.yaml" --origin "$FOLDS_REPORT/origin.csv" \
        --gt "$RESULTS/gt_common_$FOLD0.json.gz" --pred "$PRED" \
        --samples --qualitative "$FIGURE_ID" --out "$RESULTS/figures"
else
    echo "no detections kept for $FIGURE_ID ($PRED); qualitative figure skipped"
fi
if [ -d "$WORK/audit/manual_labels" ] && [ -n "$(ls -A "$WORK/audit/manual_labels" 2>/dev/null)" ]; then
    python audit_score.py --audit "$WORK/audit" --names "$NAMES_YAML" --out "$RESULTS/audit"
fi
python make_tables.py --results "$RESULTS" --protocol "$PROTOCOL" --out "$PAPER_TABLES"
echo "tables and macros in $PAPER_TABLES -- copy that folder to <latex project>/tables"
