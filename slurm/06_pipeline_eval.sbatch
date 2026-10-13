#!/bin/bash
#SBATCH --job-name=ws-pipeeval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# Pipeline parameters (calibrated on v2 if "auto"), classifier accuracy on its held-out split,
# sensitivity sweep of the pipeline, tables.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"

# tau, epsilon and minimum area: the values of config.sh, or recovered from v2 where they are "auto"
python calibrate_pipeline.py --stats "$WORK/pipeline/stats" --v2 "$V2_YAML" \
    --conf "${PIPE_CONF:-auto}" --eps "${PIPE_EPS:-auto}" --min-area "${PIPE_MIN_AREA:-auto}" \
    --out "$RESULTS/pipeline" --write-env "$RESULTS/pipeline/params.sh"
# shellcheck source=/dev/null
source "$RESULTS/pipeline/params.sh"
echo "pipeline parameters: tau=$PIPE_CONF eps=$PIPE_EPS min_area=$PIPE_MIN_AREA"

# The classifier table needs a held-out split of the classification corpus. Without it the rest
# of the job still runs; the corresponding rows of Table XV simply stay empty.
if [ -d "$CLS_TEST" ] && [ -f "$CLASSIFIER" ]; then
    python eval_classifier.py --data "$CLS_TEST" --classifier "$CLASSIFIER" --names "$NAMES_YAML" \
        --taus "0.0,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99,$PIPE_CONF" --device "$DEVICE" --out "$RESULTS/classifier"
else
    echo "classifier evaluation skipped: set CLS_TEST (held-out split) and CLASSIFIER in config.sh to fill it"
    echo "  CLS_TEST=$CLS_TEST"
    echo "  CLASSIFIER=$CLASSIFIER"
fi
python pipeline_sweep.py --stats "$WORK/pipeline/stats" --conf "$PIPE_CONF" --eps "$PIPE_EPS" \
    --min-area "$PIPE_MIN_AREA" --out "$RESULTS/pipeline"
python make_tables.py --results "$RESULTS" --protocol "$PROTOCOL" --out "$PAPER_TABLES"
