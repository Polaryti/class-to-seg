#!/bin/bash
# End-to-end smoke test on a tiny synthetic dataset with three folds. Runs every SLURM
# script in order with bash (no queue), with two epochs and 128-pixel images: about twenty
# minutes on a CPU, less on a GPU. It trains 21 tiny runs and scores 27.
# It downloads yolo11n-seg.pt / yolo11n-cls.pt from GitHub (run on a node with internet
# or put them in $SMOKE/weights first). BiRefNet is replaced by an Otsu threshold and
# Mask R-CNN starts from random weights, so no other download is needed.
#
#     bash smoke_test/run_smoke_test.sh [/tmp/ws_smoke] [cpu|0]
#
# It can also be submitted, to get a GPU for it:
#     sbatch --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
#            smoke_test/run_smoke_test.sh "$HOME/ws_smoke" 0
set -eo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
# sbatch copies the job script into its spool directory, so $0 no longer points at the code;
# the directory sbatch was called from, or CODE_DIR, is then where the code really is.
if [ ! -f "$CODE/smoke_test/make_synthetic_data.py" ]; then
    for c in "${SLURM_SUBMIT_DIR:-}" "${CODE_DIR:-}" "$HOME/waste_seg_experiments"; do
        if [ -n "$c" ] && [ -f "$c/smoke_test/make_synthetic_data.py" ]; then CODE="$c"; break; fi
    done
fi
if [ ! -f "$CODE/smoke_test/make_synthetic_data.py" ]; then
    echo "Cannot find the code: $0 is not inside waste_seg_experiments and neither is" >&2
    echo "SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) nor CODE_DIR (${CODE_DIR:-unset})." >&2
    echo "Run it from the code folder:  bash smoke_test/run_smoke_test.sh" >&2
    exit 1
fi
[ -n "${SLURM_JOB_ID:-}" ] && echo "(submitted job ${SLURM_JOB_ID}; code directory: $CODE)"
SMOKE="${1:-/tmp/ws_smoke_$USER}"
DEV="${2:-cpu}"
case "$SMOKE" in /*) ;; *) SMOKE="$(pwd)/$SMOKE" ;; esac
rm -rf "$SMOKE"
mkdir -p "$SMOKE/weights"
cd "$CODE"

echo "== synthetic data (three folds per corpus, folded independently)"
python smoke_test/make_synthetic_data.py --out "$SMOKE/data" --folds 3 --n1 50 --n2 50

# Two manifests: the first trains the runs that stand for the ones already available, the
# second is the real shape of the study -- most configurations point at existing run folders
# and only E18, E19 and E20 are trained here.
R="$SMOKE/work/runs"
cat > "$SMOKE/experiments_one_seed.tsv" << MAN
E3   yolo     v1   yolo11n-seg.pt  auto  -
E8   yolo     v2   yolo11n-seg.pt  auto  -
MAN
cat > "$SMOKE/experiments_two_seed.tsv" << MAN
E13  yolo     v3   yolo11n-seg.pt  auto  -
MAN
cat > "$SMOKE/experiments_smoke.tsv" << MAN
E3   yolo     v1              yolo11n-seg.pt  $R/E3   -
E8   yolo     v2              yolo11n-seg.pt  $R/E8   -
E13  yolo     v3              yolo11n-seg.pt  $R/E13  -
E16  yolo     v3              yolo11n-seg.pt  $R/E13  copy_paste=0.3
E18  maskrcnn v3              -               auto    -
E19  yolo     v3_sizematched  yolo11n-seg.pt  auto    -
E20  yolo     v3_deranged     yolo11n-seg.pt  auto    -
MAN

cat > "$SMOKE/config_smoke.sh" << CFG
export CODE_DIR="$CODE"
export WORK="$SMOKE/work"
export FOLDS_ROOT="$SMOKE/data/folds"
export CLS_SRC="$SMOKE/data/cls"
export CLS_TEST="$SMOKE/data/cls/test"
export CLASSIFIER="$SMOKE/work/runs/cls/weights/best.pt"
export BIREFNET="unused"
export PIPE_CONF="auto"; export PIPE_EPS="auto"; export PIPE_MIN_AREA="auto"; export PIPE_MASK_THR="0.5"
export PIPE_EPS_GRID="0.0005,0.001,0.0015,0.002,0.003,0.005,0.0075,0.01,0.02"; export PIPE_SHARDS=2
export MANIFEST="\${MANIFEST:-$SMOKE/experiments_smoke.tsv}"
export SEEDS="0"; export MASKRCNN_SEEDS="0"; export FIGURE_ID="E13"; export BOOTSTRAP=50
export PARTITION=""; export ACCOUNT=""; export TRAIN_TIME="00:30:00"; export CHAIN=1; export MAX_PARALLEL=2
activate_env() { :; }
export DEVICE="$DEV"
export EXTRA_TRAIN_ARGS="--epochs 2 --batch 4 --imgsz 128 --workers 0"
export EXTRA_MASKRCNN_ARGS="--epochs 1 --batch 2 --no-pretrained --max-images 4 --workers 0 --eval-every 1"
export EXTRA_EVAL_ARGS="--imgsz 128"
export EXTRA_AUTOANN_ARGS="--segmenter otsu"
export PROTOCOL="\$WORK/protocol"; export FOLDS_REPORT="\$WORK/folds_report"; export TASKS="\$WORK/tasks"
export RUNS="\$WORK/runs"; export RESULTS="\$WORK/results"
export WEIGHTS_DIR="$SMOKE/weights"; export LOG_DIR="\$WORK/logs"; export PAPER_TABLES="\$WORK/paper_tables"
export NAMES_YAML="\$FOLDS_ROOT/v3/fold0/data.yaml"; export V2_YAML="\$FOLDS_ROOT/v2/fold0/data.yaml"
export TORCH_HOME="\$WORK/torch_home"; export HF_HOME="\$WORK/hf_home"; export PYTHONUNBUFFERED=1
CFG
export CONFIG="$SMOKE/config_smoke.sh"
# shellcheck source=/dev/null
source "$CONFIG"
mkdir -p "$LOG_DIR"

echo "== weights"
python fetch_weights.py --weights-dir "$WEIGHTS_DIR" --manifest "$MANIFEST" --skip-birefnet --skip-maskrcnn --extra yolo11n-cls.pt

echo "== 00 prepare"
bash slurm/00_prepare.sbatch
echo "== 00 prepare again (must be a no-op)"
bash slurm/00_prepare.sbatch | grep -i "already prepared" || { echo "prepare is not idempotent"; exit 1; }

# the runs that stand for the ones already trained: same layout, <id>/<fold>_seed<seed>
echo "== runs of the configurations that already exist"
# E13 gets two seeds so that averaging seeds inside a fold is exercised; one is enough elsewhere
python list_tasks.py --manifest "$SMOKE/experiments_one_seed.tsv" --runs "$RUNS" --report "$FOLDS_REPORT" \
    --out "$TASKS/pre" --seeds "0"
python list_tasks.py --manifest "$SMOKE/experiments_two_seed.tsv" --runs "$RUNS" --report "$FOLDS_REPORT" \
    --out "$TASKS/pre2" --seeds "0 1"
cat "$TASKS/pre2/train_tasks.txt" >> "$TASKS/pre/train_tasks.txt"
read -r -a EXTRA <<< "$EXTRA_TRAIN_ARGS"
n=0
while read -r ID FOLD SEED _; do
    MAN="$SMOKE/experiments_one_seed.tsv"; grep -q "^$ID " "$MAN" || MAN="$SMOKE/experiments_two_seed.tsv"
    python train_run.py --id "$ID" --fold "$FOLD" --seed "$SEED" --manifest "$MAN" \
        --protocol "$PROTOCOL" --runs "$RUNS" --weights-dir "$WEIGHTS_DIR" --device "$DEVICE" \
        "${EXTRA[@]}" > "$LOG_DIR/pretrain_$n.log" 2>&1 \
        || { echo "training $ID $FOLD seed $SEED failed"; tail -30 "$LOG_DIR/pretrain_$n.log"; exit 1; }
    n=$((n + 1))
done < "$TASKS/pre/train_tasks.txt"
echo "   $n runs trained"

echo "== 01 train (the runs of the study: E18, E19, E20)"
N_TRAIN=$(wc -l < "$TASKS/train_tasks.txt")
for ((i = 0; i < N_TRAIN; i++)); do
    SLURM_ARRAY_TASK_ID=$i bash slurm/01_train.sbatch > "$LOG_DIR/train_$i.log" 2>&1 \
        || { tail -30 "$LOG_DIR/train_$i.log"; exit 1; }
done
echo "== 01 train again (must detect finished runs)"
SLURM_ARRAY_TASK_ID=0 bash slurm/01_train.sbatch 2>&1 | grep -iE "already finished|nothing to do"

echo "== 03 evaluate (array)"
python list_tasks.py --manifest "$MANIFEST" --runs "$RUNS" --report "$FOLDS_REPORT" --out "$TASKS" \
    --seeds "$SEEDS" --maskrcnn-seeds "$MASKRCNN_SEEDS" --figure-id "$FIGURE_ID"
N_EVAL=$(wc -l < "$TASKS/eval_tasks.txt")
for ((i = 0; i < N_EVAL; i++)); do
    SLURM_ARRAY_TASK_ID=$i bash slurm/03_eval.sbatch > "$LOG_DIR/eval_$i.log" 2>&1 \
        || { tail -30 "$LOG_DIR/eval_$i.log"; exit 1; }
done
echo "   $N_EVAL runs scored"

echo "== classifier for the pipeline experiments (tiny)"
python - << PY
from ultralytics import YOLO
YOLO("$WEIGHTS_DIR/yolo11n-cls.pt").train(data="$CLS_SRC", epochs=2, imgsz=64, batch=8, device="$DEV", workers=0,
    project="$RUNS", name="cls", exist_ok=True, plots=False, verbose=False)
PY

echo "== 05 autoannotate (array)"
for ((i = 0; i < PIPE_SHARDS; i++)); do
    SLURM_ARRAY_TASK_ID=$i bash slurm/05_autoannotate.sbatch > "$LOG_DIR/autoann_$i.log" 2>&1 \
        || { tail -30 "$LOG_DIR/autoann_$i.log"; exit 1; }
done

echo "== 06 pipeline evaluation"
bash slurm/06_pipeline_eval.sbatch > "$LOG_DIR/pipeeval.log" 2>&1 || { tail -30 "$LOG_DIR/pipeeval.log"; exit 1; }

echo "== audit (sample + simulated manual labels)"
python audit_sample.py --data "$PROTOCOL/v2_fold0.yaml" --per-class 2 --out "$WORK/audit"
cp "$WORK"/audit/generated_labels/*.txt "$WORK/audit/manual_labels/"

echo "== 04 aggregate + tables + figures"
bash slurm/04_aggregate.sbatch > "$LOG_DIR/aggregate.log" 2>&1 || { tail -40 "$LOG_DIR/aggregate.log"; exit 1; }
grep -A 20 "config  " "$LOG_DIR/aggregate.log" || true

echo "== outputs"
for f in folds_report/folds_summary.json protocol/protocol_summary.json results/metrics.csv results/per_fold.csv \
         results/per_class.csv results/comparisons.csv results/models.csv results/corpus_stats.json \
         results/pipeline/calibration.json results/pipeline/pipeline_tau_sweep.csv \
         results/classifier/classifier_summary.json results/audit/audit_summary.csv \
         paper_tables/tab_main.tex paper_tables/tab_comparisons.tex paper_tables/results_macros.tex \
         results/figures/corpus_blue_1.jpg; do
    if [ -s "$WORK/$f" ]; then echo "  ok  $f"; else echo "  MISSING $f"; exit 1; fi
done
python smoke_test/check_outputs.py --work "$WORK" --tables "$PAPER_TABLES"
echo "SMOKE TEST PASSED ($SMOKE)"
