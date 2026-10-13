#!/bin/bash
# Submit the whole study with dependencies. Run from anywhere:
#     bash slurm/submit_all.sh            # everything
#     bash slurm/submit_all.sh models     # folds + training + evaluation + tables
#     bash slurm/submit_all.sh pipeline   # pipeline experiments only
#     bash slurm/submit_all.sh eval       # evaluation + tables only (nothing is trained)
# Re-running is safe: finished runs exit at once and interrupted ones resume, so this is
# also how training that hit the time limit is continued.
#
# This script SUBMITS the jobs; it is not a job itself, so run it with bash on the login
# node. Submitting it with sbatch still works, but its output -- the job ids -- ends up in
# a slurm-*.out file instead of on your screen.
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# sbatch copies the job script into its spool directory, so $0 no longer points at the code
if [ ! -f "$HERE/experiments.tsv" ]; then
    for c in "${SLURM_SUBMIT_DIR:-}" "${CODE_DIR:-}" "$HOME/waste_seg_experiments"; do
        if [ -n "$c" ] && [ -f "$c/experiments.tsv" ]; then HERE="$c"; break; fi
    done
fi
if [ ! -f "$HERE/experiments.tsv" ]; then
    echo "Cannot find the code: $0 is not inside waste_seg_experiments and neither is" >&2
    echo "SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) nor CODE_DIR (${CODE_DIR:-unset})." >&2
    echo "Run it from the code folder:  bash slurm/submit_all.sh" >&2
    exit 1
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "NOTE: this script submits the jobs, it is not one of them."
    echo "      Next time run it on the login node:  bash slurm/submit_all.sh"
    echo "      Carrying on; the job ids are below."
fi
export CONFIG="${CONFIG:-$HERE/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
cd "$HERE"
if ! mkdir -p "$LOG_DIR" "$TASKS" 2>/dev/null || [ ! -w "$LOG_DIR" ]; then
    echo "Cannot write to $LOG_DIR." >&2
    echo "Set WORK in $CONFIG to a folder you can write to, e.g. export WORK=\"\$HOME/waste_seg_work\"" >&2
    exit 1
fi
WHAT="${1:-all}"

OPTS=(--kill-on-invalid-dep=yes)
if [ -n "$PARTITION" ]; then OPTS+=(--partition="$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then OPTS+=(--account="$ACCOUNT"); fi

# submit_job <single|array> <dependency or ""> <sbatch args...>  -> prints the job id
submit_job() {
    local kind="$1" dep="$2"
    shift 2
    local args=("${OPTS[@]}")
    if [ "$kind" = "array" ]; then
        args+=(--output="$LOG_DIR/%x-%A_%a.out")
    else
        args+=(--output="$LOG_DIR/%x-%j.out")
    fi
    if [ -n "$dep" ]; then args+=(--dependency="$dep"); fi
    sbatch --parsable "${args[@]}" "$@" | cut -d';' -f1
}

# ---------------------------------------------------------------- preparation
NEED_PREPARE=0
[ -f "$FOLDS_REPORT/folds_summary.json" ] || NEED_PREPARE=1
[ -f "$PROTOCOL/protocol_summary.json" ] || NEED_PREPARE=1

if [ "$WHAT" = "pipeline" ]; then
    AFTER0=""   # the pipeline experiments do not need the folds
elif [ "$NEED_PREPARE" = "1" ]; then
    J0=$(submit_job single "" slurm/00_prepare.sbatch)
    echo "00 prepare                : $J0"
    AFTER0="afterok:$J0"
else
    echo "00 prepare                : skipped ($PROTOCOL already built)"
    AFTER0=""
fi

# ---------------------------------------------------------------- task lists
# The array sizes have to be known now, so the lists are refreshed here as well
# (the same command runs inside 00_prepare; it only reads the manifest and the run folders).
if [ "$WHAT" != "pipeline" ]; then
    if [ -f "$FOLDS_REPORT/common_test_fold0.txt" ]; then
        python list_tasks.py --manifest "$MANIFEST" --runs "$RUNS" --report "$FOLDS_REPORT" \
            --out "$TASKS" --seeds "$SEEDS" --maskrcnn-seeds "$MASKRCNN_SEEDS" --figure-id "$FIGURE_ID"
    elif [ ! -f "$TASKS/train_tasks.txt" ]; then
        echo
        echo "The task lists do not exist yet: $FOLDS_REPORT has not been built." >&2
        echo "Job $J0 builds it; run this script again when it finishes to submit the rest." >&2
        exit 0
    fi
    N_TRAIN=$(wc -l < "$TASKS/train_tasks.txt")
    N_EVAL=$(wc -l < "$TASKS/eval_tasks.txt")
fi

# ---------------------------------------------------------------- models
if [ "$WHAT" = "all" ] || [ "$WHAT" = "models" ] || [ "$WHAT" = "eval" ]; then
    DEP="$AFTER0"
    if [ "$WHAT" != "eval" ] && [ "$N_TRAIN" -gt 0 ]; then
        for ((k = 1; k <= CHAIN; k++)); do
            J=$(submit_job array "$DEP" --array=0-$((N_TRAIN - 1))%"$MAX_PARALLEL" --time="$TRAIN_TIME" \
                slurm/01_train.sbatch)
            echo "01 train ($k/$CHAIN)          : $J   ($N_TRAIN runs)"
            DEP="afterany:$J"
        done
    elif [ "$WHAT" != "eval" ]; then
        echo "01 train                  : nothing to train (no 'auto' rows in $MANIFEST)"
    fi
    J3=$(submit_job array "$DEP" --array=0-$((N_EVAL - 1))%"$MAX_PARALLEL" slurm/03_eval.sbatch)
    echo "03 evaluate               : $J3   ($N_EVAL runs)"
    J4=$(submit_job single "afterany:$J3" slurm/04_aggregate.sbatch)
    echo "04 aggregate + tables     : $J4"
fi

# ---------------------------------------------------------------- pipeline
if [ "$WHAT" = "all" ] || [ "$WHAT" = "pipeline" ]; then
    J5=$(submit_job array "$AFTER0" --array=0-$((PIPE_SHARDS - 1)) slurm/05_autoannotate.sbatch)
    echo "05 autoannotate           : $J5"
    J6=$(submit_job single "afterok:$J5" slurm/06_pipeline_eval.sbatch)
    echo "06 pipeline evaluation    : $J6"
fi
echo "logs: $LOG_DIR"
