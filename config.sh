#!/bin/bash
# =============================================================================
#  Settings of the study. Every SLURM script sources this file.
#
#  PUT YOUR PATHS IN config.local.sh, NOT HERE: the last line of this file
#  sources config.local.sh if it exists, so your settings survive an update of
#  the code. Copy the lines you need to change into it, for example
#
#      cat > config.local.sh << 'EOF'
#      export FOLDS_ROOT="/data/$USER/exp-to-clean/folds"
#      export CLS_SRC="/data/$USER/classification"
#      export CLS_TEST="/data/$USER/classification_test"
#      export CLASSIFIER="/data/$USER/runs/cls/weights/best.pt"
#      activate_env() { source "$HOME/venvs/waste/bin/activate"; }
#      EOF
# =============================================================================

# ---- paths ------------------------------------------------------------------
export CODE_DIR="$HOME/waste_seg_experiments"          # this folder
export WORK="$HOME/waste_seg_work"                      # all outputs go here (~20 GB); any folder you can write to

# Folder holding the cross-validation datasets, one data.yaml per corpus and fold:
#   $FOLDS_ROOT/v1/fold0/data.yaml ... $FOLDS_ROOT/v3/fold4/data.yaml
export FOLDS_ROOT="/data/$USER/exp-to-clean/folds"
export FOLDS_RUNS="/data/$USER/exp-to-clean/runs"       # folder holding the runs already trained
                                                        # (used only by experiments.tsv, edit the paths there too)

export CLS_SRC="/path/to/classification_corpus"         # 120,667 images, <class>/.../*.jpg
export CLS_TEST="/path/to/classification_corpus_test"   # held-out split of the classifier, <class>/...
export CLASSIFIER="/path/to/classifier/weights/best.pt" # Ultralytics classification checkpoint
export BIREFNET="ZhengPeng7/BiRefNet"                   # leave as is: the checkpoint cited in the first version
                                                        # (downloaded automatically; fetch_weights.py for offline nodes)

# ---- pipeline parameters ----------------------------------------------------
# "auto": recovered from the released v2 corpus (calibrate_pipeline.py). Only if you know the
# values that were used to generate v2, write them instead of "auto".
export PIPE_CONF="auto"      # classifier confidence threshold
export PIPE_EPS="auto"       # RDP tolerance, fraction of the contour perimeter
export PIPE_MIN_AREA="auto"  # minimum contour area in pixels
export PIPE_MASK_THR="0.5"   # binarisation threshold of the BiRefNet map (standard value)
export PIPE_EPS_GRID="0.0005,0.001,0.0015,0.002,0.003,0.005,0.0075,0.01,0.02"
export PIPE_SHARDS=8         # parallel GPU tasks over the 120,667 source images

# ---- which experiments ------------------------------------------------------
export MANIFEST="$CODE_DIR/experiments.tsv"  # EDIT the run folders inside that file
export SEEDS="0"                 # seeds of the runs trained here (E19, E20); "0 1 2" for three
export MASKRCNN_SEEDS="0"        # seeds of the Mask R-CNN baseline (E18)
export FIGURE_ID="E13"           # configuration whose detections are kept for the qualitative figure
export BOOTSTRAP=1000            # bootstrap resamples of the per-image confidence intervals

# ---- SLURM ------------------------------------------------------------------
export PARTITION=""          # empty = cluster default
export ACCOUNT=""            # empty = none
export TRAIN_TIME="3-00:00:00"
export CHAIN=3               # submissions per training array; later ones resume runs cut by TRAIN_TIME
export MAX_PARALLEL=8        # at most this many array tasks at a time (the %N of --array)

# ---- software environment ---------------------------------------------------
activate_env() {
    # module load CUDA/12.1          # adapt to your cluster, or delete
    # shellcheck source=/dev/null
    source "$HOME/venvs/waste/bin/activate"
    # conda alternative (e.g. the environment of the original runs):
    #   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate reciclai
}

# ---- advanced: leave as is for the paper runs (the smoke test changes them) --
export DEVICE="0"                 # GPU index inside the job
export EXTRA_TRAIN_ARGS=""        # e.g. "--epochs 2 --imgsz 128" (smoke test only)
export EXTRA_MASKRCNN_ARGS=""     # extra arguments of the Mask R-CNN baseline only
export EXTRA_EVAL_ARGS=""
export EXTRA_AUTOANN_ARGS=""

# ---- derived (normally no need to edit) ------------------------------------
export PROTOCOL="$WORK/protocol"          # per-fold YAMLs and control corpora
export FOLDS_REPORT="$WORK/folds_report"  # check_folds.py: common test sets, origin of each image
export TASKS="$WORK/tasks"                # train_tasks.txt and eval_tasks.txt
export RUNS="$WORK/runs"                  # the runs trained here
export RESULTS="$WORK/results"
export WEIGHTS_DIR="$WORK/weights"
export LOG_DIR="$WORK/logs"
export PAPER_TABLES="$WORK/paper_tables"  # copy this folder to <latex project>/tables
export NAMES_YAML="${NAMES_YAML:-$FOLDS_ROOT/v3/fold0/data.yaml}"   # any YAML with the class names
# Any YAML of a corpus gives the whole corpus (its three splits together), which is what the
# pipeline calibration and the corpus statistics need.
export V1_YAML="${V1_YAML:-$FOLDS_ROOT/v1/fold0/data.yaml}"   # the public aggregation
export V2_YAML="${V2_YAML:-$FOLDS_ROOT/v2/fold0/data.yaml}"   # the generated corpus (pipeline calibration)
export V3_YAML="${V3_YAML:-$FOLDS_ROOT/v3/fold0/data.yaml}"   # the combined corpus
export TORCH_HOME="$WORK/torch_home"
export HF_HOME="$WORK/hf_home"
export PYTHONUNBUFFERED=1

# ---- your own settings ------------------------------------------------------
# Kept in a separate file so that unpacking a new version of the code does not
# overwrite them. Everything above can be overridden there.
# shellcheck source=/dev/null
[ -f "${CODE_DIR}/config.local.sh" ] && source "${CODE_DIR}/config.local.sh"

# ---- a last look ------------------------------------------------------------
for v in FOLDS_ROOT CLS_SRC CLS_TEST CLASSIFIER; do
    case "${!v}" in
        */path/to/*) echo "NOTE: $v is still the example path (${!v}); set it in $CODE_DIR/config.local.sh" >&2 ;;
    esac
done
