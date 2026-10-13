#!/bin/bash
#SBATCH --job-name=ws-resume-orig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
# OPTIONAL. Continues an ORIGINAL run that the queue limit interrupted from its own last.pt,
# in its own folder, with its own arguments. One run (one fold, one seed) per submission:
#   sbatch slurm/07_resume_original.sbatch /path/to/runs/E14/fold0_seed0/weights/last.pt
# Its best.pt is then updated in place; re-run the evaluation task of that run afterwards
# (03_eval.sbatch with the array index of its line in $TASKS/eval_tasks.txt).
# The data path stored in the checkpoint must still exist.
set -eo pipefail
CONFIG="${CONFIG:-$SLURM_SUBMIT_DIR/config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
# shellcheck source=/dev/null
source "$CODE_DIR/slurm/common.sh"

LAST="$1"
if [ -z "$LAST" ] || [ ! -f "$LAST" ]; then echo "usage: sbatch $0 /path/to/last.pt" >&2; exit 1; fi
python - "$LAST" << 'PY'
import sys
from common import torch_load
from ultralytics import YOLO
last = sys.argv[1]
ep = torch_load(last).get("epoch", -1)
if ep == -1:
    print(f"{last}: training already finished, nothing to resume")
else:
    print(f"{last}: resuming after epoch {ep + 1}")
    YOLO(last).train(resume=True)
PY
