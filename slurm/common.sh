#!/bin/bash
# Sourced by every job after config.sh: activates the environment and moves to the code folder.
activate_env
cd "$CODE_DIR" || exit 1
mkdir -p "$PROTOCOL" "$RUNS" "$RESULTS" "$WEIGHTS_DIR" "$LOG_DIR" "$PAPER_TABLES"
echo "job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-}) on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# GPU jobs: stop at once, with a clear message, if PyTorch cannot use the allocated GPU
if [ "${DEVICE:-0}" != "cpu" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "ERROR: a GPU is allocated but PyTorch cannot use it (PyTorch built for a newer CUDA than the driver?)." >&2
        python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)" >&2 || true
        nvidia-smi 2>/dev/null | head -4 >&2 || true
        echo "See README: 'PyTorch and the NVIDIA driver'." >&2
        exit 1
    fi
fi
