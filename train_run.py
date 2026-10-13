#!/usr/bin/env python3
"""Train one run (configuration, fold, seed), dispatching on the kind in the manifest.

The SLURM array calls this once per line of train_tasks.txt. YOLO runs go through
train_yolo.py (which resumes from last.pt) and Mask R-CNN runs through maskrcnn.py
(which resumes from last.pth), so a task cut by the time limit is finished by
resubmitting the same array.

Example
-------
python train_run.py --id E19 --fold fold0 --seed 0 --protocol /scratch/protocol --runs /scratch/runs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import read_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    a, extra = ap.parse_known_args()

    exp = read_manifest(a.manifest).get(a.id)
    if exp is None:
        sys.exit(f"{a.id} is not in {a.manifest}")
    if exp["weights"] != "auto":
        sys.exit(f"{a.id} points at existing runs ({exp['weights']}); nothing to train")
    here = Path(__file__).resolve().parent
    run_dir = Path(a.runs) / a.id / f"{a.fold}_seed{a.seed}"
    data = Path(a.protocol) / f"{exp['corpus']}_{a.fold}.yaml"
    if not data.exists():
        sys.exit(f"Data YAML not found: {data} (run prepare_folds.py first)")

    if exp["kind"] == "yolo":
        cmd = [sys.executable, str(here / "train_yolo.py"), "--id", a.id, "--fold", a.fold, "--seed", str(a.seed),
               "--manifest", a.manifest, "--protocol", a.protocol, "--runs", a.runs, "--device", a.device,
               "--workers", str(a.workers)]
        if a.weights_dir:
            cmd += ["--weights-dir", a.weights_dir]
    elif exp["kind"] == "maskrcnn":
        dev = a.device if a.device == "cpu" or not str(a.device).isdigit() else f"cuda:{a.device}"
        cmd = [sys.executable, str(here / "maskrcnn.py"), "--data", str(data), "--out", str(run_dir),
               "--seed", str(a.seed), "--device", dev, "--workers", str(a.workers)]
    else:
        sys.exit(f"unknown kind '{exp['kind']}' for {a.id}")
    cmd += extra
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
