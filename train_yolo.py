#!/usr/bin/env python3
"""Train (or resume) one YOLO segmentation experiment from experiments.tsv.

The configuration is the one of the paper (Table VI): framework defaults except
epochs=300, i.e. imgsz 640, batch 16, optimizer 'auto', seed 0, deterministic,
patience 100. Extra per-experiment arguments (e.g. copy_paste=0.3) come from the
manifest.

Re-running the same command is always safe:
  * no checkpoint yet          -> starts training
  * interrupted (time limit)   -> resumes from runs/<id>/weights/last.pt
  * finished (or early-stopped)-> does nothing
so a job killed by the queue limit is continued simply by submitting it again.

Example
-------
python train_yolo.py --id E3 --manifest experiments.tsv --protocol /scratch/protocol --runs /scratch/runs
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from common import environment_info, load_yaml, read_manifest, split_images, torch_load, write_json

PAPER_CONFIG = dict(epochs=300, imgsz=640, batch=16, deterministic=True, patience=100)  # seed comes from --seed


def run_state(last: Path) -> str:
    """'new', 'resume' or 'finished'."""
    if not last.exists():
        return "new"
    try:
        ck = torch_load(last)
    except Exception as e:
        print(f"WARNING: cannot read {last} ({e}); starting from scratch.")
        return "new"
    ep = ck.get("epoch", -1) if isinstance(ck, dict) else -1
    return "finished" if ep == -1 else "resume"


def expected_auto_optimizer(n_train: int, batch: int, epochs: int, nc: int, nbs: int = 64) -> dict:
    """What Ultralytics' optimizer='auto' rule resolves to (engine/trainer.py, build_optimizer)."""
    iterations = math.ceil(n_train / max(batch, nbs)) * epochs
    if iterations > 10000:
        return {"optimizer": "SGD", "lr0": 0.01, "momentum": 0.9, "iterations": iterations, "warmup_bias_lr": 0.0}
    return {"optimizer": "AdamW", "lr0": round(0.002 * 5 / (4 + nc), 6), "momentum": 0.9,
            "iterations": iterations, "warmup_bias_lr": 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="experiment id in the manifest, e.g. E19")
    ap.add_argument("--fold", required=True, help="fold name, e.g. fold0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--protocol", required=True, help="directory with the data YAMLs written by prepare_protocol.py")
    ap.add_argument("--runs", required=True, help="output directory; the run goes to <runs>/<id>")
    ap.add_argument("--weights-dir", default=None, help="directory holding pre-downloaded *.pt (offline nodes)")
    ap.add_argument("--device", default=None, help="e.g. 0 (default: Ultralytics picks)")
    ap.add_argument("--workers", type=int, default=None, help="dataloader workers (default: Ultralytics default, 8)")
    # overrides for the smoke test only; leave unset for the paper runs
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    a = ap.parse_args()

    exps = read_manifest(a.manifest)
    if a.id not in exps:
        sys.exit(f"{a.id} not in {a.manifest}")
    exp = exps[a.id]
    if exp["kind"] != "yolo":
        sys.exit(f"{a.id} is of kind '{exp['kind']}', use the matching training script.")
    if exp["weights"] != "auto":
        print(f"{a.id} uses an existing checkpoint ({exp['weights']}); nothing to train.")
        return

    runs = Path(a.runs).resolve()
    name = f"{a.fold}_seed{a.seed}"
    run_dir = runs / a.id / name
    last = run_dir / "weights" / "last.pt"
    data_yaml = (Path(a.protocol) / f"{exp['corpus']}_{a.fold}.yaml").resolve()
    if not data_yaml.exists():
        sys.exit(f"Data YAML not found: {data_yaml} (run prepare_protocol.py first)")

    model_src = exp["model"]
    if a.weights_dir:
        wd = Path(a.weights_dir).resolve()
        if (wd / model_src).exists():
            model_src = str(wd / model_src)
        os.chdir(wd)  # Ultralytics looks here for yolo11n.pt during its AMP check (offline nodes)

    from ultralytics import YOLO

    state = run_state(last)
    print(f"[{a.id} {name}] corpus={exp['corpus']} model={exp['model']} state={state}", flush=True)
    if state == "finished":
        print(f"[{a.id} {name}] already finished: {last}")
        return

    resolved = {}

    def on_train_start(trainer):
        try:
            opt = trainer.optimizer
            g = opt.param_groups[0]
            resolved.update({"optimizer": type(opt).__name__,
                             "lr0": g.get("initial_lr", g.get("lr")),
                             "momentum": g.get("momentum", g.get("betas", [None])[0]),
                             "weight_decay_group0": g.get("weight_decay"),
                             "warmup_bias_lr": getattr(trainer.args, "warmup_bias_lr", None),
                             "n_train_images": len(trainer.train_loader.dataset)})
            write_json(run_dir / "optimizer_resolved.json", resolved)
        except Exception as e:  # never let bookkeeping break a training run
            print(f"WARNING: could not record optimizer: {e}")

    info = {"id": a.id, "fold": a.fold, "seed": a.seed, "manifest_entry": exp, "data": str(data_yaml), "state_at_start": state,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": environment_info()}
    t0 = time.time()
    if state == "resume":
        model = YOLO(str(last))
        model.add_callback("on_train_start", on_train_start)
        kw = {"resume": True}
        if a.device is not None:
            kw["device"] = a.device
        model.train(**kw)
    else:
        cfg = dict(PAPER_CONFIG)
        for k in ("epochs", "batch", "imgsz"):
            if getattr(a, k) is not None:
                cfg[k] = getattr(a, k)
        cfg.update(exp["args"])
        cfg.update(data=str(data_yaml), project=str(runs / a.id), name=name, exist_ok=True, seed=a.seed)
        if a.device is not None:
            cfg["device"] = a.device
        if a.workers is not None:
            cfg["workers"] = a.workers
        info["train_args"] = cfg
        try:
            nc = len(load_yaml(data_yaml)["names"])
            n_train = len(split_images(data_yaml)["train"])
            info["expected_auto_optimizer"] = expected_auto_optimizer(n_train, cfg["batch"], cfg["epochs"], nc)
        except Exception:
            pass
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_info_start.json", info)
        model = YOLO(model_src)
        model.add_callback("on_train_start", on_train_start)
        model.train(**cfg)

    info["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["wall_clock_hours_this_job"] = round((time.time() - t0) / 3600, 3)
    info["state_at_end"] = run_state(last)
    info["resolved_optimizer"] = resolved
    stamp = time.strftime("%Y%m%d-%H%M%S")
    write_json(run_dir / f"run_info_{stamp}.json", info)
    print(f"[{a.id} {name}] done in {info['wall_clock_hours_this_job']} h, state: {info['state_at_end']}")


if __name__ == "__main__":
    main()
