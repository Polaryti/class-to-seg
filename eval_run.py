#!/usr/bin/env python3
"""Score one run (one configuration, one fold, one seed) on the common test set of its fold.

Predicts and evaluates in the same step, so the detections of ~300 runs never have to be
stored. Writes one JSON per run in <results>/per_run/, with the metrics on the common test
set and on its public-origin (v1) and generated-origin (v2) halves, the per-class AP, the
training progress of the run and, with --gflops, the cost of the model.

Example
-------
python eval_run.py --id E13 --fold fold0 --seed 0 --manifest experiments.tsv \
    --runs /scratch/runs --results /scratch/results --report /scratch/folds_report
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import time
from pathlib import Path

import numpy as np

from cocoutils import (coco_from_dict, load_json_gz, masks_to_detections, prepare_arrays, run_cocoeval, summarise,
                       weighted_metrics)
from common import environment_info, load_canonical, read_csv, torch_load, write_json
from folds import enumerate_runs
from predict import count_gflops, yolo_training_progress


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--report", required=True, help="output folder of check_folds.py (origin.csv)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gflops", action="store_true", help="also measure parameters and GFLOPs (slow, once per configuration)")
    ap.add_argument("--save-dets", action="store_true", help="keep the detections (for the figures)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; 640 is the size the study uses and the size of the ground truth "
                         "(masks are returned at the image resolution either way)")
    a = ap.parse_args()

    import torch
    runs = [r for r in enumerate_runs(a.manifest, a.runs, folds=[a.fold], seeds=[a.seed]) if r["id"] == a.id]
    if not runs:
        raise SystemExit(f"no run {a.id} {a.fold} seed {a.seed} (check the manifest and the run folders)")
    run = runs[0]
    if not run["exists"]:
        raise SystemExit(f"checkpoint not found: {run['weights']}")
    res = Path(a.results)
    gt = load_json_gz(res / f"gt_common_{a.fold}.json.gz")
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    images = gt["images"][: a.limit] if a.limit else gt["images"]
    dev = a.device if a.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    tdev = torch.device("cpu" if dev == "cpu" else f"cuda:{dev}" if str(dev).isdigit() else dev)

    t0 = time.time()
    dets, times = [], []
    if run["kind"] == "yolo":
        from ultralytics import YOLO
        model = YOLO(run["weights"])
        quiet = {"project": str(res / "_ultralytics"), "name": f"{a.id}_{a.fold}_{a.seed}", "exist_ok": True}
        for info in images:
            im = load_canonical(info["file_name"])
            r = model.predict(im, imgsz=a.imgsz, conf=0.001, iou=0.7, max_det=100, retina_masks=True,
                              verbose=False, device=dev, **quiet)[0]
            times.append(sum(r.speed.values()))
            if r.masks is None or len(r.boxes) == 0:
                continue
            dets += masks_to_detections(info["id"], r.masks.data.cpu().numpy() > 0.5,
                                        r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(),
                                        (info["height"], info["width"]))
        cost = {}
        if a.gflops:
            net = copy.deepcopy(model.model).float().eval().cpu()
            cost = {"params": int(sum(p.numel() for p in net.parameters())),
                    "gflops": count_gflops(net, torch.zeros(1, 3, a.imgsz, a.imgsz))}
        progress = yolo_training_progress(Path(run["weights"]))
    else:
        from maskrcnn import load_for_inference, predict_image
        model, ck = load_for_inference(run["weights"], tdev)
        for info in images:
            im = load_canonical(info["file_name"])
            if tdev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            masks, cls, scores = predict_image(model, im, tdev)
            if tdev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
            dets += masks_to_detections(info["id"], masks, cls, scores, (info["height"], info["width"]))
        cost = {"params": int(sum(p.numel() for p in model.parameters()))}
        if a.gflops:
            x = load_canonical(images[0]["file_name"])[:, :, ::-1]
            x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255)
            cost["gflops"] = count_gflops(copy.deepcopy(model).cpu().eval(), [x])
        budget = ck.get("epochs_configured")
        last = Path(run["run_dir"]) / "last.pth"
        state = torch_load(last) if (budget is None or "history" not in ck) and last.exists() else {}
        progress = {"epochs_run": len(ck.get("history") or state.get("history") or []) or None,
                    "best_epoch": ck.get("epoch"),
                    "epochs_configured": budget or state.get("epochs_configured")}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = coco_from_dict(gt)
        E = run_cocoeval(coco_gt, dets, [im["id"] for im in images])
    overall, per_class = summarise(E, names)
    out = {"id": a.id, "fold": a.fold, "seed": a.seed, "kind": run["kind"], "corpus": run["corpus"],
           "model": run["model"], "weights": run["weights"], "run_dir": run["run_dir"],
           "n_images": len(images), "n_detections": len(dets),
           "latency_ms_median": round(float(np.median(times[10:] or times)), 2) if times else None,
           "size_mb": round(Path(run["weights"]).stat().st_size / 1e6, 1), **cost, **progress,
           "eval_seconds": round(time.time() - t0, 1), "environment": environment_info(),
           "common": overall, "per_class": {r["class"]: r for r in per_class}}

    if E is not None:
        arr = prepare_arrays(E)
        origin = {r["image"]: r["origin"] for r in read_csv(Path(a.report) / "origin.csv")}
        orig = [origin.get(Path(im["file_name"]).name, "unknown") for im in images]
        for o in ("v1", "v2"):
            w = np.array([1.0 if x == o else 0.0 for x in orig])
            if w.sum() == 0:
                continue
            wm = weighted_metrics(arr, w)
            out[f"{o}_origin"] = {k: wm[k] for k in ("mAP50_95", "mAP50", "mAP75", "P", "R", "F1")}
            out[f"{o}_origin"]["images"] = int(w.sum())
            out[f"{o}_origin"]["per_class_AP"] = {names[k]: v for k, v in wm["per_class_AP"].items()}
        del arr

    write_json(res / "per_run" / f"{a.id}__{a.fold}_seed{a.seed}.json", out)
    if a.save_dets:
        from cocoutils import save_json_gz
        save_json_gz(dets, res / "predictions" / f"{a.id}__{a.fold}_seed{a.seed}.json.gz")
    c = out["common"]
    print(f"{a.id} {a.fold} seed{a.seed}: mAP50-95 {c['mAP50_95']:.4f}  mAP50 {c['mAP50']:.4f}  "
          f"P {c['P']:.4f}  R {c['R']:.4f}  ({out['eval_seconds']} s)")


if __name__ == "__main__":
    main()
