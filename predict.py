#!/usr/bin/env python3
"""Run one model of the manifest on the evaluation images and save COCO-format mask detections.

All models see the same images, loaded with OpenCV and resized to a long side of
640 px; YOLO models run with conf=0.001, iou=0.7, max_det=100, retina masks;
Mask R-CNN with score threshold 0.001 and 100 detections. Also records model cost
(parameters, GFLOPs for one 640x640 input, checkpoint size, latency at batch 1)
and training progress (epochs run, best epoch, how the run ended).

Example
-------
python predict.py --id E13 --manifest experiments.tsv --runs /scratch/runs \
    --gt /scratch/results/gt_v3_test.json.gz --out /scratch/results/predictions
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from cocoutils import load_json_gz, masks_to_detections, save_json_gz
from common import environment_info, experiment_weights, load_canonical, load_yaml, read_csv, read_manifest, torch_load, write_json


def count_gflops(model, inputs) -> float:
    """FLOPs of one forward pass (multiply and add counted separately), via torch.utils.flop_counter."""
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(inputs)
        return round(fc.get_total_flops() / 1e9, 2)
    except Exception as e:
        print(f"  [GFLOPs not available: {e}]")
        return float("nan")


def yolo_training_progress(weights: Path) -> dict:
    run_dir = weights.parent.parent
    out = {}
    try:
        rows = read_csv(run_dir / "results.csv")
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        fit = []
        for r in rows:
            f = 0.0
            for s in ("B", "M"):
                f += 0.1 * float(r.get(f"metrics/mAP50({s})", 0) or 0) + 0.9 * float(r.get(f"metrics/mAP50-95({s})", 0) or 0)
            fit.append(f)
        out["epochs_run"] = len(rows)
        out["best_epoch"] = int(np.argmax(fit)) + 1 if fit else None
    except Exception as e:
        out["progress_note"] = f"results.csv not readable: {e}"
    try:
        out["epochs_configured"] = int(load_yaml(run_dir / "args.yaml").get("epochs"))
    except Exception:
        pass
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        try:
            ep = torch_load(last).get("epoch", -1)
            finished = ep == -1
            if not finished:
                out["end"] = "interrupted"
            elif out.get("epochs_run") is not None and out.get("epochs_configured") is not None \
                    and out["epochs_run"] < out["epochs_configured"]:
                out["end"] = "early_stop"
            else:
                out["end"] = "complete"
        except Exception as e:
            out["progress_note"] = f"last.pt not readable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gt", required=True, help="COCO GT written by build_coco_gt.py")
    ap.add_argument("--out", required=True, help="directory for <id>.json.gz and <id>.meta.json")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: GPU if available)")
    ap.add_argument("--cpu-latency", type=int, default=0, help="also time N images on CPU")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    a = ap.parse_args()

    import torch
    exp = read_manifest(a.manifest)[a.id]
    weights = experiment_weights(exp, a.runs)
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found for {a.id}: {weights}")
    gt = load_json_gz(a.gt)
    images = gt["images"][: a.limit] if a.limit else gt["images"]
    dev = a.device if a.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    tdev = torch.device("cpu" if dev == "cpu" else f"cuda:{dev}" if dev.isdigit() else dev)
    meta = {"id": a.id, "kind": exp["kind"], "weights": str(weights), "n_images": len(images),
            "size_mb": round(weights.stat().st_size / 1e6, 1), "environment": environment_info()}

    dets, times, shape_mismatch = [], [], 0
    # Ultralytics creates an (empty) run folder for predictions; keep it out of the code folder
    quiet = {"project": str(Path(a.out).resolve() / "_ultralytics"), "name": a.id, "exist_ok": True}
    if exp["kind"] == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(weights))
        for info in images:
            im = load_canonical(info["file_name"])
            if im.shape[:2] != (info["height"], info["width"]):
                shape_mismatch += 1
            r = model.predict(im, imgsz=640, conf=0.001, iou=0.7, max_det=100, retina_masks=True,
                              verbose=False, device=dev, **quiet)[0]
            times.append(sum(r.speed.values()))
            if r.masks is None or len(r.boxes) == 0:
                continue
            masks = r.masks.data.cpu().numpy() > 0.5
            dets += masks_to_detections(info["id"], masks, r.boxes.cls.cpu().numpy().astype(int),
                                        r.boxes.conf.cpu().numpy(), (info["height"], info["width"]))
        net = copy.deepcopy(model.model).float().eval().cpu()
        meta["params"] = int(sum(p.numel() for p in net.parameters()))
        meta["gflops"] = count_gflops(net, torch.zeros(1, 3, 640, 640))
        try:
            meta["gflops_ultralytics"] = round(float(model.info(verbose=False)[3]), 2)
        except Exception:
            pass
        meta.update(yolo_training_progress(weights))
        if a.cpu_latency:
            torch.set_num_threads(a.cpu_threads)
            ct = []
            for info in images[: a.cpu_latency + 5]:
                r = model.predict(load_canonical(info["file_name"]), imgsz=640, conf=0.001, iou=0.7, max_det=100,
                                  retina_masks=True, verbose=False, device="cpu", **quiet)[0]
                ct.append(sum(r.speed.values()))
            meta["latency_cpu_ms_median"] = round(float(np.median(ct[5:] or ct)), 1)
    elif exp["kind"] == "maskrcnn":
        from maskrcnn import load_for_inference, predict_image
        model, ck = load_for_inference(weights, tdev)
        for info in images:
            im = load_canonical(info["file_name"])
            if im.shape[:2] != (info["height"], info["width"]):
                shape_mismatch += 1
            if tdev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            masks, cls, scores = predict_image(model, im, tdev)
            if tdev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            dets += masks_to_detections(info["id"], masks, cls, scores, (info["height"], info["width"]))
        meta["params"] = int(sum(p.numel() for p in model.parameters()))
        x0 = load_canonical(images[0]["file_name"])[:, :, ::-1]
        x0 = torch.from_numpy(np.ascontiguousarray(x0)).permute(2, 0, 1).float().div(255)
        meta["gflops"] = count_gflops(copy.deepcopy(model).cpu().eval(), [x0])
        run_dir = weights.parent
        try:
            hist = load_json_gz(run_dir / "history.json")
            meta["epochs_run"] = len(hist)
            meta["best_epoch"] = ck.get("epoch")
            meta["end"] = "complete" if torch_load(run_dir / "last.pth").get("finished") else "interrupted"
        except Exception:
            pass
        if a.cpu_latency:
            torch.set_num_threads(a.cpu_threads)
            model = model.cpu()
            ct = []
            for info in images[: a.cpu_latency + 5]:
                im = load_canonical(info["file_name"])
                t0 = time.perf_counter()
                predict_image(model, im, torch.device("cpu"))
                ct.append((time.perf_counter() - t0) * 1000)
            meta["latency_cpu_ms_median"] = round(float(np.median(ct[5:] or ct)), 1)
    else:
        raise SystemExit(f"Unknown kind {exp['kind']}")

    meta["latency_ms_median"] = round(float(np.median(times[10:] or times)), 2) if times else None
    meta["latency_device"] = str(tdev) if exp["kind"] == "maskrcnn" else dev
    meta["n_detections"] = len(dets)
    meta["canonical_shape_mismatches"] = shape_mismatch
    out = Path(a.out)
    save_json_gz(dets, out / f"{a.id}.json.gz")
    write_json(out / f"{a.id}.meta.json", meta)
    print({k: v for k, v in meta.items() if k != "environment"})


if __name__ == "__main__":
    main()
