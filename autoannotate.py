#!/usr/bin/env python3
"""Reference implementation of the annotation pipeline (Algorithm 1) with full logging.

For every image of a classification corpus laid out as <src>/<class>/.../*.jpg:
  1. EXIF orientation is applied;
  2. BiRefNet produces a foreground probability map (input resized to --birefnet-size);
  3. the classifier predicts a class and a confidence;
  4. the map is resized to the original size and binarised at --mask-thr;
  5. external contours are extracted; for every contour its area and the number of
     vertices after Ramer-Douglas-Peucker simplification are logged for each
     epsilon in --eps-grid (fraction of the contour perimeter).
One JSON line per image is written to <out>/stats/shard_XXX.jsonl, so the effect of
the confidence threshold, of epsilon and of the minimum contour area can be
studied afterwards without re-running the networks (pipeline_sweep.py).

With --write, images whose confidence is >= --conf are also written as a YOLO
segmentation corpus (<out>/dataset/images, <out>/dataset/labels) using --eps and
--min-area, i.e. the pipeline itself.

Runs are resumable (already logged images are skipped) and can be split into
shards for a SLURM array (--shard i --num-shards n).

Example
-------
python autoannotate.py --src /data/cls_corpus --names /scratch/protocol/v3.yaml \
    --classifier /models/classifier.pt --out /scratch/pipeline --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from common import IMG_EXTS, class_names, load_yaml, write_label


def norm_name(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).lower())


def human_label(path: Path, src: Path, names: list[str]) -> str:
    """Class name from the first directory component of the path that matches a class name."""
    lookup = {norm_name(n): n for n in names}
    for part in path.relative_to(src).parts[:-1]:
        if norm_name(part) in lookup:
            return lookup[norm_name(part)]
    return "unknown"


def list_source_images(src: Path) -> list[Path]:
    files = [Path(r) / f for r, _, fs in os.walk(src, followlinks=True) for f in fs]
    return sorted(f for f in files if f.suffix.lower() in IMG_EXTS)


# ----------------------------------------------------------------------------
class Classifier:
    """Ultralytics classification checkpoint. Replace this class to use another classifier."""

    def __init__(self, weights: str, names: list[str], device: str, imgsz: int | None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.imgsz = device, imgsz
        ta = getattr(self.model, "ckpt", None) or {}
        ta = ta.get("train_args", {}) if isinstance(ta, dict) else {}
        self.info = {"weights": str(weights),
                     "architecture": Path(str(ta.get("model", ""))).stem or type(self.model.model).__name__,
                     "imgsz": imgsz or ta.get("imgsz"),
                     "ultralytics": getattr(self.model, "ckpt", {}).get("version") if isinstance(
                         getattr(self.model, "ckpt", None), dict) else None}
        cls_names = self.model.names
        lookup = {norm_name(n): i for i, n in enumerate(names)}
        self.map = {}
        missing = []
        for k, n in cls_names.items():
            if norm_name(n) in lookup:
                self.map[int(k)] = lookup[norm_name(n)]
            else:
                missing.append(n)
        if missing:
            sys.exit(f"Classifier classes {missing} do not match the target names {names}.")

    def __call__(self, pil_img):
        kw = {"verbose": False, "device": self.device, "exist_ok": True, "name": "cls",
              "project": os.path.join(tempfile.gettempdir(), f"ultralytics_{os.getpid()}")}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        r = self.model.predict(pil_img, **kw)[0]
        probs = r.probs.data.cpu().numpy()
        order = np.argsort(-probs)
        return self.map[int(order[0])], float(probs[order[0]]), self.map[int(order[1])], float(probs[order[1]])


class BiRefNetSegmenter:
    def __init__(self, repo: str, device: str, size: int, half: bool):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        self.torch = torch
        self.dev = torch.device("cpu" if device == "cpu" else f"cuda:{device}" if str(device).isdigit() else device)
        torch.set_float32_matmul_precision("high")
        self.half = half and self.dev.type == "cuda"
        model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
        # the published weights are stored in fp16; set the dtype explicitly instead of inheriting it
        self.model = (model.half() if self.half else model.float()).to(self.dev).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, pil_img) -> np.ndarray:
        x = self.tf(pil_img).unsqueeze(0).to(self.dev, dtype=self.dtype)
        with self.torch.no_grad():
            pred = self.model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
        return pred


class OtsuSegmenter:
    """Stand-in used only by the smoke test (no network download needed)."""

    def __call__(self, pil_img) -> np.ndarray:
        g = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2GRAY)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (m > 0).astype(np.float32)


# ----------------------------------------------------------------------------
def contours_from_prob(prob: np.ndarray, w: int, h: int, thr: float):
    prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob >= thr).astype(np.uint8)
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    return cnts, float(binary.mean())


def simplify(c, eps: float):
    return cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="classification corpus: <src>/<class>/.../*.jpg")
    ap.add_argument("--names", required=True, help="data YAML whose 'names' define the output class ids")
    ap.add_argument("--classifier", required=True, help="Ultralytics classification weights (.pt)")
    ap.add_argument("--cls-imgsz", type=int, default=None, help="classifier input size (default: as trained)")
    ap.add_argument("--segmenter", choices=["birefnet", "otsu"], default="birefnet")
    ap.add_argument("--birefnet", default="ZhengPeng7/BiRefNet", help="Hugging Face id or local directory")
    ap.add_argument("--birefnet-size", type=int, default=1024)
    ap.add_argument("--half", action="store_true", help="fp16 BiRefNet inference")
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--eps-grid", default="0.0005,0.001,0.0015,0.002,0.003,0.005,0.0075,0.01,0.02")
    ap.add_argument("--conf", type=float, default=None, help="confidence threshold (required with --write)")
    ap.add_argument("--eps", type=float, default=None, help="RDP epsilon for --write (fraction of perimeter)")
    ap.add_argument("--min-area", type=float, default=None, help="minimum contour area in px for --write")
    ap.add_argument("--write", action="store_true", help="also write the YOLO corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    if a.write and None in (a.conf, a.eps, a.min_area):
        sys.exit("--write needs --conf, --eps and --min-area (the values used in the paper).")

    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    names = class_names(load_yaml(a.names))
    eps_grid = [float(x) for x in a.eps_grid.split(",")]
    files = list_source_images(src)[a.shard::a.num_shards]
    if a.limit:
        files = files[: a.limit]
    stats_file = out / "stats" / f"shard_{a.shard:03d}.jsonl"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if stats_file.exists():
        with open(stats_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:
                    pass
    todo = [p for p in files if str(p) not in done]
    print(f"shard {a.shard}/{a.num_shards}: {len(files)} images, {len(done)} already done, {len(todo)} to process", flush=True)
    meta = {"src": str(src), "names": names, "segmenter": a.segmenter, "birefnet": a.birefnet,
            "birefnet_size": a.birefnet_size, "mask_thr": a.mask_thr, "eps_grid": eps_grid,
            "classifier": a.classifier, "cls_imgsz": a.cls_imgsz, "conf": a.conf, "eps": a.eps, "min_area": a.min_area}
    (out / "stats" / "meta.json").write_text(json.dumps(meta, indent=2))
    if not todo:
        return

    clf = Classifier(a.classifier, names, a.device, a.cls_imgsz)
    # The human image-level label is read from the directory a source image sits in. If no
    # directory matches a class name the agreement columns of the pipeline tables stay empty,
    # which is worth saying now rather than after the whole corpus has been processed.
    if todo:
        probe = [human_label(p, src, names) for p in todo[:200]]
        if all(h == "unknown" for h in probe):
            print(f"WARNING: none of the first {len(probe)} images is under a directory named after a class.\n"
                  f"  --src is {src}\n"
                  f"  classes are {names}\n"
                  f"  the layout has to be <src>/<class>/.../*.jpg (any depth), or the label agreement of\n"
                  f"  Table XIV cannot be computed. Point --src at the labelled corpus, or rename its folders.",
                  flush=True)
    seg = BiRefNetSegmenter(a.birefnet, a.device, a.birefnet_size, a.half) if a.segmenter == "birefnet" else OtsuSegmenter()
    with open(stats_file, "a") as fout:
        for k, p in enumerate(todo):
            try:
                with Image.open(p) as im0:
                    im = ImageOps.exif_transpose(im0).convert("RGB")
            except Exception as e:
                print(f"skip unreadable {p}: {e}")
                continue
            w, h = im.size
            pred, conf, pred2, conf2 = clf(im)
            prob = seg(im)
            cnts, fg = contours_from_prob(prob, w, h, a.mask_thr)
            rows = []
            for c in cnts:
                area = float(cv2.contourArea(c))
                rows.append([round(area, 1)] + [int(len(simplify(c, e))) for e in eps_grid])
            rec = {"path": str(p), "human": human_label(p, src, names), "pred": names[pred], "conf": round(conf, 5),
                   "pred2": names[pred2], "conf2": round(conf2, 5), "w": w, "h": h, "fg": round(fg, 5),
                   "contours": rows}
            fout.write(json.dumps(rec) + "\n")
            if a.write and conf >= a.conf:
                inst = []
                for c in cnts:
                    if cv2.contourArea(c) < a.min_area:
                        continue
                    poly = simplify(c, a.eps)
                    if len(poly) >= 3:
                        inst.append((pred, poly / np.array([w, h], dtype=np.float64)))
                if inst:
                    rel = p.relative_to(src)
                    stem = "__".join(rel.with_suffix("").parts)
                    img_out = out / "dataset" / "images" / f"{stem}.jpg"
                    img_out.parent.mkdir(parents=True, exist_ok=True)
                    im.save(img_out, quality=95)
                    write_label(out / "dataset" / "labels" / f"{stem}.txt", inst)
            if k % 500 == 0:
                fout.flush()
                print(f"  {k}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
