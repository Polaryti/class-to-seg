#!/usr/bin/env python3
"""Draw a stratified sample of generated annotations for a manual audit.

Creates in --out:
  images/                  the sampled images (copies)
  generated_labels/        their generated YOLO polygons
  overlays/                images with the generated polygons and class drawn on them
  manual_labels/           EMPTY: put here the manual YOLO-segmentation labels (same file stems),
                           drawn from scratch without looking at the generated polygons
  manifest.csv             one row per image; fill label_correct with 1/0 while looking at overlays/
  classes.txt              class id -> name, for the annotation tool

Then run audit_score.py.

Example
-------
python audit_sample.py --data /scratch/protocol/v2.yaml --per-class 30 --out /scratch/audit
"""
import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import class_names, label_path, load_yaml, read_label, split_images, write_csv

PALETTE = [(56, 56, 255), (255, 112, 31), (0, 128, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data YAML of the generated corpus (v2)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = class_names(load_yaml(a.data))
    imgs = [p for s, v in split_images(a.data).items() if s in a.splits.split(",") for p in v]
    by_cls = defaultdict(list)
    for p in imgs:
        inst = read_label(label_path(p))
        if inst:
            cnt = Counter(c for c, _, _ in inst)
            by_cls[min(cnt, key=lambda c: (-cnt[c], c))].append(p)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    for d in ("images", "generated_labels", "overlays", "manual_labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(f"{i} {n}" for i, n in enumerate(names)) + "\n")
    rows = []
    for c in sorted(by_cls):
        pool = by_cls[c]
        pick = rng.choice(len(pool), size=min(a.per_class, len(pool)), replace=False)
        for i in sorted(pick):
            p = pool[i]
            stem = f"{names[c]}__{p.stem}"
            dst = out / "images" / f"{stem}{p.suffix.lower()}"
            shutil.copy2(p.resolve(), dst)
            shutil.copyfile(label_path(p), out / "generated_labels" / f"{stem}.txt")
            im = cv2.imread(str(dst))
            h, w = im.shape[:2]
            for k, pts, _ in read_label(label_path(p)):
                poly = np.round(pts * np.array([w, h])).astype(np.int32)
                cv2.polylines(im, [poly], True, PALETTE[k % len(PALETTE)], max(2, w // 300))
            cv2.putText(im, names[c], (10, max(30, h // 20)), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w / 800),
                        (255, 255, 255), max(2, w // 400))
            cv2.imwrite(str(out / "overlays" / f"{stem}.jpg"), im)
            rows.append({"image": dst.name, "stem": stem, "generated_class": names[c], "source": str(p),
                         "label_correct": "", "notes": ""})
    write_csv(out / "manifest.csv", rows, ["image", "stem", "generated_class", "source", "label_correct", "notes"])
    print(f"{len(rows)} images sampled into {out} ({ {names[c]: min(a.per_class, len(v)) for c, v in sorted(by_cls.items())} })")


if __name__ == "__main__":
    main()
