#!/usr/bin/env python3
"""Score the generated annotations against the manual audit (see audit_sample.py).

For every audited image that has a manual label file:
  * foreground mask IoU and boundary IoU (Cheng et al., CVPR 2021, dilation 2% of
    the image diagonal) between the union of generated and the union of manual masks;
  * instance matching (greedy, mask IoU >= 0.5): matched mean IoU, instance precision
    and recall of the generated polygons, and class agreement of matched pairs;
  * label correctness from the manifest (label_correct = 1/0), if filled in.
Masks are rasterised at the canonical scale (long side 640 px).

Example
-------
python audit_score.py --audit /scratch/audit --names /scratch/protocol/v3.yaml --out /scratch/results/audit
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import canonical_size, class_names, image_size, load_yaml, read_csv, read_label, write_csv, write_json


def rasterise(instances, w, h):
    masks, cls = [], []
    for c, pts, _ in instances:
        m = np.zeros((h, w), np.uint8)
        poly = np.round(pts * np.array([w, h])).astype(np.int32)
        if len(poly) >= 3:
            cv2.fillPoly(m, [poly], 1)
        if m.any():
            masks.append(m)
            cls.append(c)
    return masks, cls


def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    d = max(1, int(round(dilation_ratio * np.sqrt(h ** 2 + w ** 2))))
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=d)[1:h + 1, 1:w + 1]
    return mask - eroded


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def boundary_iou(a, b, dilation_ratio=0.02):
    ba, bb = mask_to_boundary(a, dilation_ratio), mask_to_boundary(b, dilation_ratio)
    u = ((ba + bb) > 0).sum()
    return float(((ba * bb) > 0).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = class_names(load_yaml(a.names))
    audit = Path(a.audit)
    rows = []
    for r in read_csv(audit / "manifest.csv"):
        man_f = audit / "manual_labels" / f"{r['stem']}.txt"
        gen_f = audit / "generated_labels" / f"{r['stem']}.txt"
        row = {"image": r["image"], "generated_class": r["generated_class"],
               "label_correct": r.get("label_correct", "").strip()}
        if man_f.exists():
            w, h = canonical_size(*image_size(audit / "images" / r["image"]))
            gm, gc = rasterise(read_label(gen_f), w, h)
            mm, mc = rasterise(read_label(man_f), w, h)
            gu = np.clip(sum(gm), 0, 1).astype(np.uint8) if gm else np.zeros((h, w), np.uint8)
            mu = np.clip(sum(mm), 0, 1).astype(np.uint8) if mm else np.zeros((h, w), np.uint8)
            row.update({"mask_iou": iou(gu, mu), "boundary_iou": boundary_iou(gu, mu),
                        "n_generated": len(gm), "n_manual": len(mm)})
            pairs = sorted(((iou(g, m), i, j) for i, g in enumerate(gm) for j, m in enumerate(mm)), reverse=True)
            used_g, used_m, matched = set(), set(), []
            for v, i, j in pairs:
                if v < 0.5:
                    break
                if i in used_g or j in used_m:
                    continue
                used_g.add(i)
                used_m.add(j)
                matched.append((v, gc[i] == mc[j]))
            row.update({"n_matched": len(matched),
                        "matched_mean_iou": float(np.mean([v for v, _ in matched])) if matched else float("nan"),
                        "instance_precision": len(matched) / len(gm) if gm else float("nan"),
                        "instance_recall": len(matched) / len(mm) if mm else float("nan"),
                        "matched_class_agreement": float(np.mean([s for _, s in matched])) if matched else float("nan")})
        rows.append(row)
    out = Path(a.out)
    write_csv(out / "audit_per_image.csv", rows)

    def agg(rs):
        def m(k):
            v = [float(x[k]) for x in rs if k in x and x[k] == x[k]]
            return float(np.mean(v)) if v else float("nan")
        lab = [int(x["label_correct"]) for x in rs if x["label_correct"] in ("0", "1")]
        return {"n_images": len(rs), "n_with_manual": sum(1 for x in rs if "mask_iou" in x),
                "mask_iou": m("mask_iou"), "boundary_iou": m("boundary_iou"), "matched_mean_iou": m("matched_mean_iou"),
                "instance_precision": m("instance_precision"), "instance_recall": m("instance_recall"),
                "label_accuracy": float(np.mean(lab)) if lab else float("nan"), "n_label_checked": len(lab)}
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["generated_class"]].append(r)
    summary = [{"class": c, **agg(by_cls[c])} for c in names if c in by_cls] + [{"class": "all", **agg(rows)}]
    write_csv(out / "audit_summary.csv", summary)
    write_json(out / "audit_summary.json", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
