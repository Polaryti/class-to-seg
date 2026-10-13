#!/usr/bin/env python3
"""Accuracy of the labelling classifier on a held-out split of its own corpus.

The held-out images must be laid out as <root>/<class>/.../*.jpg and must not
have been used to train the classifier. Reports top-1 accuracy, per-class
precision / recall / F1, the confusion matrix, and -- because the pipeline only
keeps confident predictions -- accuracy and coverage as a function of the
confidence threshold.

Example
-------
python eval_classifier.py --data /data/cls_corpus/test --classifier /models/classifier.pt \
    --names /scratch/protocol/v3.yaml --out /scratch/results/classifier
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from autoannotate import Classifier, human_label, list_source_images
from common import class_names, load_yaml, write_csv, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--cls-imgsz", type=int, default=None)
    ap.add_argument("--taus", default="0.0,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = class_names(load_yaml(a.names))
    root = Path(a.data).resolve()
    clf = Classifier(a.classifier, names, a.device, a.cls_imgsz)
    y, yhat, conf = [], [], []
    for p in list_source_images(root):
        h = human_label(p, root, names)
        if h == "unknown":
            continue
        with Image.open(p) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")
        k, c, _, _ = clf(im)
        y.append(names.index(h))
        yhat.append(k)
        conf.append(c)
    y, yhat, conf = np.array(y), np.array(yhat), np.array(conf)
    if y.size == 0:
        raise SystemExit(f"No labelled images found under {root}")
    K = len(names)
    cm = np.zeros((K, K), int)
    for t, p in zip(y, yhat):
        cm[t, p] += 1
    per = []
    for k, n in enumerate(names):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else float("nan")
        rec = tp / cm[k].sum() if cm[k].sum() else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else float("nan")
        per.append({"class": n, "n": int(cm[k].sum()), "precision": prec, "recall": rec, "f1": f1})
    taus = [float(x) for x in a.taus.split(",")]
    cov = []
    for t in taus:
        m = conf >= t
        cov.append({"tau": t, "coverage": float(m.mean()),
                    "accuracy": float((y[m] == yhat[m]).mean()) if m.any() else float("nan"), "n": int(m.sum())})
    out = Path(a.out)
    write_csv(out / "classifier_per_class.csv", per)
    write_csv(out / "classifier_coverage.csv", cov)
    f1s = [r["f1"] for r in per if not np.isnan(r["f1"])]
    summary = {"n": int(y.size), "top1": float((y == yhat).mean()), "macro_f1": float(np.mean(f1s)) if f1s else None,
               "confusion_matrix": cm.tolist(), "names": names, "classifier": clf.info}
    print(f"classifier: {clf.info}")
    write_json(out / "classifier_summary.json", summary)
    print(f"top-1 {summary['top1']:.4f}, macro-F1 {summary['macro_f1']:.4f} on {y.size} images")


if __name__ == "__main__":
    main()
