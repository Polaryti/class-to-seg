#!/usr/bin/env python3
"""Sensitivity of the pipeline to its three parameters, from the logs written by autoannotate.py.

For each confidence threshold tau in --taus (at the chosen epsilon and minimum
area): retained images, retention rate, generated instances (total and per
class) and the agreement between the classifier label and the human image-level
label of the source corpus among the retained images. Also: vertices per
instance for each epsilon, and instances as a function of the minimum area.

Example
-------
python pipeline_sweep.py --stats /scratch/pipeline/stats --conf 0.8 --eps 0.002 --min-area 1000 \
    --out /scratch/results/pipeline
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import write_csv, write_json


def load(stats_dir):
    recs = []
    for f in sorted(Path(stats_dir).glob("shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    meta = json.loads((Path(stats_dir) / "meta.json").read_text())
    return recs, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True, help="<out>/stats directory of autoannotate.py")
    ap.add_argument("--conf", type=float, required=True, help="operating confidence threshold (paper value)")
    ap.add_argument("--eps", type=float, required=True, help="operating epsilon (must be in the logged eps grid)")
    ap.add_argument("--min-area", type=float, required=True, help="operating minimum contour area (px)")
    ap.add_argument("--taus", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99")
    ap.add_argument("--min-areas", default="0,100,500,1000,5000,10000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs, meta = load(a.stats)
    names, grid = meta["names"], meta["eps_grid"]
    if not recs:
        raise SystemExit("no records found")
    ei = min(range(len(grid)), key=lambda i: abs(grid[i] - a.eps))
    if abs(grid[ei] - a.eps) > 1e-12:
        print(f"WARNING: eps {a.eps} not in the logged grid {grid}; using {grid[ei]}")
    taus = sorted(set([float(x) for x in a.taus.split(",")] + [a.conf]))
    n_all = len(recs)
    human_known = [r for r in recs if r["human"] != "unknown"]
    src_counts = Counter(r["human"] for r in recs)

    def inst_of(r, min_area):
        return [c for c in r["contours"] if c[0] >= min_area and c[1 + ei] >= 3]

    rows = []
    for t in taus:
        kept = [r for r in recs if r["conf"] >= t]
        kept_with_inst = [r for r in kept if inst_of(r, a.min_area)]
        per_cls = Counter()
        for r in kept_with_inst:
            per_cls[r["pred"]] += len(inst_of(r, a.min_area))
        agree = [r["pred"] == r["human"] for r in kept_with_inst if r["human"] != "unknown"]
        row = {"tau": t, "operating": t == a.conf, "images_retained": len(kept_with_inst),
               "retention": len(kept_with_inst) / n_all, "instances": sum(per_cls.values()),
               "label_agreement": float(np.mean(agree)) if agree else float("nan")}
        for n in names:
            row[f"inst_{n}"] = per_cls.get(n, 0)
        rows.append(row)
    out = Path(a.out)
    write_csv(out / "pipeline_tau_sweep.csv", rows)

    # operating point: per class breakdown
    kept = [r for r in recs if r["conf"] >= a.conf and inst_of(r, a.min_area)]
    per_class = []
    for n in names:
        src_n = src_counts.get(n, 0)
        kept_h = [r for r in kept if r["human"] == n]
        kept_p = [r for r in kept if r["pred"] == n]
        inst_p = sum(len(inst_of(r, a.min_area)) for r in kept_p)
        per_class.append({"class": n, "source_images": src_n, "retained_images_by_human_label": len(kept_h),
                          "retention": len(kept_h) / src_n if src_n else float("nan"),
                          "generated_images": len(kept_p), "generated_instances": inst_p,
                          "delta_instances_vs_source_images": (inst_p - src_n) / src_n if src_n else float("nan"),
                          "precision_vs_human": float(np.mean([r["human"] == n for r in kept_p if r["human"] != "unknown"])) if kept_p else float("nan"),
                          "recall_vs_human": float(np.mean([r["pred"] == n for r in kept_h])) if kept_h else float("nan")})
    write_csv(out / "pipeline_per_class.csv", per_class)

    # epsilon and minimum area
    eps_rows = []
    for j, e in enumerate(grid):
        v = [c[1 + j] for r in kept for c in inst_of(r, a.min_area)]
        eps_rows.append({"eps": e, "vertices_mean": float(np.mean(v)) if v else float("nan"),
                         "vertices_median": float(np.median(v)) if v else float("nan"),
                         "operating": j == ei})
    write_csv(out / "pipeline_eps.csv", eps_rows)
    area_rows = []
    kept_conf = [r for r in recs if r["conf"] >= a.conf]
    for ma in sorted(set([float(x) for x in a.min_areas.split(",")] + [a.min_area])):
        inst = [len(inst_of(r, ma)) for r in kept_conf]
        area_rows.append({"min_area": ma, "images_with_instances": sum(1 for x in inst if x),
                          "instances": int(sum(inst)), "operating": ma == a.min_area})
    write_csv(out / "pipeline_min_area.csv", area_rows)

    summary = {"source_images": n_all, "source_images_with_human_label": len(human_known),
               "classifier_agreement_all_images": float(np.mean([r["pred"] == r["human"] for r in human_known])) if human_known else None,
               "operating": {"conf": a.conf, "eps": grid[ei], "min_area": a.min_area, "mask_thr": meta["mask_thr"],
                             "segmenter": meta["segmenter"], "birefnet": meta["birefnet"], "birefnet_size": meta["birefnet_size"]},
               "images_retained": len(kept), "instances": int(sum(len(inst_of(r, a.min_area)) for r in kept)),
               "images_without_contour": sum(1 for r in recs if not r["contours"]),
               "median_foreground_fraction": float(np.median([r["fg"] for r in recs]))}
    write_json(out / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"tau {r['tau']:.2f}: retained {r['images_retained']} ({100 * r['retention']:.1f}%), "
              f"instances {r['instances']}, agreement {100 * r['label_agreement']:.1f}%")


if __name__ == "__main__":
    main()
