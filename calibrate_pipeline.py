#!/usr/bin/env python3
"""Recover the three pipeline parameters (tau, epsilon, minimum contour area) from the generated corpus v2.

Needed when the values used to generate v2 were not recorded. Inputs:
  * the logs of autoannotate.py over the whole source classification corpus: for every image,
    the classifier confidence and, for every contour, its area and its simplified vertex count
    for each epsilon of the grid;
  * the generated corpus v2 as released (its data YAML).
Procedure:
  1. for each candidate minimum area A, tau(A) is the confidence at which the number of retained
     images (confidence >= tau and at least one contour of area >= A) equals the number of
     annotated images in v2; the chosen A is the one whose retained instances best match the
     number of instances in v2;
  2. epsilon is the grid value whose median vertex count per instance is closest to that of v2;
  3. per-class image and instance counts are compared with v2 -- this is the independent check,
     since only the totals were used to calibrate.
Values given explicitly (numbers instead of 'auto') are used as they are and only checked.

Writes calibration.json, calibration_area_curve.csv, calibration_eps.csv, calibration_per_class.csv
and, with --write-env, a shell file exporting PIPE_CONF, PIPE_EPS and PIPE_MIN_AREA.

Example
-------
python calibrate_pipeline.py --stats /scratch/pipeline/stats --v2 /data/v2/data.yaml \
    --conf auto --eps auto --min-area auto --out /scratch/results/pipeline --write-env /scratch/results/pipeline/params.sh
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from common import class_names, label_path, load_yaml, read_label, split_images, write_csv, write_json
from pipeline_sweep import load

# 0 and ten values per decade from 10 to 100,000 px (2 significant digits)
AREA_GRID = ",".join(["0"] + [f"{float(f'{v:.2g}'):g}" for v in np.logspace(1, 5, 41)])


def v2_targets(yaml_path):
    names = class_names(load_yaml(yaml_path))
    n_img, n_inst, verts = 0, 0, []
    img_cls, inst_cls = Counter(), Counter()
    for imgs in split_images(yaml_path).values():
        for p in imgs:
            inst = read_label(label_path(p))
            if not inst:
                continue
            n_img += 1
            n_inst += len(inst)
            cnt = Counter(c for c, _, _ in inst)
            img_cls[names[min(cnt, key=lambda c: (-cnt[c], c))]] += 1
            for c, pts, is_box in inst:
                inst_cls[names[c]] += 1
                if not is_box:
                    verts.append(len(pts))
    return {"images": n_img, "instances": n_inst, "vertices_median": float(np.median(verts)) if verts else None,
            "images_per_class": dict(img_cls), "instances_per_class": dict(inst_cls)}


def parse(v):
    return None if str(v).strip().lower() in ("", "auto", "none") else float(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True, help="<out>/stats directory of autoannotate.py")
    ap.add_argument("--v2", required=True, help="data YAML of the generated corpus v2 as released")
    ap.add_argument("--conf", default="auto")
    ap.add_argument("--eps", default="auto")
    ap.add_argument("--min-area", default="auto")
    ap.add_argument("--area-grid", default=AREA_GRID)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-env", default=None, help="shell file with the resolved PIPE_* values")
    a = ap.parse_args()

    recs, meta = load(a.stats)
    grid = meta["eps_grid"]
    tgt = v2_targets(a.v2)
    if not recs or not tgt["images"]:
        raise SystemExit("nothing to calibrate (no logs or empty v2)")
    if tgt["images"] > len(recs):
        raise SystemExit(
            f"--v2 points at a corpus of {tgt['images']} images, more than the {len(recs)} source images the "
            f"pipeline logged. The pipeline cannot have produced it, so this is not the generated corpus: "
            f"--v2 must be the v2 data YAML (V2_YAML in config.sh), not the combined corpus.\n"
            f"  --v2 = {a.v2}")
    conf = np.array([r["conf"] for r in recs])
    user = {"conf": parse(a.conf), "eps": parse(a.eps), "min_area": parse(a.min_area)}

    def n_inst(A, ei):
        return np.array([sum(1 for c in r["contours"] if c[0] >= A and c[1 + ei] >= 3) for r in recs])

    def tau_for(counts, n_target):
        """Largest threshold (floored to 3 decimals) that retains at least n_target images."""
        cand = np.sort(conf[counts > 0])[::-1]
        if len(cand) == 0:
            return 0.0
        t = cand[min(n_target, len(cand)) - 1]
        return math.floor(t * 1000) / 1000

    ei0 = len(grid) // 2 if user["eps"] is None else min(range(len(grid)), key=lambda i: abs(grid[i] - user["eps"]))

    # 1. minimum area (and tau)
    curve = []
    areas = [user["min_area"]] if user["min_area"] is not None else [float(x) for x in a.area_grid.split(",")]
    for A in areas:
        counts = n_inst(A, ei0)
        t = user["conf"] if user["conf"] is not None else tau_for(counts, tgt["images"])
        keep = (conf >= t) & (counts > 0)
        inst = int(counts[keep].sum())
        curve.append({"min_area": A, "tau": t, "images": int(keep.sum()), "instances": inst,
                      "instances_rel_error": (inst - tgt["instances"]) / tgt["instances"]})
    # every area in the tie set yields the same corpus: A_min is identified only up to that range
    err_min = min(abs(r["instances_rel_error"]) for r in curve)
    ties = [r for r in curve if abs(abs(r["instances_rel_error"]) - err_min) < 1e-12]
    pos = [r["min_area"] for r in ties if r["min_area"] > 0]
    if not pos:
        best = ties[0]
    else:
        mid = math.sqrt(min(pos) * max(pos))
        best = min((r for r in ties if r["min_area"] > 0), key=lambda r: abs(math.log(r["min_area"] / mid)))
    area_range = [min(r["min_area"] for r in ties), max(r["min_area"] for r in ties)]
    A, t = best["min_area"], best["tau"]

    # 2. epsilon
    eps_rows = []
    keep = conf >= t
    for j, e in enumerate(grid):
        v = [c[1 + j] for r, k in zip(recs, keep) if k for c in r["contours"] if c[0] >= A and c[1 + j] >= 3]
        med = float(np.median(v)) if v else float("nan")
        eps_rows.append({"eps": e, "vertices_median": med,
                         "log_ratio_to_v2": abs(math.log(med / tgt["vertices_median"])) if v and tgt["vertices_median"] else float("inf")})
    if user["eps"] is not None:
        ej = ei0
    else:
        ej = min(range(len(grid)), key=lambda j: eps_rows[j]["log_ratio_to_v2"])
    eps = grid[ej]

    # 3. reproduction at the resolved values, per class
    counts = n_inst(A, ej)
    keep = (conf >= t) & (counts > 0)
    img_p, inst_p = Counter(), Counter()
    for r, k, n in zip(recs, keep, counts):
        if k:
            img_p[r["pred"]] += 1
            inst_p[r["pred"]] += int(n)
    per_class, img_err, inst_err = [], [], []
    for cls in meta["names"]:
        ti, tn = tgt["images_per_class"].get(cls, 0), tgt["instances_per_class"].get(cls, 0)
        row = {"class": cls, "v2_images": ti, "reproduced_images": img_p.get(cls, 0),
               "v2_instances": tn, "reproduced_instances": inst_p.get(cls, 0)}
        if ti:
            row["images_rel_error"] = (img_p.get(cls, 0) - ti) / ti
            img_err.append(abs(row["images_rel_error"]))
        if tn:
            row["instances_rel_error"] = (inst_p.get(cls, 0) - tn) / tn
            inst_err.append(abs(row["instances_rel_error"]))
        per_class.append(row)

    source = {k: ("given" if user[k] is not None else "calibrated") for k in user}
    summary = {"conf": t, "eps": eps, "min_area": A, "min_area_range": area_range, "source": source, "v2": tgt,
               "reproduced": {"images": int(keep.sum()), "instances": int(counts[keep].sum())},
               "max_class_images_rel_error": max(img_err) if img_err else None,
               "max_class_instances_rel_error": max(inst_err) if inst_err else None,
               "total_instances_rel_error": (int(counts[keep].sum()) - tgt["instances"]) / tgt["instances"],
               "source_images_logged": len(recs)}
    out = Path(a.out)
    write_csv(out / "calibration_area_curve.csv", curve)
    write_csv(out / "calibration_eps.csv", eps_rows)
    write_csv(out / "calibration_per_class.csv", per_class)
    write_json(out / "calibration.json", summary)
    if a.write_env:
        Path(a.write_env).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write_env).write_text(f'export PIPE_CONF="{t}"\nexport PIPE_EPS="{eps}"\nexport PIPE_MIN_AREA="{A:g}"\n')
    print(f"tau = {t} ({source['conf']}), eps = {eps} ({source['eps']}), min area = {A:g} px ({source['min_area']}; "
          f"any value in [{area_range[0]:g}, {area_range[1]:g}] gives the same corpus)")
    print(f"v2: {tgt['images']} images, {tgt['instances']} instances, median {tgt['vertices_median']} vertices; "
          f"reproduced: {summary['reproduced']['images']} images, {summary['reproduced']['instances']} instances")
    if img_err:
        print(f"largest per-class error: images {100 * max(img_err):.1f}%, instances {100 * max(inst_err):.1f}%")
        if max(img_err) > 0.15:
            print("\nThe per-class counts differ from v2 by more than 15%, so the parameters below reproduce the"
                  "\ntotals of v2 but not its composition. The usual causes, in order of likelihood:"
                  "\n  1. --v2 is not the generated corpus (check V2_YAML: it must cover v2, not v1 or the union);"
                  "\n  2. the classifier checkpoint in CLASSIFIER is not the one that generated v2;"
                  "\n  3. the self-segmentation checkpoint in BIREFNET is not the one that generated v2."
                  "\nThe manuscript reports these errors, so a large value here is a result, not a crash; but if it"
                  "\ncomes from 1 it is a configuration mistake and the numbers should not be used.")
    if not user["conf"] and t >= 0.999:
        print("\nNOTE: the calibrated threshold sits at the top of the confidence range, which means the logged"
              "\nimages barely reach the number of images of v2 even when none is discarded. Check that the stats"
              "\nin --stats cover the whole classification corpus and that --v2 is the generated corpus.")


if __name__ == "__main__":
    main()
