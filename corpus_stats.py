#!/usr/bin/env python3
"""Composition and geometry of each corpus (Tables III-V of the paper).

For every corpus and partition: images, instances, instances per image, and per
class instance counts; for the whole corpus: polygon vertices per instance,
share of multi-class images, class co-occurrences, and per class the mean
instance area and the small / medium / large shares under the COCO thresholds
(32^2 and 96^2 px) measured at the canonical scale (long side 640 px).

Example
-------
python corpus_stats.py --data v1=/scratch/protocol/v1.yaml v2=/scratch/protocol/v2.yaml \
    v3=/scratch/protocol/v3.yaml --out /scratch/results/corpus_stats.json
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from common import (MEDIUM_AREA, SMALL_AREA, canonical_size, class_names, image_size, label_path, load_yaml,
                    polygon_area, read_label, split_images, write_json)


def describe(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x))}


def _read_one(p):
    """Everything that touches the filesystem for one image, so it can be done in parallel."""
    inst = read_label(label_path(p))
    return (inst, canonical_size(*image_size(p)) if inst else None)


def corpus_stats(yaml_path, workers: int = 1):
    names = class_names(load_yaml(yaml_path))
    out = {"names": names, "splits": {}}
    verts, per_img_all = [], []
    cls_count = Counter()
    area_norm, area_px = defaultdict(list), defaultdict(list)
    multi, labelled, boxes, degenerate = 0, 0, 0, 0
    cooc = Counter()
    for split, imgs in split_images(yaml_path).items():
        n_inst, per_img, split_cls = 0, [], Counter()
        if workers > 1 and len(imgs) > 64:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                records = ex.map(_read_one, imgs, chunksize=64)
                records = list(records)
        else:
            records = [_read_one(p) for p in imgs]
        for inst, size in records:
            per_img.append(len(inst))
            n_inst += len(inst)
            if not inst:
                continue
            labelled += 1
            present = sorted({c for c, _, _ in inst})
            if len(present) > 1:
                multi += 1
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        cooc[(present[i], present[j])] += 1
            w, h = size
            for c, pts, is_box in inst:
                split_cls[c] += 1
                cls_count[c] += 1
                if is_box:
                    boxes += 1
                elif len(pts) < 3:
                    degenerate += 1
                else:
                    verts.append(len(pts))
                a = polygon_area(pts)
                area_norm[c].append(a)
                area_px[c].append(a * w * h)
        out["splits"][split] = {"images": len(imgs), "instances": n_inst,
                                "instances_per_image": describe(per_img),
                                "per_class": {names[c] if 0 <= c < len(names) else str(c): n for c, n in sorted(split_cls.items())}}
        per_img_all += per_img
    out["total"] = {"images": len(per_img_all), "instances": int(sum(per_img_all)),
                    "instances_per_image": describe(per_img_all),
                    "per_class": {names[c] if 0 <= c < len(names) else str(c): n for c, n in sorted(cls_count.items())},
                    "vertices": describe(verts), "box_lines": boxes, "polygons_under_3_points": degenerate,
                    "labelled_images": labelled, "multiclass_images": multi,
                    "multiclass_share": multi / labelled if labelled else 0.0,
                    "cooccurrence": {f"{names[a]}+{names[b]}": n for (a, b), n in cooc.most_common()}}
    per_class = {}
    for c in sorted(area_px):
        px = np.asarray(area_px[c])
        name = names[c] if 0 <= c < len(names) else str(c)
        per_class[name] = {"instances": int(px.size), "mean_area_norm": float(np.mean(area_norm[c])),
                           "median_area_px": float(np.median(px)),
                           "small": float(np.mean(px < SMALL_AREA)),
                           "medium": float(np.mean((px >= SMALL_AREA) & (px < MEDIUM_AREA))),
                           "large": float(np.mean(px >= MEDIUM_AREA))}
    out["per_class_geometry"] = per_class
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="name=path/to/data.yaml ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    res = {}
    for item in a.data:
        name, path = item.split("=", 1)
        print(f"{name}: {path}", flush=True)
        res[name] = corpus_stats(path, workers=a.workers)
        t = res[name]["total"]
        print(f"  {t['images']} images, {t['instances']} instances, vertices median {t['vertices']['median']}, "
              f"multi-class {100 * t['multiclass_share']:.2f}%")
    write_json(a.out, res)


if __name__ == "__main__":
    main()
