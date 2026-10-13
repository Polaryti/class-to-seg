#!/usr/bin/env python3
"""Why common test images share a source photograph with training images, broken down.

prepare_folds.py reports one number per fold: how many images of the common test set belong
to the same source photograph as an image the fold trained on. Two things can put two images
in the same group, and they mean very different things:

  rf-stem  both file names carry the same Roboflow source identifier, so they are augmented
           copies of one photograph -- real leakage;
  dhash    their 64-bit difference hashes are equal. For photographs that is near-duplicate
           content, but for studio images on a flat background it can also be two different
           objects whose downscaled 8x8 greyscale happens to match -- a false positive.

This script separates the two, reports them per fold and per origin, and lists the largest
hash groups so that a degenerate hash (hundreds of images in one group) is visible at once.

Example
-------
python leakage_report.py --folds-root /data/user/exp-to-clean/folds --report /scratch/folds_report \
    --out /scratch/results/leakage --workers 16
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from common import read_csv, read_json, write_csv, write_json
from folds import common_test_images, folds_available
from prepare_protocol import compute_dhashes, rf_stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-root", required=True)
    ap.add_argument("--report", required=True, help="output folder of check_folds.py")
    ap.add_argument("--protocol", default=None, help="protocol folder (default: alongside --report)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combined", default="v3")
    ap.add_argument("--hash-cache", default=None, help="JSON of {image: dhash}, written if absent")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    a = ap.parse_args()

    from common import split_images
    folds = folds_available(a.report)
    proto = Path(a.protocol) if a.protocol else Path(a.report).parent / "protocol"
    origin = {r["image"]: r["origin"] for r in read_csv(Path(a.report) / "origin.csv")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def fold_yaml(fold):
        p = proto / f"{a.combined}_{fold}.yaml"
        return p if p.exists() else Path(a.folds_root) / a.combined / fold / "data.yaml"

    allimgs = sorted({p for f in folds for v in split_images(fold_yaml(f)).values() for p in v}, key=str)
    cache = Path(a.hash_cache) if a.hash_cache else out / "dhashes.json"
    if cache.exists():
        hashes = {Path(k): v for k, v in read_json(cache).items()}
        print(f"{len(hashes)} hashes read from {cache}")
    else:
        print(f"hashing {len(allimgs)} images with {a.workers} workers ...", flush=True)
        hashes = dict(zip(allimgs, compute_dhashes(allimgs, a.workers)))
        write_json(cache, {str(k): v for k, v in hashes.items()})

    hgroups = Counter(h for h in hashes.values() if h)
    big = hgroups.most_common(15)
    print(f"\n{len(hgroups)} distinct hashes for {sum(hgroups.values())} images; "
          f"the largest group holds {big[0][1] if big else 0} images")
    write_csv(out / "largest_hash_groups.csv",
              [{"dhash": h, "images": n,
                "example": next((str(p) for p, v in hashes.items() if v == h), "")} for h, n in big])

    rows, summary = [], {}
    for fold in folds:
        sp = split_images(fold_yaml(fold))
        seen = set(sp["train"]) | set(sp["val"])
        test = [p for p in common_test_images(a.report, fold)]
        stem_of_seen = {rf_stem(p.name) for p in seen}
        hash_of_seen = {hashes.get(p) for p in seen} - {None}
        c = Counter()
        for p in test:
            o = origin.get(p.name, "unknown")
            by_stem = rf_stem(p.name) in stem_of_seen
            by_hash = hashes.get(p) is not None and hashes.get(p) in hash_of_seen
            c[("any", o)] += int(by_stem or by_hash)
            c[("stem", o)] += int(by_stem)
            c[("hash", o)] += int(by_hash)
            c[("hash_only", o)] += int(by_hash and not by_stem)
            c[("total", o)] += 1
        for o in ("v1", "v2", "unknown"):
            if not c[("total", o)]:
                continue
            rows.append({"fold": fold, "origin": o, "test_images": c[("total", o)],
                         "shares_source": c[("any", o)], "by_roboflow_stem": c[("stem", o)],
                         "by_difference_hash": c[("hash", o)], "by_hash_only": c[("hash_only", o)],
                         "share_pct": round(100 * c[("any", o)] / c[("total", o)], 1)})
        tot = sum(c[("total", o)] for o in ("v1", "v2", "unknown"))
        any_ = sum(c[("any", o)] for o in ("v1", "v2", "unknown"))
        stem = sum(c[("stem", o)] for o in ("v1", "v2", "unknown"))
        honly = sum(c[("hash_only", o)] for o in ("v1", "v2", "unknown"))
        summary[fold] = {"test_images": tot, "shares_source": any_, "by_roboflow_stem": stem,
                         "by_hash_only": honly}
        print(f"{fold}: {any_} of {tot} common test images share a source "
              f"({stem} by Roboflow identifier, {honly} only by difference hash)")
        for o in ("v1", "v2"):
            if c[("total", o)]:
                print(f"    {o}-origin: {c[('any', o)]} of {c[('total', o)]} "
                      f"({100 * c[('any', o)] / c[('total', o)]:.1f}%)")
    write_csv(out / "leakage_by_fold.csv", rows)
    write_json(out / "leakage_summary.json", {"folds": summary,
                                              "largest_hash_groups": [{"dhash": h, "images": n} for h, n in big]})
    print(f"\nWrote {out}/leakage_by_fold.csv, largest_hash_groups.csv, leakage_summary.json")


if __name__ == "__main__":
    main()
