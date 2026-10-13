#!/usr/bin/env python3
"""Everything the fold-based evaluation needs, built once.

For every fold:
  * data YAMLs for the corpora as they were trained (v1_<fold>.yaml, v2_<fold>.yaml, v3_<fold>.yaml);
  * the two control corpora:
      v3_sizematched_<fold>  v3 training set subsampled to the size of the v1 training set,
                             written as an image list -- no files are copied or linked;
      v3_deranged_<fold>     v3 training set with the generated labels outside the majority
                             class replaced by wrong ones, class frequencies unchanged; only
                             the images whose label changes are materialised, the rest of the
                             training set is referenced where it already is;
  * the COCO ground truth of the common test set of the fold (from check_folds.py).
Then, once: a leakage check (common test images that share a source photograph with a training
image of the same fold) and the corpus statistics of v1, v2 and v3 (Tables III-V of the paper).

Every stage prints how long it took, and finished folds are skipped when the script is re-run,
so an interrupted preparation continues where it stopped.

Example
-------
python prepare_folds.py --folds-root /data/user/exp-to-clean/folds --report /scratch/folds_report \
    --out /scratch/protocol --results /scratch/results --workers 16
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cocoutils import build_gt, save_json_gz
from common import (class_names, label_path, load_yaml, read_csv, read_json, save_yaml, split_images,
                    split_sources, write_json)
from folds import common_test_images, folds_available
from prepare_protocol import (compute_dhashes, derange_labels, image_classes, place_image, rarest_class, rf_stem,
                              stratified_subsample, warmup_cache, UnionFind)


class Timer:
    """Prints how long each stage took, so a slow preparation says where the time goes."""

    def __init__(self):
        self.t0 = time.time()
        self.marks = {}

    def __call__(self, label):
        now = time.time()
        self.marks[label] = now - self.marks.pop("_last", self.t0)
        print(f"    [{label}: {self.marks[label]:.1f} s]", flush=True)
        self.marks["_last"] = now

    def start(self):
        self.marks["_last"] = time.time()


def write_list(path: Path, images) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{Path(p).resolve()}\n" for p in images))
    return path


def fold_yaml_dict(yaml_path: Path, names) -> dict:
    """A YAML pointing at the splits of an existing fold dataset, with absolute paths."""
    d = {}
    for s, srcs in split_sources(yaml_path).items():
        d[s] = [str(x) for x in srcs] if len(srcs) > 1 else str(srcs[0])
    d["nc"] = len(names)
    d["names"] = list(names)
    return d


def classes_of_images(images, workers: int, cache: Path | None = None) -> dict:
    """{image path: [class ids]} for every image, read in parallel and cached on disk.

    Reading one small label file is latency-bound, so this is the difference between a
    few minutes and the best part of an hour on a shared filesystem.
    """
    known = {}
    if cache is not None and cache.exists():
        known = read_json(cache)
    todo = [str(p) for p in images if str(p) not in known]
    if todo:
        print(f"  reading the labels of {len(todo)} images with {workers} workers ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, cls in zip(todo, ex.map(lambda q: image_classes(Path(q)), todo, chunksize=64)):
                known[p] = cls
        if cache is not None:
            write_json(cache, known)
    return known


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-root", required=True, help="folder with <corpus>/<fold>/data.yaml")
    ap.add_argument("--report", required=True, help="output folder of check_folds.py")
    ap.add_argument("--out", required=True, help="protocol folder (YAMLs and control corpora)")
    ap.add_argument("--results", required=True, help="results folder (ground truth of the common test sets)")
    ap.add_argument("--combined", default="v3")
    ap.add_argument("--sources", default="v1,v2")
    ap.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--derange-exclude", default=None, help="class(es) left unchanged; default: the majority class of v2")
    ap.add_argument("--folds", default=None, help="only these folds, e.g. 'fold0 fold1'")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-leakage", action="store_true", help="skip the duplicate-source check (it hashes every image)")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--warmup", action="store_true",
                    help="pre-build the Ultralytics label caches (slow: it verifies every image of every corpus; "
                         "training builds them anyway on first use)")
    ap.add_argument("--force", action="store_true", help="rebuild folds that are already prepared")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    a = ap.parse_args()

    froot, out, res = Path(a.folds_root).resolve(), Path(a.out).resolve(), Path(a.results).resolve()
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    sources = a.sources.split(",")
    folds = folds_available(a.report)
    if a.folds:
        want = set(a.folds.split())
        folds = [f for f in folds if f in want]
    if not folds:
        raise SystemExit(f"no common_test_*.txt in {a.report} (run check_folds.py first)")
    names = class_names(load_yaml(froot / a.combined / folds[0] / "data.yaml"))
    origin = {r["image"]: r["origin"] for r in read_csv(Path(a.report) / "origin.csv")}
    rng = np.random.default_rng(a.seed)
    state_path = out / "protocol_summary.json"
    summary = read_json(state_path) if state_path.exists() else {"folds": {}, "names": names, "seed": a.seed}
    summary.setdefault("folds", {})
    T = Timer()

    todo = [f for f in folds if a.force or f not in summary["folds"]]
    if not todo:
        print(f"{out} already prepared for {', '.join(folds)}; nothing to do (--force rebuilds).")
    else:
        # one pass over the labels of every training image of the combined corpus, reused by every fold
        train_imgs = sorted({str(p) for f in todo for p in split_images(froot / a.combined / f / "data.yaml")["train"]})
        T.start()
        classes = {} if a.no_controls else classes_of_images(train_imgs, a.workers, out / "train_classes.json")
        if not a.no_controls:
            T("labels of the combined training sets")

    for fold in todo:
        print(f"\n=== {fold}", flush=True)
        T.start()
        info = {}
        yamls = {}
        for c in [a.combined] + sources:
            y = froot / c / fold / "data.yaml"
            yamls[c] = split_images(y)
            save_yaml(fold_yaml_dict(y, names), out / f"{c}_{fold}.yaml")
            info[c] = {s: len(v) for s, v in yamls[c].items()}
        v3 = yamls[a.combined]
        src = split_sources(froot / a.combined / fold / "data.yaml")
        v3_val_test = {k: ([str(x) for x in src[k]] if len(src[k]) > 1 else str(src[k][0]))
                       for k in ("val", "test") if k in src}
        T("data YAMLs")

        # ground truth of the common test set
        imgs = common_test_images(a.report, fold)
        gt, gstats = build_gt(imgs, names, workers=a.workers)
        save_json_gz(gt, res / f"gt_common_{fold}.json.gz")
        info["common_test"] = {"images": len(imgs), **gstats}
        print(f"  common test: {len(imgs)} images, {gstats['annotations_kept']} instances -> gt_common_{fold}.json.gz")
        T("ground truth of the common test set")

        if not a.no_controls:
            tr = [str(p) for p in v3["train"]]
            classes_of = {p: classes.get(p, []) for p in tr}
            freq = Counter(c for v in classes_of.values() for c in v)

            # --- size-matched: an image list, so nothing is copied or linked
            keys = [(origin.get(Path(p).name, "?"), rarest_class(classes_of[p], freq)) for p in tr]
            n_target = len(yamls[sources[0]]["train"])
            sub = stratified_subsample(tr, keys, n_target, rng)
            lst = write_list(out / "lists" / f"v3_sizematched_{fold}_train.txt", sub)
            save_yaml({"path": str(out), "train": str(lst), **v3_val_test, "nc": len(names), "names": list(names)},
                      out / f"v3_sizematched_{fold}.yaml")
            info["sizematched"] = {"images": len(sub), "target": n_target,
                                   "v1_origin": sum(1 for p in sub if origin.get(Path(p).name) == sources[0])}
            print(f"  size-matched: {len(sub)} of {len(tr)} training images (v1 training set has {n_target})")

            # --- deranged labels: only the images whose class changes are materialised
            img_class = {}
            for p in tr:
                if origin.get(Path(p).name) == sources[1] and classes_of[p]:
                    cnt = Counter(classes_of[p])
                    img_class[p] = min(cnt, key=lambda c: (-cnt[c], c))
            maj = Counter(img_class.values())
            excl = [names.index(x.strip()) for x in a.derange_exclude.split(",")] if a.derange_exclude else \
                   ([maj.most_common(1)[0][0]] if maj else [])
            eligible = {p: c for p, c in img_class.items() if c not in excl}
            relabel = derange_labels(eligible, exclude=None, rng=rng)
            root = out / "corpora" / f"v3_deranged_{fold}" / "train"
            n_links = materialise_relabelled(root, relabel, a.link)
            keep = write_list(out / "lists" / f"v3_deranged_{fold}_keep.txt", [p for p in tr if p not in relabel])
            save_yaml({"path": str(out), "train": [str(keep), str(root / "images")], **v3_val_test,
                       "nc": len(names), "names": list(names)}, out / f"v3_deranged_{fold}.yaml")
            info["deranged"] = {"excluded": [names[c] for c in excl], "generated_train_images": len(img_class),
                                "relabelled_images": len(relabel), "materialised": n_links,
                                "relabelled_instances": sum(len(classes_of[p]) for p in relabel),
                                "all_changed": all(relabel[p] != eligible[p] for p in relabel)}
            print(f"  deranged labels: {len(relabel)} of {len(img_class)} generated training images relabelled "
                  f"(all changed: {info['deranged']['all_changed']}, excluded: {info['deranged']['excluded']}); "
                  f"the other {len(tr) - len(relabel)} are referenced where they are")
            T("control corpora")
        summary["folds"][fold] = info
        write_json(state_path, summary)   # after every fold, so a rerun continues here

    # ---------------------------------------------------------------- leakage inside each fold
    if not a.no_leakage and (todo or "common_test_sharing_source_with_training" not in
                             summary["folds"].get(folds[0], {})):
        print("\n=== leakage inside the folds (source photograph shared between training and common test)")
        T.start()
        allimgs = sorted({p for fold in folds for v in split_images(out / f"{a.combined}_{fold}.yaml").values()
                          for p in v}, key=str)
        print(f"  difference hashes of {len(allimgs)} images with {a.workers} workers ...", flush=True)
        hashes = dict(zip(allimgs, compute_dhashes(allimgs, a.workers)))
        idx = {p: i for i, p in enumerate(allimgs)}
        T("difference hashes")
        for fold in folds:
            sp = split_images(out / f"{a.combined}_{fold}.yaml")
            seen = set(sp["train"]) | set(sp["val"])
            test = set(common_test_images(a.report, fold))
            uf = UnionFind(len(allimgs))
            first = {}
            for p in seen | test:
                for k in (rf_stem(p.name), hashes.get(p)):
                    if k is None:
                        continue
                    if k in first:
                        uf.union(first[k], idx[p])
                    else:
                        first[k] = idx[p]
            groups = defaultdict(list)
            for p in seen | test:
                groups[uf.find(idx[p])].append(p)
            n = sum(1 for p in test if any(q in seen for q in groups[uf.find(idx[p])]))
            summary["folds"].setdefault(fold, {})["common_test_sharing_source_with_training"] = n
            print(f"  {fold}: {n} of {len(test)} common test images share a source with a training image")
        write_json(state_path, summary)
        T("grouping")

    # ---------------------------------------------------------------- corpus statistics
    if not a.no_stats and not (res / "corpus_stats.json").exists():
        print("\n=== corpus statistics", flush=True)
        T.start()
        import corpus_stats
        stats = {}
        for c in [*sources, a.combined]:
            stats[c] = corpus_stats.corpus_stats(out / f"{c}_{folds[0]}.yaml", workers=a.workers)
            t = stats[c]["total"]
            print(f"  {c}: {t['images']} images, {t['instances']} instances, "
                  f"vertices median {t['vertices']['median']}, multi-class {100 * t['multiclass_share']:.2f}%")
            T(f"statistics of {c}")
        write_json(res / "corpus_stats.json", stats)

    if a.warmup:
        print("\n=== pre-building the Ultralytics label caches", flush=True)
        T.start()
        for y in sorted(out.glob("v3_*_fold*.yaml")):
            warmup_cache(y)
        T("cache warm-up")
    print(f"\nDone in {time.time() - T.t0:.0f} s. YAMLs in {out}, ground truth in {res}")


def materialise_relabelled(root: Path, relabel: dict, mode: str) -> int:
    """Link the images whose class changes into <root>/images and write their new labels.

    Only these images need a path of their own; every other training image keeps the one it
    already has, which is what makes this cheap.
    """
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    used, n = set(), 0
    for p, new_c in sorted(relabel.items()):
        p = Path(p)
        name, k = p.name, 1
        while name in used:                         # two sources can hold the same basename
            name = f"{p.stem}__{k}{p.suffix}"
            k += 1
        used.add(name)
        place_image(p, root / "images" / name, mode)
        lines = []
        for line in label_path(p).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                parts[0] = str(new_c)
                lines.append(" ".join(parts))
        (root / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
        n += 1
    return n


if __name__ == "__main__":
    main()
