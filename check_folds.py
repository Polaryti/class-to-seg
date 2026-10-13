#!/usr/bin/env python3
"""Compare the per-fold partitions of the corpora, and build a common test set per fold.

Reads <root>/<corpus>/<fold>/data.yaml for every corpus and fold and, using image file
names, reports:
  * how the test partition of the combined corpus overlaps the partitions of the
    single-source corpora -- an image of v3's test set that a v1 model saw during its
    training cannot be used to compare the two corpora;
  * the images of each v3 test partition that NO model of that fold was trained or
    validated on: the common test set of the fold, written to common_test_<fold>.txt;
  * the origin (v1 or v2) of every image, written to origin.csv;
  * whether the test partitions of the folds are disjoint and cover the corpus.

Example
-------
python check_folds.py --root /data/user/exp-to-clean/folds --out /scratch/folds_report
"""
import argparse
import itertools
from pathlib import Path

from common import split_images, write_csv, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder containing <corpus>/<fold>/data.yaml")
    ap.add_argument("--combined", default="v3", help="name of the combined corpus (default v3)")
    ap.add_argument("--sources", default="v1,v2", help="names of the single-source corpora (default v1,v2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sources = a.sources.split(",")

    yamls = sorted(root.glob("*/*/data.yaml"))
    if not yamls:
        raise SystemExit(f"no <corpus>/<fold>/data.yaml under {root}")
    data, paths = {}, {}
    for y in yamls:
        corpus, fold = y.parent.parent.name, y.parent.name
        sp = split_images(y)
        data[(corpus, fold)] = {s: {p.name for p in v} for s, v in sp.items()}
        if corpus == a.combined:
            paths[fold] = {p.name: p for v in sp.values() for p in v}
        print(f"{corpus}/{fold}: " + ", ".join(f"{s} {len(v)}" for s, v in data[(corpus, fold)].items()), flush=True)
    corpora = sorted({c for c, _ in data})
    folds = sorted({f for _, f in data})

    out = Path(a.out)
    rows, summary = [], {"folds": {}, "corpora": corpora}
    print("\n== images of the combined test partition that a single-source model saw in training")
    for f in folds:
        comb = data.get((a.combined, f), {})
        test = comb.get("test", set())
        seen = set()
        row = {"fold": f, "combined_test": len(test)}
        for c in sources:
            d = data.get((c, f), {})
            tr, va, te = d.get("train", set()), d.get("val", set()), d.get("test", set())
            row[f"{c}_train"] = len(test & tr)
            row[f"{c}_val"] = len(test & va)
            row[f"{c}_test"] = len(test & te)
            seen |= (test & tr) | (test & va)
            print(f" {f}: {len(test & tr):6d} in {c} train, {len(test & va):5d} in {c} val, {len(test & te):6d} in {c} test")
        common = sorted(test - seen)
        row["common_test"] = len(common)
        rows.append(row)
        lst = out / f"common_test_{f}.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("\n".join(str(paths[f][n].resolve()) for n in common) + "\n")
        print(f" {f}: common test set (no model of this fold trained or validated on it): {len(common)} images -> {lst}")
        summary["folds"][f] = row
    write_csv(out / "fold_overlap.csv", rows)

    # origin of every image, from the corpus it appears in
    origin = {}
    for (c, f), d in data.items():
        if c == a.combined:
            continue
        for s in d.values():
            for n in s:
                origin.setdefault(n, c)
    allnames = {n for (c, f), d in data.items() if c == a.combined for s in d.values() for n in s}
    write_csv(out / "origin.csv", [{"image": n, "origin": origin.get(n, "unknown")} for n in sorted(allnames)],
              ["image", "origin"])
    unknown = sum(1 for n in allnames if n not in origin)
    print(f"\n== origin: {sum(1 for n in allnames if origin.get(n) == sources[0])} {sources[0]}, "
          f"{sum(1 for n in allnames if origin.get(n) == sources[1])} {sources[1]}, {unknown} unknown "
          f"(of {len(allnames)} images of {a.combined})")

    print("\n== test partitions across folds")
    for c in corpora:
        tests = {f: data[(c, f)].get("test", set()) for f in folds if (c, f) in data}
        inter = max((len(tests[x] & tests[y]) for x, y in itertools.combinations(tests, 2)), default=0)
        union = set().union(*tests.values()) if tests else set()
        allimg = set().union(*[s for (cc, f), d in data.items() if cc == c for s in d.values()])
        print(f" {c}: at most {inter} images shared between two fold test sets; union {len(union)} of {len(allimg)}")
        summary.setdefault("cross_fold", {})[c] = {"max_overlap": inter, "union": len(union), "corpus": len(allimg)}
    write_json(out / "folds_summary.json", summary)
    print(f"\nWrote {out}/fold_overlap.csv, origin.csv, common_test_<fold>.txt, folds_summary.json")


if __name__ == "__main__":
    main()
