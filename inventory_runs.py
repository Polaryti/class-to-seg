#!/usr/bin/env python3
"""Inventory of existing training runs (any folder with an args.yaml), and of the data splits they used.

For each run: data YAML, model, seed, epochs configured and run, best validation mask
mAP50-95, whether it finished, and the Ultralytics version. For each data YAML used by
some run: images per split and how its partitions overlap with the partitions of --v3
(the split the new protocol is derived from). This tells which runs (e.g. which fold) were
trained on the same split as --v3, i.e. never saw its test images.

Example
-------
python inventory_runs.py --root /data/user/exp-to-clean/runs --v3 /data/user/datasets/v3/data.yaml --out inventory
"""
import argparse
import itertools
import os
from collections import defaultdict
from pathlib import Path

from common import _ultralytics_datasets_dir, load_yaml, read_csv, split_images, torch_load, write_csv


def run_info(d: Path, check_ckpt: bool):
    a = load_yaml(d / "args.yaml") or {}
    r = {"run": str(d), "data": a.get("data"), "model": Path(str(a.get("model", ""))).name, "seed": a.get("seed"),
         "epochs": a.get("epochs"), "batch": a.get("batch"), "imgsz": a.get("imgsz"), "optimizer": a.get("optimizer"),
         "copy_paste": a.get("copy_paste"), "fraction": a.get("fraction")}
    try:
        rows = [{k.strip(): v for k, v in x.items()} for x in read_csv(d / "results.csv")]
        r["epochs_run"] = len(rows)
        vals = [float(x.get("metrics/mAP50-95(M)", "nan") or "nan") for x in rows]
        fit = [0.1 * float(x.get("metrics/mAP50(B)", 0) or 0) + 0.9 * float(x.get("metrics/mAP50-95(B)", 0) or 0)
               + 0.1 * float(x.get("metrics/mAP50(M)", 0) or 0) + 0.9 * float(x.get("metrics/mAP50-95(M)", 0) or 0)
               for x in rows]
        if rows:
            b = max(range(len(fit)), key=lambda i: fit[i])
            r["best_epoch"] = b + 1
            r["best_val_mask_mAP50_95"] = round(vals[b], 4)
    except Exception:
        r["epochs_run"] = 0
    r["best_pt"] = (d / "weights" / "best.pt").exists()
    r["last_pt"] = (d / "weights" / "last.pt").exists()
    if check_ckpt and r["last_pt"]:
        try:
            ck = torch_load(d / "weights" / "last.pt")
            r["finished"] = ck.get("epoch", -1) == -1
            r["ultralytics"] = ck.get("version")
        except Exception as e:
            r["finished"] = f"unreadable: {e}"
    return r


def resolve_data(rel: str, roots) -> Path | None:
    """Training args store the data YAML as it was given, often relative to the folder the job ran in."""
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    for r in roots:
        for cand in (r / p, *r.glob(f"*/{p}"), *r.glob(f"*/*/{p}")):
            if cand.exists():
                return cand
    return None


def split_names(yaml_path):
    try:
        return {s: {p.name for p in v} for s, v in split_images(yaml_path).items()}, None
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder that contains the run folders (searched recursively)")
    ap.add_argument("--v3", required=True, help="data YAML of v3 as set in config.sh (V3_YAML)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ckpt", action="store_true", help="do not open last.pt (faster)")
    ap.add_argument("--data-root", default=None,
                    help="folder the training jobs ran in, i.e. the one the data YAML paths in args.yaml are relative to")
    a = ap.parse_args()

    if a.data_root:  # also used to resolve the images / .txt lists inside the data YAMLs
        os.environ["WASTE_EXTRA_DATA_ROOTS"] = os.pathsep.join(
            [a.data_root] + [x for x in os.environ.get("WASTE_EXTRA_DATA_ROOTS", "").split(os.pathsep) if x])
    runs = sorted({Path(r) for r, _, fs in os.walk(a.root) if "args.yaml" in fs})
    print(f"{len(runs)} runs under {a.root}", flush=True)
    rows = [run_info(d, not a.no_ckpt) for d in runs]
    out = Path(a.out)
    write_csv(out / "runs.csv", rows)

    v3, err = split_names(a.v3)
    if v3 is None:
        raise SystemExit(f"cannot read --v3 {a.v3}: {err}")
    root = Path(a.root).resolve()
    roots = [Path(a.data_root).resolve()] if a.data_root else []
    roots += [root, *root.parents[:3], Path.cwd()]
    ds = _ultralytics_datasets_dir()
    if ds is not None:
        roots += [ds, ds.parent]
    resolved = {}
    ds_rows = []
    for data in sorted({str(r["data"]) for r in rows if r.get("data")}):
        found = resolve_data(data, roots)
        resolved[data] = found
        sp, err = split_names(found) if found else (None, "not found (pass --data-root)")
        row = {"data": data, "resolved": str(found) if found else "", "runs": sum(1 for r in rows if str(r.get("data")) == data)}
        if sp is None:
            row["error"] = err
        else:
            for s in ("train", "val", "test"):
                row[f"{s}_images"] = len(sp.get(s, ()))
            row["same_partition_as_v3"] = all(sp.get(s, set()) == v3.get(s, set()) for s in ("train", "val", "test"))
            row["v3_test_images_in_its_train"] = len(v3.get("test", set()) & sp.get("train", set()))
            row["v3_test_images_in_its_val"] = len(v3.get("test", set()) & sp.get("val", set()))
        ds_rows.append(row)
    write_csv(out / "datasets.csv", ds_rows)

    # readable summary, grouped by the first folder level under --root
    groups = defaultdict(list)
    for r in rows:
        rel = Path(r["run"]).relative_to(a.root)
        groups[rel.parts[0]].append((rel, r))
    print("\n== runs")
    for g in sorted(groups):
        print(f"{g}: {len(groups[g])} run(s)")
        for rel, r in groups[g]:
            state = {True: "finished", False: "interrupted"}.get(r.get("finished"), r.get("finished", "?"))
            print(f"   {str(rel):28s} ep {r.get('epochs_run')}/{r.get('epochs')} best@{r.get('best_epoch')} "
                  f"val mAP50-95(M) {r.get('best_val_mask_mAP50_95')}  {state}  best.pt={r['best_pt']}  "
                  f"seed={r.get('seed')} model={r.get('model')} cp={r.get('copy_paste')}  data={r.get('data')}")
    print("\n== data splits used by these runs, compared with V3_YAML")
    for d in ds_rows:
        if "error" in d:
            print(f" {d['data']}  ({d['runs']} runs)  NOT READABLE: {d['error']}")
        else:
            print(f" {d['data']}  ({d['runs']} runs)  train/val/test {d['train_images']}/{d['val_images']}/{d['test_images']}  "
                  f"same as V3_YAML: {d['same_partition_as_v3']}  V3 test images in its train: {d['v3_test_images_in_its_train']}")
    # ---------------------------------------------------------------- folds
    # data paths of the form .../<corpus>/<fold>/data.yaml are grouped into a corpus x fold table
    sets = {}
    for data, found in resolved.items():
        if not found:
            continue
        sp, err = split_names(found)
        if sp is None:
            continue
        parts = Path(data).parts
        if len(parts) >= 3:
            sets[(parts[-3], parts[-2])] = sp
    if sets:
        corpora = sorted({c for c, _ in sets})
        folds = sorted({f for _, f in sets})
        print("\n== images per split (corpus x fold)")
        print(f"{'':10s}" + "".join(f"{f:>22s}" for f in folds))
        for c in corpora:
            line = f"{c:10s}"
            for f in folds:
                sp = sets.get((c, f))
                line += f"{'-':>22s}" if sp is None else f"{len(sp.get('train', ())):>8}/{len(sp.get('val', ())):>6}/{len(sp.get('test', ())):>6}"
            print(line)
        print("\n== checks")
        for f in folds:
            v1, v2, v3 = sets.get(("v1", f)), sets.get(("v2", f)), sets.get(("v3", f))
            if v1 and v2 and v3:
                for s in ("train", "val", "test"):
                    u = v1.get(s, set()) | v2.get(s, set())
                    ok = u == v3.get(s, set())
                    print(f" {f} {s:5s}: v1+v2 == v3 ? {ok}" + ("" if ok else
                          f"  (only in v1+v2: {len(u - v3.get(s, set()))}, only in v3: {len(v3.get(s, set()) - u)})"))
            if v3:
                print(f" {f} test images also in its own train: {len(v3.get('test', set()) & v3.get('train', set()))}, "
                      f"in its own val: {len(v3.get('test', set()) & v3.get('val', set()))}")
        for c in corpora:
            fs = [(f, sets[(c, f)]) for f in folds if (c, f) in sets]
            tests = {f: sp.get("test", set()) for f, sp in fs}
            inter = max((len(tests[a] & tests[b]) for a, b in itertools.combinations(tests, 2)), default=0)
            union = set().union(*tests.values()) if tests else set()
            allimg = set().union(*[sp.get(s, set()) for _, sp in fs for s in ("train", "val", "test")]) if fs else set()
            print(f" {c}: test sets overlap at most {inter} images between folds; their union is {len(union)} "
                  f"of {len(allimg)} images of the corpus")

    print(f"\nDetails: {out/'runs.csv'}, {out/'datasets.csv'}")


if __name__ == "__main__":
    main()
