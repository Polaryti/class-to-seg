#!/usr/bin/env python3
"""Build the common evaluation protocol and the control corpora.

What it does
------------
1. Finds the origin (v1 = public aggregation, v2 = generated) of every image in
   the combined corpus v3, by file name and, if needed, by MD5.
2. Checks v3 for train/test leakage: images derived from the same source photo
   (Roboflow '.rf.' copies, or identical difference hash) that fall in
   different partitions.
3. Defines the partitions:
     --mode derive  (default) v1 and v2 inherit the partition of v3, i.e.
                    v1_split = v1-origin images of v3_split. v3 is unchanged, so
                    the existing v3 checkpoints (E11-E17) stay valid and the v3
                    test partition is a common test set disjoint from every
                    training set.
     --mode resplit a new group-aware stratified 70/20/10 split of v3 (use this
                    only if the leakage check finds a problem; then every model,
                    including E11-E17, must be retrained).
4. Materialises (symlinks + label copies, no image duplication):
     v1, v2                 the two single-source corpora
     v3_sizematched         v3 train subsampled to the size of v1 train (control for corpus size)
     v3_deranged            v3 train in which every generated (v2-origin) image outside the
                            majority class carries a wrong class, class frequencies unchanged
                            (control for the transferred labels)
   and writes one data YAML per corpus in --out.
5. Writes origin.csv (used by evaluate.py to score v1-origin / v2-origin test subsets),
   leakage.csv and protocol_summary.json.

Example
-------
python prepare_protocol.py --v1 /data/v1/data.yaml --v2 /data/v2/data.yaml \
    --v3 /data/v3/data.yaml --out /scratch/waste/protocol
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import (class_names, dhash, label_path, load_yaml, md5sum, read_label, save_yaml,
                    split_images, split_sources, write_csv, write_json)

MARKER = ".created_by_prepare_protocol"
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


# ----------------------------------------------------------------------------
def rf_stem(name: str) -> str:
    """'IMG_12_jpg.rf.3fa9c.jpg' -> 'IMG_12_jpg' (Roboflow export copies share this stem)."""
    base = os.path.basename(name)
    if ".rf." in base:
        return base.split(".rf.", 1)[0]
    return os.path.splitext(base)[0]


def image_classes(img: Path) -> list[int]:
    return [c for c, _, _ in read_label(label_path(img))]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _dhash_safe(p):
    try:
        return dhash(p)
    except Exception:
        return None


def compute_dhashes(paths, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_dhash_safe, [str(p) for p in paths], chunksize=64))


# ----------------------------------------------------------------------------
def assign_origin(v3_imgs, v1_imgs, v2_imgs, origin_csv=None):
    """Return {v3_image_path_str: 'v1'|'v2'|'unknown'} and a small report."""
    report = Counter()
    if origin_csv:
        from common import read_csv
        table = {}
        for r in read_csv(origin_csv):
            table[os.path.basename(r["image"])] = r["origin"]
        out = {}
        for p in v3_imgs:
            o = table.get(p.name, "unknown")
            out[str(p)] = o
            report[f"csv_{o}"] += 1
        return out, dict(report)

    by_name = {"v1": defaultdict(list), "v2": defaultdict(list)}
    for p in v1_imgs:
        by_name["v1"][p.name].append(p)
    for p in v2_imgs:
        by_name["v2"][p.name].append(p)

    md5_index = None  # built lazily only if file names are not enough

    def build_md5_index():
        print("  building MD5 index of v1 and v2 (file names were not sufficient) ...", flush=True)
        idx = {}
        for o, imgs in (("v1", v1_imgs), ("v2", v2_imgs)):
            for p in imgs:
                idx.setdefault(md5sum(p), set()).add(o)
        return idx

    out = {}
    for p in v3_imgs:
        in1, in2 = p.name in by_name["v1"], p.name in by_name["v2"]
        if in1 != in2:
            out[str(p)] = "v1" if in1 else "v2"
            report["by_name"] += 1
            continue
        if md5_index is None:
            md5_index = build_md5_index()
        origins = md5_index.get(md5sum(p), set())
        if len(origins) == 1:
            out[str(p)] = next(iter(origins))
            report["by_md5"] += 1
        elif len(origins) == 2:
            out[str(p)] = "v1"          # identical file in both sources: count it once, as public data
            report["in_both_assigned_v1"] += 1
        else:
            out[str(p)] = "unknown"
            report["unknown"] += 1
    return out, dict(report)


# ----------------------------------------------------------------------------
def leakage_groups(v3_split_of, class_freq, use_dhash, workers, max_bucket):
    """Group v3 images that derive from the same source photo."""
    paths = list(v3_split_of)
    n = len(paths)
    uf = UnionFind(n)
    first = {}
    for i, p in enumerate(paths):
        k = rf_stem(p)
        if k in first:
            uf.union(first[k], i)
        else:
            first[k] = i
    skipped_buckets = 0
    if use_dhash:
        print(f"  computing difference hashes of {n} images with {workers} workers ...", flush=True)
        hashes = compute_dhashes(paths, workers)
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            if h is not None:
                buckets[h].append(i)
        for h, idx in buckets.items():
            if len(idx) > max_bucket:          # usually blank / uniform images
                skipped_buckets += 1
                continue
            for j in idx[1:]:
                uf.union(idx[0], j)
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(paths[i])
    return list(groups.values()), skipped_buckets


def rarest_class(classes, class_freq):
    if not classes:
        return -1
    return min(set(classes), key=lambda c: (class_freq.get(c, 0), c))


def group_resplit(groups, origin, classes_of, class_freq, fractions, seed):
    """Group-aware stratified split. Returns {image: split}."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for g in groups:
        cls = [c for p in g for c in classes_of[p]]
        key = (origin[g[0]], rarest_class(cls, class_freq))
        strata[key].append(g)
    split_of = {}
    for key in sorted(strata, key=str):
        gs = strata[key]
        order = rng.permutation(len(gs))
        total = sum(len(g) for g in gs)
        c = 0
        for gi in order:
            g = gs[gi]
            mid = (c + len(g) / 2) / max(total, 1)
            s = "train" if mid < fractions[0] else ("val" if mid < fractions[0] + fractions[1] else "test")
            for p in g:
                split_of[p] = s
            c += len(g)
    return split_of


def stratified_subsample(items, keys, n_target, rng):
    """Proportional stratified sample without replacement (largest-remainder rounding)."""
    strata = defaultdict(list)
    for it, k in zip(items, keys):
        strata[k].append(it)
    N = len(items)
    if n_target >= N:
        return list(items)
    names = sorted(strata, key=str)
    exact = np.array([len(strata[k]) * n_target / N for k in names])
    quota = np.floor(exact).astype(int)
    rem = n_target - quota.sum()
    for i in np.argsort(-(exact - quota))[:rem]:
        quota[i] += 1
    out = []
    for k, q in zip(names, quota):
        pool = strata[k]
        q = min(q, len(pool))
        idx = rng.choice(len(pool), size=q, replace=False)
        out.extend(pool[i] for i in sorted(idx))
    return out


def derange_labels(img_class, exclude, rng):
    """Permute image-level classes so that every eligible image gets a different class.

    img_class: {image: class}. Images of class `exclude` are left untouched.
    Returns {image: new_class}; the multiset of classes is preserved exactly.
    """
    groups = defaultdict(list)
    for p, c in img_class.items():
        if c != exclude:
            groups[c].append(p)
    order = list(groups)
    rng.shuffle(order)
    seq = []
    for c in order:
        members = list(groups[c])
        rng.shuffle(members)
        seq.extend((p, c) for p in members)
    n = len(seq)
    if n == 0:
        return {}
    m = max(len(v) for v in groups.values())
    if 2 * m > n:
        raise RuntimeError(
            f"Cannot derange: largest eligible class has {m} of {n} images (>50%). "
            f"Use --derange-exclude to exclude that class as well.")
    return {seq[i][0]: seq[(i + m) % n][1] for i in range(n)}


# ----------------------------------------------------------------------------
def prepare_corpus_dir(root: Path):
    if root.exists():
        if not (root / MARKER).exists() and any(root.iterdir()):
            sys.exit(f"Refusing to overwrite {root}: it exists and was not created by this script.")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / MARKER).write_text("safe to delete\n")


def place_image(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_abs = src.resolve()
    if mode == "symlink":
        os.symlink(src_abs, dst)
    elif mode == "hardlink":
        try:
            os.link(src_abs, dst)
        except OSError:
            shutil.copy2(src_abs, dst)
    else:
        shutil.copy2(src_abs, dst)


def materialise(root: Path, split_members: dict, mode: str, relabel: dict | None = None):
    """Create root/<split>/images + labels. relabel: {image: new_class} for all instances of that image."""
    relabel = relabel or {}
    mapping = []
    for s, imgs in split_members.items():
        used = set()
        for p in imgs:
            p = Path(p)
            name = p.name
            k = 1
            while name in used:                     # avoid basename collisions
                name = f"{p.stem}__{k}{p.suffix}"
                k += 1
            used.add(name)
            dst_img = root / SPLIT_DIR[s] / "images" / name
            place_image(p, dst_img, mode)
            src_lab = label_path(p)
            dst_lab = root / SPLIT_DIR[s] / "labels" / (Path(name).stem + ".txt")
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            if str(p) in relabel:
                new_c = relabel[str(p)]
                lines = []
                for line in src_lab.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        parts[0] = str(new_c)
                        lines.append(" ".join(parts))
                dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""))
            elif src_lab.exists():
                shutil.copyfile(src_lab, dst_lab)
            mapping.append({"split": s, "image": str(dst_img), "source": str(p)})
    return mapping


def yaml_for_tree(root: Path, names, splits=("train", "val", "test"), overrides=None):
    d = {"path": str(root.resolve())}
    for s in splits:
        d[s] = f"{SPLIT_DIR[s]}/images"
    for k, v in (overrides or {}).items():
        d[k] = v
    d["nc"] = len(names)
    d["names"] = list(names)
    return d


def warmup_cache(yaml_path: Path):
    """Build the Ultralytics label caches once, so that parallel training jobs do not race to write them."""
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset
    except Exception as e:  # ultralytics not installed in this environment
        print(f"  [cache warm-up skipped: {e}]")
        return
    try:
        data = check_det_dataset(str(yaml_path))
    except Exception as e:
        print(f"  [cache warm-up skipped for {yaml_path.name}: {e}]")
        return
    for s in ("train", "val", "test"):
        if not data.get(s):
            continue
        try:
            YOLODataset(img_path=data[s], data=data, task="segment", augment=False, prefix=f"{yaml_path.stem}/{s}: ")
        except Exception as e:
            print(f"  [cache warm-up failed for {yaml_path.name}/{s} (training will build it): {e}]")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True, help="data YAML of the public aggregation (as used for E1-E5)")
    ap.add_argument("--v2", required=True, help="data YAML of the generated corpus (as used for E6-E10)")
    ap.add_argument("--v3", required=True, help="data YAML of the combined corpus (as used for E11-E17)")
    ap.add_argument("--out", required=True, help="output directory (YAMLs, trees, reports)")
    ap.add_argument("--mode", choices=["derive", "resplit"], default="derive")
    ap.add_argument("--fractions", default="0.7,0.2,0.1", help="train,val,test fractions for --mode resplit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--origin-csv", default=None, help="optional CSV with columns image,origin (v1|v2)")
    ap.add_argument("--allow-unmatched", action="store_true", help="drop v3 images of unknown origin instead of stopping")
    ap.add_argument("--no-dhash", action="store_true", help="leakage check by Roboflow file stem only")
    ap.add_argument("--dhash-max-bucket", type=int, default=50)
    ap.add_argument("--derange-exclude", default=None,
                    help="class name(s) excluded from the label derangement (comma separated); default: the majority class of v2")
    ap.add_argument("--no-controls", action="store_true", help="do not build v3_sizematched / v3_deranged")
    ap.add_argument("--no-warmup", action="store_true", help="do not pre-build the Ultralytics label caches")
    ap.add_argument("--force", action="store_true", help="rebuild even if the protocol already exists "
                    "(never do this while jobs are using it)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if (out / "protocol_summary.json").exists() and not a.force:
        print(f"{out} already contains a protocol; nothing done (use --force to rebuild, never while jobs use it).")
        return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    names3 = class_names(load_yaml(a.v3))
    for other in (a.v1, a.v2):
        n_o = class_names(load_yaml(other))
        if n_o != names3:
            print(f"WARNING: class names of {other} differ from v3: {n_o} vs {names3}. "
                  f"The v3 labels (and names) are used for every derived corpus.")

    print("Listing images ...", flush=True)
    s1, s2, s3 = split_images(a.v1), split_images(a.v2), split_images(a.v3)
    v1_imgs = [p for v in s1.values() for p in v]
    v2_imgs = [p for v in s2.values() for p in v]
    v3_split_of = {str(p): s for s, v in s3.items() for p in v}
    print(f"  v1: {len(v1_imgs)} images, v2: {len(v2_imgs)} images, v3: {len(v3_split_of)} images "
          f"({ {s: len(v) for s, v in s3.items()} })")

    print("Assigning origin to v3 images ...", flush=True)
    origin, origin_report = assign_origin([Path(p) for p in v3_split_of], v1_imgs, v2_imgs, a.origin_csv)
    print(f"  {origin_report}")
    unknown = [p for p, o in origin.items() if o not in ("v1", "v2")]
    if unknown:
        msg = f"{len(unknown)} v3 images could not be traced to v1 or v2, e.g. {unknown[:3]}"
        if not a.allow_unmatched:
            sys.exit("ERROR: " + msg + ". Provide --origin-csv or use --allow-unmatched to drop them.")
        print("WARNING: " + msg + " -> dropped.")
        for p in unknown:
            v3_split_of.pop(p)

    classes_of = {p: image_classes(Path(p)) for p in v3_split_of}
    class_freq = Counter(c for cs in classes_of.values() for c in cs)

    print("Checking leakage between partitions ...", flush=True)
    groups, skipped = leakage_groups(v3_split_of, class_freq, not a.no_dhash, a.workers, a.dhash_max_bucket)
    leak_rows, n_leak_test = [], 0
    for gi, g in enumerate(groups):
        splits = {v3_split_of[p] for p in g}
        if len(splits) > 1:
            for p in g:
                leak_rows.append({"group": gi, "image": p, "split": v3_split_of[p], "origin": origin[p]})
            if "test" in splits and "train" in splits:
                n_leak_test += sum(1 for p in g if v3_split_of[p] == "test")
    write_csv(out / "leakage.csv", leak_rows, ["group", "image", "split", "origin"])
    n_groups_multi = len({r["group"] for r in leak_rows})
    print(f"  {n_groups_multi} source groups span more than one partition; "
          f"{n_leak_test} test images share a source with a training image "
          f"(dHash buckets skipped as uniform: {skipped}). Details: {out/'leakage.csv'}")

    # ---------------------------------------------------------------- partition
    if a.mode == "derive":
        split_of = dict(v3_split_of)
    else:
        fr = [float(x) for x in a.fractions.split(",")]
        assert len(fr) == 3 and abs(sum(fr) - 1) < 1e-6, "--fractions must be three numbers summing to 1"
        split_of = group_resplit(groups, origin, classes_of, class_freq, fr, a.seed)

    members = {c: {s: [] for s in ("train", "val", "test")} for c in ("v1", "v2", "v3")}
    for p in sorted(split_of):
        s = split_of[p]
        members["v3"][s].append(p)
        members[origin[p]][s].append(p)

    all_maps = []
    yamls = {}
    for c in ("v1", "v2"):
        root = out / "corpora" / c
        prepare_corpus_dir(root)
        all_maps += [dict(m, corpus=c) for m in materialise(root, members[c], a.link)]
        yamls[c] = yaml_for_tree(root, names3)

    if a.mode == "derive":
        v3_yaml = {}  # the original v3 split directories, as absolute paths
        for s, srcs in split_sources(a.v3).items():
            v3_yaml[s] = [str(x) for x in srcs] if len(srcs) > 1 else str(srcs[0])
        v3_yaml["nc"] = len(names3)
        v3_yaml["names"] = list(names3)
        v3_valtest = {"val": v3_yaml["val"], "test": v3_yaml["test"]}
    else:
        root = out / "corpora" / "v3"
        prepare_corpus_dir(root)
        all_maps += [dict(m, corpus="v3") for m in materialise(root, members["v3"], a.link)]
        v3_yaml = yaml_for_tree(root, names3)
        v3_valtest = {"val": str(root / "valid" / "images"), "test": str(root / "test" / "images")}
    yamls["v3"] = v3_yaml

    summary = {"mode": a.mode, "seed": a.seed, "origin_assignment": origin_report,
               "leakage": {"groups_spanning_partitions": n_groups_multi,
                           "test_images_sharing_source_with_train": n_leak_test,
                           "uniform_dhash_buckets_skipped": skipped},
               "names": names3, "corpora": {}}

    # ---------------------------------------------------------------- controls
    if not a.no_controls:
        # (a) size-matched: v3 train subsampled to |v1 train| images, stratified by origin and rarest class
        tr = members["v3"]["train"]
        keys = [(origin[p], rarest_class(classes_of[p], class_freq)) for p in tr]
        n_target = len(members["v1"]["train"])
        sub = stratified_subsample(tr, keys, n_target, rng)
        root = out / "corpora" / "v3_sizematched"
        prepare_corpus_dir(root)
        all_maps += [dict(m, corpus="v3_sizematched") for m in materialise(root, {"train": sub}, a.link)]
        yamls["v3_sizematched"] = yaml_for_tree(root, names3, splits=("train",), overrides=v3_valtest)
        summary["corpora"]["v3_sizematched_note"] = (
            f"{len(sub)} of {len(tr)} v3 training images "
            f"({sum(1 for p in sub if origin[p] == 'v1')} v1-origin, {sum(1 for p in sub if origin[p] == 'v2')} v2-origin)")

        # (b) deranged labels for v2-origin training images
        img_class = {}
        for p in tr:
            if origin[p] == "v2" and classes_of[p]:
                cnt = Counter(classes_of[p])
                img_class[p] = min(cnt, key=lambda c: (-cnt[c], c))
        maj_counts = Counter(img_class.values())
        if a.derange_exclude:
            excl_names = [x.strip() for x in a.derange_exclude.split(",")]
            excl = [names3.index(x) for x in excl_names]
        else:
            excl = [maj_counts.most_common(1)[0][0]] if maj_counts else []
        eligible = {p: c for p, c in img_class.items() if c not in excl}
        relabel = derange_labels(eligible, exclude=None, rng=rng)
        root = out / "corpora" / "v3_deranged"
        prepare_corpus_dir(root)
        all_maps += [dict(m, corpus="v3_deranged") for m in materialise(root, {"train": tr}, a.link, relabel)]
        yamls["v3_deranged"] = yaml_for_tree(root, names3, splits=("train",), overrides=v3_valtest)
        n_inst_changed = sum(len(classes_of[p]) for p in relabel)
        summary["corpora"]["v3_deranged_note"] = {
            "excluded_classes": [names3[c] for c in excl],
            "v2_train_images": len(img_class),
            "images_relabelled": len(relabel),
            "instances_relabelled": n_inst_changed,
            "all_relabelled_changed_class": all(relabel[p] != eligible[p] for p in relabel)}

    # ---------------------------------------------------------------- write
    for name, d in yamls.items():
        save_yaml(d, out / f"{name}.yaml")
    rows = [{"image": p, "split": split_of[p], "origin": origin[p]} for p in sorted(split_of)]
    write_csv(out / "origin.csv", rows, ["image", "split", "origin"])
    write_csv(out / "materialised_files.csv", all_maps, ["corpus", "split", "image", "source"])

    for c in ("v1", "v2", "v3"):
        summary["corpora"][c] = {s: {"images": len(v),
                                     "instances": sum(len(classes_of[p]) for p in v)}
                                 for s, v in members[c].items()}
    write_json(out / "protocol_summary.json", summary)
    print(f"Wrote YAMLs to {out}: {', '.join(sorted(yamls))}")
    for c in ("v1", "v2", "v3"):
        print(f"  {c}: " + ", ".join(f"{s} {len(v)}" for s, v in members[c].items()))

    if not a.no_warmup:
        print("Pre-building Ultralytics label caches ...", flush=True)
        for name in yamls:
            warmup_cache(out / f"{name}.yaml")
    print("Done.")


if __name__ == "__main__":
    main()
