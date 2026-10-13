#!/usr/bin/env python3
"""Create a tiny synthetic replica of the cross-validation layout of the study.

Images and labels live in one pool; every corpus (v1, v2 and their union v3) is split into
folds independently of the others, exactly as in the real experiments, and each fold is a
folder with its own data.yaml and three image lists. This is what makes the common test set
of a fold -- the images that no model of that fold trained or validated on -- non-trivial.

  <out>/pool/{images,labels}/            all images and their polygons
  <out>/folds/<corpus>/<fold>/data.yaml  path: <pool>, train/val/test: the lists beside it
  <out>/cls/{train,test}/<class>/*.jpg   classification corpus of the pipeline experiments

Used only by run_smoke_test.sh to check that every script runs end to end (on CPU, in a few
minutes) before the real jobs are submitted.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = ["SIGRE", "blue", "brown", "green", "green_point", "grey", "yellow"]
COLORS = [(200, 200, 255), (255, 80, 40), (30, 90, 140), (40, 180, 40), (0, 200, 255), (128, 128, 128), (0, 230, 230)]


def random_polygon(rng, w, h, scale):
    cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
    r = scale * min(w, h) * rng.uniform(0.5, 1.0)
    n = rng.integers(5, 12)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = r * rng.uniform(0.6, 1.0, n)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
    pts[:, 0] = np.clip(pts[:, 0], 1, w - 2)
    pts[:, 1] = np.clip(pts[:, 1], 1, h - 2)
    return pts


def draw_image(rng, w, h, objects):
    img = np.full((h, w, 3), rng.integers(20, 60), np.uint8)
    img += rng.integers(0, 25, (h, w, 3), dtype=np.uint8)
    for c, pts in objects:
        cv2.fillPoly(img, [pts.astype(np.int32)], COLORS[c])
    return img


def save_pool(pool, name, img, objects, w, h):
    cv2.imwrite(str(pool / "images" / name), img)
    lines = [f"{c} " + " ".join(f"{v:.5f}" for v in (pts / np.array([w, h])).reshape(-1)) for c, pts in objects]
    (pool / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")


def write_fold(root, pool, members, fold, seed, fractions=(0.4, 0.1)):
    """One fold of one corpus: three image lists and the data.yaml beside them."""
    d = root / f"fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = list(members)
    rng.shuffle(order)
    n = len(order)
    n_tr, n_va = int(fractions[0] * n), int(fractions[1] * n)
    splits = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    for s, items in splits.items():
        (d / f"{s}.txt").write_text("".join(f"{pool/'images'/x}\n" for x in items))
    (d / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(pool.resolve()), "train": "train.txt", "val": "val.txt", "test": "test.txt",
         "nc": len(NAMES), "names": NAMES}, sort_keys=False))
    return {s: len(v) for s, v in splits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n1", type=int, default=80, help="public-like images")
    ap.add_argument("--n2", type=int, default=80, help="generated-like images")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--ncls", type=int, default=14, help="classification images per class")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    pool = out / "pool"
    (pool / "images").mkdir(parents=True)
    (pool / "labels").mkdir(parents=True)

    members = {"v1": [], "v2": []}
    # v1: public-like, Roboflow names, several small objects, 640x640, some '.rf.' augmented copies
    for i in range(a.n1):
        w = h = 640
        objs = [(int(rng.choice([0, 0, 0, 1, 3, 4, 6])), random_polygon(rng, w, h, 0.08))
                for _ in range(int(rng.integers(1, 6)))]
        img = draw_image(rng, w, h, objs)
        stem = f"photo{i:03d}_jpg"
        for k in range(2 if i % 10 == 0 else 1):  # an augmented copy keeps the source stem
            name = f"{stem}.rf.{rng.integers(1e9):x}.jpg"
            im = img if k == 0 else img[:, ::-1].copy()
            ob = objs if k == 0 else [(c, np.stack([w - 1 - p[:, 0], p[:, 1]], 1)) for c, p in objs]
            save_pool(pool, name, im, ob, w, h)
            members["v1"].append(name)
    # v2: generated-like, one class per image, large objects, various sizes
    for i in range(a.n2):
        w, h = (480, 360) if i % 2 else (360, 480)
        c = int(rng.choice([2, 2, 2, 2, 0, 1, 3, 4, 5, 6]))
        k = 1 if c != 0 else int(rng.integers(1, 4))
        objs = [(c, random_polygon(rng, w, h, 0.3 if k == 1 else 0.12)) for _ in range(k)]
        name = f"{NAMES[c]}_{i:05d}.jpg"
        save_pool(pool, name, draw_image(rng, w, h, objs), objs, w, h)
        members["v2"].append(name)
    members["v3"] = members["v1"] + members["v2"]

    # every corpus folded independently of the others: different seeds, hence different partitions
    counts = {}
    for c, seed0 in (("v1", 100), ("v2", 200), ("v3", 300)):
        for k in range(a.folds):
            counts[(c, k)] = write_fold(out / "folds" / c, pool, members[c], k, seed0 + k)
    for c in ("v1", "v2", "v3"):
        print(f"  {c}: {len(members[c])} images, folds " + ", ".join(
            f"{counts[(c, k)]['train']}/{counts[(c, k)]['val']}/{counts[(c, k)]['test']}" for k in range(a.folds)))

    # classification corpus: <root>/<split>/<class>/*.jpg, one centred object per image
    for split, n in (("train", a.ncls), ("test", max(4, a.ncls // 2))):
        for c, name in enumerate(NAMES):
            d = out / "cls" / split / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                w, h = 400, 300
                cv2.imwrite(str(d / f"{name}_{split}_{i:04d}.jpg"),
                            draw_image(rng, w, h, [(c, random_polygon(rng, w, h, 0.35))]))
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main()
