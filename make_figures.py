#!/usr/bin/env python3
"""Figures for the manuscript.

--samples   two UN-annotated crops per class from the training partition of v3
            (one public-origin and one generated-origin image where possible):
            <out>/corpus_<class>_1.jpg (public) and <out>/corpus_<class>_2.jpg (generated).
--qualitative <id>
            predictions of one model on the common test partition: for each class the
            test image of median mask IoU among those where the top detection is correct
            (representative, not best-case) -> <out>/pred_<class>.jpg, plus the most
            confident misclassification -> <out>/pred_failure.jpg (details in pred_failure.json).

Example
-------
python make_figures.py --data /scratch/protocol/v3.yaml --origin /scratch/protocol/origin.csv \
    --gt /scratch/results/gt_v3_test.json.gz --pred /scratch/results/predictions/E13.json.gz \
    --samples --qualitative E13 --out /scratch/results/figures
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import class_names, label_path, load_canonical, load_yaml, read_csv, read_label, split_images

PALETTE = [(56, 56, 255), (255, 112, 31), (0, 128, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146)]


def crop_square(im, pts_list, margin=0.15, size=320):
    h, w = im.shape[:2]
    allp = np.concatenate(pts_list) * np.array([w, h])
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = max(side, 32)
    xa, ya = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    xb, yb = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    c = im[ya:yb, xa:xb]
    s = max(c.shape[:2])
    pad = np.zeros((s, s, 3), np.uint8)
    pad[(s - c.shape[0]) // 2:(s - c.shape[0]) // 2 + c.shape[0], (s - c.shape[1]) // 2:(s - c.shape[1]) // 2 + c.shape[1]] = c
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)


def contact_sheet(tiles, path, cols=4, size=220):
    """One numbered montage of candidate crops, so a human can choose at a glance."""
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (size + 22), cols * size, 3), 255, np.uint8)
    for k, im in enumerate(tiles):
        r, c = divmod(k, cols)
        y, x = r * (size + 22), c * size
        sheet[y + 22:y + 22 + size, x:x + size] = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, str(k + 1), (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _crop_of(p, size=320):
    im = load_canonical(p)
    pts = [pp for _, pp, _ in read_label(label_path(p))]
    return crop_square(im, pts, size=size)


def samples(data, origin_csv, out, seed, exclude=(), candidates=0, pool=60):
    """One crop per class and origin, plus optional ranked candidates to choose from.

    Many source images carry a stock-photography watermark, which cannot go into a figure.
    textscore.py ranks crops by how much text-like structure they contain; it is a useful
    ordering but not a detector, so with --candidates the script writes the best few per class
    and origin into candidates/ and the choice is made by looking at them.
    """
    from textscore import text_score
    names = class_names(load_yaml(data))
    origin = {r["image"]: r["origin"] for r in read_csv(origin_csv)} if origin_csv else {}  # keyed by file name
    skip = {Path(x).name for x in exclude}
    cand = defaultdict(lambda: defaultdict(list))
    for p in split_images(data)["train"]:
        if p.name in skip:
            continue
        inst = read_label(label_path(p))
        cls = {c for c, _, _ in inst}
        if len(cls) == 1:
            cand[cls.pop()][origin.get(p.name, "any")].append(p)
    rng = np.random.default_rng(seed)
    cdir = out / "candidates"
    if candidates:
        cdir.mkdir(parents=True, exist_ok=True)

    def best_of(paths, k):
        """The k crops with the least text-like structure, out of a random pool of `pool`."""
        sel = [paths[i] for i in rng.choice(len(paths), size=min(pool, len(paths)), replace=False)]
        scored = []
        for q in sel:
            try:
                crop = _crop_of(q)
            except Exception:
                continue
            scored.append((text_score(crop), q, crop))
        scored.sort(key=lambda x: x[0])
        return scored[:k]

    for c, n in enumerate(names):
        pools = cand.get(c, {})
        chosen = []
        for o in ("v1", "v2", "any"):
            if not pools.get(o):
                continue
            top = best_of(pools[o], max(1, candidates))
            if not top:
                continue
            chosen.append((top[0][1], top[0][2]))
            for k, (s, q, crop) in enumerate(top if candidates else []):
                cv2.imwrite(str(cdir / f"{n}_{o}_{k + 1}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (cdir / f"{n}_{o}_{k + 1}.txt").write_text(f"{q}\ntext score {s:.0f}\n")
            if candidates:
                contact_sheet([c for _, _, c in top], cdir / f"sheet_corpus_{n}_{o}.jpg")
            if len(chosen) == 2:
                break
        allp = [q for v in pools.values() for q in v]
        while len(chosen) < 2 and allp:
            q = allp[rng.integers(len(allp))]
            chosen.append((q, _crop_of(q)))
        for k, (q, crop) in enumerate(chosen[:2]):
            cv2.imwrite(str(out / f"corpus_{n}_{k + 1}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            # keep the provenance, so a crop can be excluded by name on the next run
            (out / f"corpus_{n}_{k + 1}.txt").write_text(f"{q}\n")
    print(f"corpus samples written to {out}" + (f"; {candidates} candidates per class and origin in {cdir}"
                                                if candidates else ""))


def draw(im, dets, names, gt=None):
    from pycocotools import mask as mu
    vis = im.copy()
    if gt is not None:
        for a in gt:
            m = mu.decode(a["segmentation"]).astype(np.uint8)
            cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)
    for d in dets:
        m = mu.decode(d["segmentation"]).astype(bool)
        col = np.array(PALETTE[(d["category_id"] - 1) % len(PALETTE)], np.uint8)
        vis[m] = (0.55 * vis[m] + 0.45 * col).astype(np.uint8)
        cnts = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(vis, cnts, -1, col.tolist(), 2)
        ys, xs = np.nonzero(m)
        txt = f"{names[d['category_id'] - 1]} {d['score']:.2f}"
        org = (int(xs.min()), max(18, int(ys.min()) - 6))
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def qualitative(gt_path, pred_path, out, show_thr, candidates=0, exclude=()):
    from pycocotools import mask as mu
    from cocoutils import load_json_gz
    gt = load_json_gz(gt_path)
    dets = load_json_gz(pred_path)
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    by_img_gt, by_img_dt = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        by_img_gt[a["image_id"]].append(a)
    for d in dets:
        by_img_dt[d["image_id"]].append(d)
    info = {im["id"]: im for im in gt["images"]}
    per_class = defaultdict(list)
    failures = []
    for iid, gts in by_img_gt.items():
        dts = sorted(by_img_dt.get(iid, []), key=lambda d: -d["score"])
        if not dts:
            continue
        g_rle = [a["segmentation"] for a in gts]
        d_rle = [d["segmentation"] for d in dts]
        ious = mu.iou(d_rle, g_rle, [0] * len(g_rle))  # [D, G]
        cls = {a["category_id"] for a in gts}
        top = dts[0]
        if len(cls) == 1 and top["category_id"] in cls and ious[0].max() >= 0.5:
            per_class[top["category_id"]].append((float(ious[0].max()), iid))
        for k, d in enumerate(dts):
            if d["score"] < show_thr:
                break
            same = [j for j, a in enumerate(gts) if a["category_id"] == d["category_id"]]
            other = [j for j, a in enumerate(gts) if a["category_id"] != d["category_id"]]
            if other and max(ious[k][j] for j in other) >= 0.5 and (not same or max(ious[k][j] for j in same) < 0.5):
                j = max(other, key=lambda j: ious[k][j])
                failures.append((d["score"], iid, d, gts[j]))
    skip = {Path(x).name for x in exclude}
    cdir = out / "candidates"
    if candidates:
        cdir.mkdir(parents=True, exist_ok=True)

    def panel(iid):
        im = load_canonical(info[iid]["file_name"])
        shown = [d for d in by_img_dt[iid] if d["score"] >= show_thr] \
            or sorted(by_img_dt[iid], key=lambda d: -d["score"])[:1]
        return draw(im, shown, names)

    for cid, lst in sorted(per_class.items()):
        lst.sort()
        lst = [(i, j) for i, j in lst if Path(info[j]["file_name"]).name not in skip]
        if not lst:
            continue
        mid = len(lst) // 2
        # candidates are taken around the median IoU, so they stay representative
        around = [lst[k] for k in range(max(0, mid - candidates // 2),
                                        min(len(lst), mid - candidates // 2 + candidates))] if candidates else []
        tiles = []
        for k, (iou, iid) in enumerate(around):
            pan = panel(iid)
            tiles.append(pan)
            cv2.imwrite(str(cdir / f"pred_{names[cid - 1]}_{k + 1}.jpg"), pan)
            (cdir / f"pred_{names[cid - 1]}_{k + 1}.txt").write_text(
                f"{info[iid]['file_name']}\niou {iou:.3f}\n")
        contact_sheet(tiles, cdir / f"sheet_pred_{names[cid - 1]}.jpg")
        cv2.imwrite(str(out / f"pred_{names[cid - 1]}.jpg"), panel(lst[mid][1]))
    if failures:
        failures.sort(key=lambda x: -x[0])
        s, iid, d, g = failures[0]
        im = load_canonical(info[iid]["file_name"])
        cv2.imwrite(str(out / "pred_failure.jpg"), draw(im, [d], names, gt=by_img_gt[iid]))
        (out / "pred_failure.json").write_text(json.dumps(
            {"image": info[iid]["file_name"], "predicted": names[d["category_id"] - 1], "score": d["score"],
             "true": names[g["category_id"] - 1]}, indent=2))
    print(f"qualitative figures written to {out} ({len(per_class)} classes, {len(failures)} confident misclassifications)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="v3 data YAML (for --samples)")
    ap.add_argument("--origin", default=None)
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--qualitative", default=None, help="experiment id, e.g. E13")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--show-thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=0,
                    help="also write this many alternatives per class into <out>/candidates, least text-like first")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated image file names never to choose, or a file holding one per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    excl = []
    if a.exclude:
        q = Path(a.exclude)
        excl = [x.strip() for x in (q.read_text().split() if q.exists() else a.exclude.split(",")) if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.samples:
        samples(a.data, a.origin, out, a.seed, exclude=excl, candidates=a.candidates)
    if a.qualitative:
        qualitative(a.gt, a.pred, out, a.show_thr, candidates=a.candidates, exclude=excl)


if __name__ == "__main__":
    main()
