"""COCO-style mask evaluation shared by every model (YOLO and Mask R-CNN).

* Ground truth: YOLO polygon labels rasterised at the canonical scale (long side
  640 px), stored as RLE, category_id = class index + 1.
* Metrics: pycocotools 'segm' evaluation (AP, AP50, AP75, AP_S/M/L, AR100,
  per-class AP) plus precision and recall at the confidence that maximises the
  mean F1 over classes at IoU 0.50 (the operating point Ultralytics reports).
* Paired bootstrap over test images: a weighted re-implementation of
  COCOeval.accumulate (area 'all', 100 detections) that reproduces pycocotools
  exactly when every weight is 1 (checked by selftest()).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from common import canonical_size, image_size, json_default, label_path, read_label

try:
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:  # pragma: no cover
    raise ImportError("pycocotools is required: pip install pycocotools") from e

REC_THRS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
IOU_THRS = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)


# ----------------------------------------------------------------------------
# Ground truth / predictions I/O
# ----------------------------------------------------------------------------
def rle_encode(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def build_gt(images: list, names: list[str], workers: int = 1) -> tuple[dict, dict]:
    """COCO ground-truth dict for a list of image paths (ids = position + 1).

    With workers > 1 the image headers and label files are read in parallel, which is what
    dominates on a shared filesystem; the rasterisation itself stays in this thread.
    """
    imgs, anns, stats = [], [], {"instances": 0, "skipped_empty": 0, "skipped_bad_class": 0, "box_lines": 0}
    per_class = {n: 0 for n in names}
    aid = 1
    if workers > 1 and len(images) > 64:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            read = list(ex.map(lambda q: (image_size(q), read_label(label_path(q))), images, chunksize=32))
    else:
        read = [(image_size(q), read_label(label_path(q))) for q in images]
    for i, p in enumerate(images):
        (w0, h0), instances = read[i]
        w, h = canonical_size(w0, h0)
        imgs.append({"id": i + 1, "file_name": str(p), "width": w, "height": h})
        for c, pts, is_box in instances:
            stats["instances"] += 1
            stats["box_lines"] += int(is_box)
            if not 0 <= c < len(names):
                stats["skipped_bad_class"] += 1
                continue
            m = np.zeros((h, w), np.uint8)
            poly = np.round(pts * np.array([w, h])).astype(np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(m, [poly], 1)
            area = int(m.sum())
            if area == 0:
                stats["skipped_empty"] += 1
                continue
            ys, xs = np.nonzero(m)
            x0, y0 = int(xs.min()), int(ys.min())
            anns.append({"id": aid, "image_id": i + 1, "category_id": c + 1, "segmentation": rle_encode(m),
                         "area": area, "bbox": [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
                         "iscrowd": 0})
            aid += 1
            per_class[names[c]] += 1
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    stats["annotations_kept"] = len(anns)
    stats["per_class"] = per_class
    return {"info": {"description": "canonical long side 640"}, "images": imgs, "annotations": anns,
            "categories": cats}, stats


def save_json_gz(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, default=json_default)


def load_json_gz(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def coco_from_dict(d: dict) -> COCO:
    c = COCO()
    c.dataset = d
    c.createIndex()
    return c


def masks_to_detections(image_id: int, masks: np.ndarray, classes, scores, target_hw=None) -> list[dict]:
    """masks: [N, H, W] bool; classes: 0-based; returns COCO result dicts."""
    out = []
    for m, c, s in zip(masks, classes, scores):
        if target_hw is not None and m.shape != tuple(target_hw):
            m = cv2.resize(m.astype(np.uint8), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        if not m.any():
            continue
        out.append({"image_id": int(image_id), "category_id": int(c) + 1, "segmentation": rle_encode(m),
                    "score": float(s)})
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_cocoeval(coco_gt: COCO, dets: list, img_ids=None) -> COCOeval | None:
    img_ids = sorted(img_ids) if img_ids is not None else sorted(coco_gt.getImgIds())
    keep = set(img_ids)
    dets = [d for d in dets if d["image_id"] in keep]
    if not dets:
        return None
    coco_dt = coco_gt.loadRes(dets)
    E = COCOeval(coco_gt, coco_dt, "segm")
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    return E


def _mean_valid(x):
    x = x[x > -1]
    return float(np.mean(x)) if x.size else float("nan")


def summarise(E: COCOeval | None, names: list[str]) -> tuple[dict, list[dict]]:
    """Overall metrics and per-class AP from an accumulated COCOeval (maxDets=100)."""
    if E is None:
        m = {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "AP_S", "AP_M", "AP_L", "AR100", "P", "R", "F1", "conf")}
        return m, [{"class": n, "AP50_95": 0.0, "AP50": 0.0, "n_gt": 0} for n in names]
    pr = E.eval["precision"]  # [T, R, K, A, M]
    rc = E.eval["recall"]     # [T, K, A, M]
    m_idx = len(E.params.maxDets) - 1
    out = {"mAP50_95": _mean_valid(pr[:, :, :, 0, m_idx]),
           "mAP50": _mean_valid(pr[0, :, :, 0, m_idx]),
           "mAP75": _mean_valid(pr[5, :, :, 0, m_idx]),
           "AP_S": _mean_valid(pr[:, :, :, 1, m_idx]),
           "AP_M": _mean_valid(pr[:, :, :, 2, m_idx]),
           "AP_L": _mean_valid(pr[:, :, :, 3, m_idx]),
           "AR100": _mean_valid(rc[:, :, 0, m_idx])}
    arrays = prepare_arrays(E)
    w = np.ones(len(E.params.imgIds))
    wm = weighted_metrics(arrays, w)
    out.update({"P": wm["P"], "R": wm["R"], "F1": wm["F1"], "conf": wm["conf"]})
    per = []
    cat_ids = E.params.catIds
    for k, cid in enumerate(cat_ids):
        n_gt = int(sum(1 for a in E.cocoGt.loadAnns(E.cocoGt.getAnnIds(imgIds=E.params.imgIds, catIds=[cid]))))
        per.append({"class": names[cid - 1], "AP50_95": _mean_valid(pr[:, :, k, 0, m_idx]),
                    "AP50": _mean_valid(pr[0, :, k, 0, m_idx]),
                    "AP_S": _mean_valid(pr[:, :, k, 1, m_idx]), "AP_M": _mean_valid(pr[:, :, k, 2, m_idx]),
                    "AP_L": _mean_valid(pr[:, :, k, 3, m_idx]),
                    "P": wm["per_class_P"].get(k, float("nan")), "R": wm["per_class_R"].get(k, float("nan")),
                    "n_gt": n_gt})
    return out, per


# ----------------------------------------------------------------------------
# Weighted accumulate (bootstrap)
# ----------------------------------------------------------------------------
def prepare_arrays(E: COCOeval) -> dict:
    """Per-class detection arrays (area 'all', last maxDet) sorted by score, as in COCOeval.accumulate."""
    p = E.params
    K, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
    maxdet = p.maxDets[-1]
    a_idx = [i for i, a in enumerate(p.areaRngLbl) if a == "all"][0]
    classes = {}
    for k in range(K):
        scores, matches, ignore, img_idx = [], [], [], []
        npig = np.zeros(I)
        base = k * A * I + a_idx * I
        for i in range(I):
            e = E.evalImgs[base + i]
            if e is None:
                continue
            s = np.asarray(e["dtScores"][:maxdet])
            scores.append(s)
            matches.append(np.asarray(e["dtMatches"])[:, :maxdet])
            ignore.append(np.asarray(e["dtIgnore"])[:, :maxdet])
            img_idx.append(np.full(len(s), i))
            npig[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
        if not scores:
            classes[k] = None
            continue
        s = np.concatenate(scores)
        order = np.argsort(-s, kind="mergesort")
        dtm = np.concatenate(matches, axis=1)[:, order] if s.size else np.zeros((len(p.iouThrs), 0))
        dti = np.concatenate(ignore, axis=1)[:, order] if s.size else np.zeros((len(p.iouThrs), 0))
        classes[k] = {"scores": s[order], "tp": np.logical_and(dtm, np.logical_not(dti)),
                      "fp": np.logical_and(np.logical_not(dtm), np.logical_not(dti)),
                      "img": np.concatenate(img_idx)[order].astype(int), "npig": npig}
    return {"classes": classes, "n_images": I, "rec_thrs": np.asarray(p.recThrs), "K": K}


def _smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    pad = np.ones(nf // 2)
    yp = np.concatenate((pad * y[0], y, pad * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def weighted_metrics(arrays: dict, w: np.ndarray) -> dict:
    """AP (COCO definition) and P/R at the max-mean-F1 confidence, with per-image weights w."""
    rec_thrs = arrays["rec_thrs"]
    eps = np.spacing(1)
    ap_all, ap50, ap75 = {}, {}, {}
    grid = np.linspace(0, 1, 1000)
    p_curves, r_curves = {}, {}
    for k, c in arrays["classes"].items():
        if c is None:
            continue
        n_gt = float(np.dot(w, c["npig"]))
        if n_gt <= 0:
            continue
        wd = w[c["img"]]
        keep = wd > 0
        tp = c["tp"][:, keep] * wd[keep]
        fp = c["fp"][:, keep] * wd[keep]
        scores = c["scores"][keep]
        nd = tp.shape[1]
        if nd == 0:
            ap_all[k] = ap50[k] = ap75[k] = 0.0
            p_curves[k] = np.ones_like(grid)
            r_curves[k] = np.zeros_like(grid)
            continue
        tps, fps = np.cumsum(tp, axis=1), np.cumsum(fp, axis=1)
        rc = tps / n_gt
        pr = tps / (fps + tps + eps)
        env = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        q = np.zeros((tp.shape[0], len(rec_thrs)))
        for t in range(tp.shape[0]):
            inds = np.searchsorted(rc[t], rec_thrs, side="left")
            valid = inds < nd
            q[t, valid] = env[t, inds[valid]]
        ap_all[k], ap50[k], ap75[k] = float(q.mean()), float(q[0].mean()), float(q[5].mean())
        # operating point (IoU 0.50): precision / recall as a function of the confidence threshold
        p_curves[k] = np.interp(-grid, -scores, pr[0], left=1)
        r_curves[k] = np.interp(-grid, -scores, rc[0], left=0)
    ks = sorted(ap_all)
    if not ks:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "P": 0.0, "R": 0.0, "F1": 0.0, "conf": 0.0,
                "per_class_AP": {}, "per_class_P": {}, "per_class_R": {}}
    P = np.stack([p_curves[k] for k in ks])
    R = np.stack([r_curves[k] for k in ks])
    F1 = 2 * P * R / (P + R + 1e-16)
    i = int(_smooth(F1.mean(0), 0.1).argmax())
    return {"mAP50_95": float(np.mean([ap_all[k] for k in ks])),
            "mAP50": float(np.mean([ap50[k] for k in ks])),
            "mAP75": float(np.mean([ap75[k] for k in ks])),
            "P": float(P[:, i].mean()), "R": float(R[:, i].mean()), "F1": float(F1[:, i].mean()),
            "conf": float(grid[i]),
            "per_class_AP": ap_all, "per_class_P": {k: float(P[j, i]) for j, k in enumerate(ks)},
            "per_class_R": {k: float(R[j, i]) for j, k in enumerate(ks)}}


def paired_bootstrap(arr_a: dict, arr_b: dict, n_boot: int = 1000, seed: int = 0,
                     keys=("mAP50_95", "mAP50", "P", "R")) -> dict:
    """Resample test images with replacement; return the distribution summary of metric(a) - metric(b)."""
    n = arr_a["n_images"]
    assert n == arr_b["n_images"], "both models must be evaluated on the same images"
    rng = np.random.default_rng(seed)
    point_a, point_b = weighted_metrics(arr_a, np.ones(n)), weighted_metrics(arr_b, np.ones(n))
    deltas = {k: [] for k in keys}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        ma, mb = weighted_metrics(arr_a, w), weighted_metrics(arr_b, w)
        for k in keys:
            deltas[k].append(ma[k] - mb[k])
    out = {}
    for k in keys:
        d = np.asarray(deltas[k])
        out[k] = {"a": point_a[k], "b": point_b[k], "delta": point_a[k] - point_b[k],
                  "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                  "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
    return out


# ----------------------------------------------------------------------------
def selftest(seed: int = 0) -> None:
    """Weighted accumulate with unit weights must equal pycocotools; raises otherwise."""
    rng = np.random.default_rng(seed)
    H = W = 64
    imgs, anns, dets = [], [], []
    aid = 1
    for i in range(40):
        imgs.append({"id": i + 1, "file_name": f"{i}.png", "width": W, "height": H})
        for _ in range(rng.integers(0, 4)):
            c = int(rng.integers(0, 3))
            m = np.zeros((H, W), np.uint8)
            x, y, s = rng.integers(0, 40, 2).tolist() + [int(rng.integers(4, 24))]
            m[y:y + s, x:x + s] = 1
            ys, xs = np.nonzero(m)
            anns.append({"id": aid, "image_id": i + 1, "category_id": c + 1, "segmentation": rle_encode(m),
                         "area": int(m.sum()), "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1,
                                                        int(np.ptp(ys)) + 1], "iscrowd": 0})
            aid += 1
            for _ in range(rng.integers(0, 3)):
                dm = np.roll(m, tuple(rng.integers(-3, 4, 2)), axis=(0, 1))
                dets.append({"image_id": i + 1, "category_id": int(rng.integers(1, 4)) if rng.random() < .3 else c + 1,
                             "segmentation": rle_encode(dm), "score": float(rng.random())})
        if rng.random() < 0.3:
            m = np.zeros((H, W), np.uint8)
            m[5:15, 5:15] = 1
            dets.append({"image_id": i + 1, "category_id": 1, "segmentation": rle_encode(m), "score": float(rng.random())})
    gt = coco_from_dict({"images": imgs, "annotations": anns, "categories": [{"id": j, "name": str(j)} for j in (1, 2, 3)]})
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        E = run_cocoeval(gt, dets)
    ref, _ = summarise(E, ["0", "1", "2"])
    got = weighted_metrics(prepare_arrays(E), np.ones(len(imgs)))
    for k in ("mAP50_95", "mAP50", "mAP75"):
        if abs(ref[k] - got[k]) > 1e-12:
            raise AssertionError(f"selftest failed for {k}: pycocotools {ref[k]} vs weighted {got[k]}")
    # duplicating every image twice must not change AP (weights = 2)
    got2 = weighted_metrics(prepare_arrays(E), 2 * np.ones(len(imgs)))
    if abs(got2["mAP50_95"] - got["mAP50_95"]) > 1e-12:
        raise AssertionError("selftest failed: uniform weights changed AP")


if __name__ == "__main__":
    selftest()
    print("cocoutils selftest passed")
