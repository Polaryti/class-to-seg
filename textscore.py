#!/usr/bin/env python3
"""How much text-like structure an image carries, used to keep watermarked crops out of the figures.

Stock-photography watermarks ("shutterstock.com", a diagonal URL, a logo band) are text drawn
over the photograph. Text has a signature that ordinary object boundaries do not: many small
regions of nearly uniform intensity, similar in height, with a stroke width that hardly varies
inside each region. Maximally Stable Extremal Regions find exactly those, and filtering them by
size, aspect and stroke-width uniformity leaves mostly glyphs.

The score is the number of surviving glyph-like regions per megapixel. It is a ranking signal,
not a detector: use it to choose between candidate crops, and look at the result.

    python textscore.py image1.jpg image2.jpg ...     # prints a score per file, lowest first
"""
from __future__ import annotations

import sys

import cv2
import numpy as np


def _stroke_uniformity(mask: np.ndarray) -> float:
    """1 means a perfectly constant stroke width inside the region, 0 means none at all."""
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    v = d[d > 0]
    if v.size < 8:
        return 0.0
    return float(1.0 / (1.0 + v.std() / (v.mean() + 1e-6)))


def text_score(img: np.ndarray, debug: bool = False):
    """Glyph-like regions per megapixel. Pass a BGR image."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    area = h * w
    mser = cv2.MSER_create()
    mser.setMinArea(max(12, int(2e-5 * area)))
    mser.setMaxArea(int(0.02 * area))
    mser.setDelta(4)
    boxes = []
    for pol in mser.detectRegions(g)[0]:
        x, y, bw, bh = cv2.boundingRect(pol.reshape(-1, 1, 2))
        if bh < 5 or bw < 2:
            continue
        ar = bw / bh
        if not 0.08 < ar < 4.0:                       # glyphs are tallish, not long bars
            continue
        if bh > 0.30 * h:                             # too tall to be a letter
            continue
        fill = len(pol) / float(bw * bh)
        if not 0.10 < fill < 0.92:                    # letters are neither blobs nor outlines
            continue
        m = np.zeros((bh + 2, bw + 2), np.uint8)
        q = pol - [x, y] + 1
        m[q[:, 1], q[:, 0]] = 255
        if _stroke_uniformity(m) < 0.55:
            continue
        boxes.append((x, y, bw, bh))
    # glyphs come in rows: keep only regions that have a neighbour of similar height nearby
    kept = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        for j, (x2, y2, bw2, bh2) in enumerate(boxes):
            if i == j:
                continue
            if abs(bh - bh2) <= 0.35 * max(bh, bh2) and abs(y - y2) <= 0.8 * bh \
               and 0 < abs(x - x2) <= 3.5 * max(bw, bw2):
                kept.append((x, y, bw, bh))
                break
    score = len(kept) / (area / 1e6)
    return (score, kept) if debug else score


AGENCIES = ("shutterstock", "istock", "istockphoto", "gettyimages", "getty images", "bigstock",
            "dreamstime", "freepik", "depositphotos", "123rf", "alamy", "adobestock", "adobe stock",
            "vectorstock", "stockphoto", "stock photo", "canstock", "fotolia", "pond5", "zoonar",
            "megapixl", "crushpixel", "picfair", "stockfresh", "watermark")


def _ocr_words(img):
    """(text, confidence, box) for every word tesseract is reasonably sure of, or [] without it."""
    try:
        import pytesseract
    except Exception:
        return []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out = []
    for im in (up, cv2.createCLAHE(3.0, (8, 8)).apply(up)):
        try:
            d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 11")
        except Exception:
            return []
        for txt, conf, x, y, w_, h_ in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
            txt = txt.strip()
            if len(txt) >= 4 and float(conf) >= 55 and any(c.isalpha() for c in txt):
                out.append((txt.lower(), float(conf), (x / 2, y / 2, w_ / 2, h_ / 2)))
    return out


def watermark_evidence(img) -> dict:
    """What suggests this crop carries an overlay that must not be published.

    Two cheap, high-precision rules on top of the structural score: the name of a stock-photo
    agency anywhere in the image, and readable words in the outer border, where a photograph's
    subject almost never puts text but an agency's identifier band always does. Neither is
    complete -- faint tiled watermarks are missed -- so this reorders candidates, it does not
    certify a crop. Look at the contact sheet.
    """
    h, w = img.shape[:2]
    words = _ocr_words(img)
    agency = sorted({a for txt, _, _ in words for a in AGENCIES if a in txt or txt in a})
    border = [txt for txt, _, (x, y, bw, bh) in words
              if y + bh < 0.14 * h or y > 0.86 * h or x + bw < 0.10 * w or x > 0.90 * w]
    return {"agency": agency, "border_words": sorted(set(border)), "words": len(words)}


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    rows = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        ev = watermark_evidence(im)
        rows.append((text_score(im), ev, f))
    for s, ev, f in sorted(rows, key=lambda r: (-len(r[1]["agency"]), -len(r[1]["border_words"]), -r[0])):
        flag = ("AGENCY:" + ",".join(ev["agency"])) if ev["agency"] else \
               ("border:" + ",".join(ev["border_words"][:3]) if ev["border_words"] else "")
        print(f"{s:9.1f}  {flag:34s} {f}")


if __name__ == "__main__":
    main()
