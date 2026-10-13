"""Shared helpers for the waste-segmentation experiments.

Only the standard library, numpy, PyYAML, Pillow and OpenCV are used here, so
this module can be imported by every script (including CPU-only jobs).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
SPLITS = ("train", "val", "test")
CANONICAL_LONG_SIDE = 640          # every evaluation / statistic is done at this scale
SMALL_AREA = 32 ** 2               # COCO thresholds, in pixels at the canonical scale
MEDIUM_AREA = 96 ** 2


# ----------------------------------------------------------------------------
# Data YAML handling (same resolution rules as Ultralytics)
# ----------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def class_names(data: dict) -> list[str]:
    """Return the class names of a data YAML as an ordered list."""
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def _resolve_entry(root: Path, entry: str) -> Path:
    p = Path(entry)
    if p.is_absolute():
        return p
    x = (root / entry).resolve()
    if not x.exists() and entry.startswith("../"):
        x = (root / entry[3:]).resolve()
    return x


def _ultralytics_datasets_dir():
    try:
        from ultralytics.utils import DATASETS_DIR
        return Path(DATASETS_DIR)
    except Exception:
        return None


def split_sources(yaml_path: str | Path) -> dict[str, list[Path]]:
    """Map 'train'/'val'/'test' to the list of directories or .txt files they point to.

    Each entry is resolved against the first of several roots under which it exists: the
    'path:' of the YAML, the YAML's own folder and its parent, and the Ultralytics
    datasets_dir. This covers absolute paths, paths relative to the datasets folder, and
    image-list .txt files stored next to the YAML or with the dataset.
    """
    yaml_path = Path(yaml_path).resolve()
    data = load_yaml(yaml_path)
    base = data.get("path")
    roots = []
    if base:
        b = Path(base)
        if b.is_absolute():
            roots.append(b)
        else:
            roots += [yaml_path.parent / b, yaml_path.parent.parent / b]
            ds = _ultralytics_datasets_dir()
            if ds is not None:
                roots.append(ds / b)
    roots += [yaml_path.parent, yaml_path.parent.parent]
    ds = _ultralytics_datasets_dir()
    if ds is not None:
        roots.append(ds)
    # escape hatch when the images or the .txt lists live somewhere else entirely
    roots += [Path(r) for r in os.environ.get("WASTE_EXTRA_DATA_ROOTS", "").split(os.pathsep) if r]
    out = {}
    for s in SPLITS:
        v = data.get(s)
        if not v:
            continue
        entries = v if isinstance(v, (list, tuple)) else [v]
        resolved = []
        for e in entries:
            cands = [_resolve_entry(r.resolve(), str(e)) for r in roots]
            resolved.append(next((c for c in cands if c.exists()), cands[0]))
        out[s] = resolved
    return out


def list_images(source: Path) -> list[Path]:
    """List image files under a directory (recursively) or inside a .txt list."""
    source = Path(source)
    if source.is_file() and source.suffix == ".txt":
        parent = source.parent
        files = []
        for line in source.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (parent / line).resolve()
            files.append(p)
        return sorted(f for f in files if f.suffix.lower() in IMG_EXTS)
    if not source.exists():
        raise FileNotFoundError(
            f"Split source does not exist: {source}. If the images or the .txt lists are stored elsewhere, "
            f"set WASTE_EXTRA_DATA_ROOTS=/first/root{os.pathsep}/second/root (the folders those paths are relative to).")
    files = [Path(r) / f for r, _, fs in os.walk(source, followlinks=True) for f in fs]
    return sorted(f for f in files if f.suffix.lower() in IMG_EXTS)


def split_images(yaml_path: str | Path) -> dict[str, list[Path]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} image paths of a data YAML."""
    return {s: [p for src in srcs for p in list_images(src)] for s, srcs in split_sources(yaml_path).items()}


def label_path(img_path: str | Path) -> Path:
    """Ultralytics convention: .../images/xxx.jpg -> .../labels/xxx.txt (last 'images' component)."""
    s = str(img_path)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in s:
        raise ValueError(f"Image path has no '{sa}' component, cannot locate its label: {s}")
    return Path(sb.join(s.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


# ----------------------------------------------------------------------------
# Label I/O
# ----------------------------------------------------------------------------
def read_label(path: str | Path) -> list[tuple[int, np.ndarray, bool]]:
    """Read a YOLO label file.

    Returns a list of (class_id, points[N, 2] normalised, is_box). Lines with five
    values are boxes (cx, cy, w, h) and are returned as 4-point rectangles with
    is_box=True. Missing file -> empty list (background image).
    """
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        c = int(float(vals[0]))
        nums = np.asarray([float(v) for v in vals[1:]], dtype=np.float64)
        if len(nums) == 4:
            cx, cy, w, h = nums
            pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
            out.append((c, pts, True))
        else:
            if len(nums) % 2:
                nums = nums[:-1]
            out.append((c, nums.reshape(-1, 2), False))
    return out


def write_label(path: str | Path, instances: list[tuple[int, np.ndarray]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for c, pts in instances:
        coords = " ".join(f"{v:.6f}" for v in np.clip(np.asarray(pts).reshape(-1), 0.0, 1.0))
        lines.append(f"{int(c)} {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def polygon_area(pts: np.ndarray) -> float:
    """Shoelace area of a polygon given as [N, 2] (any units)."""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------
def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) as OpenCV would load the image (EXIF orientation applied)."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return int(w), int(h)


def canonical_size(w: int, h: int, long_side: int = CANONICAL_LONG_SIDE) -> tuple[int, int]:
    s = long_side / max(w, h)
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def load_canonical(path: str | Path, long_side: int = CANONICAL_LONG_SIDE) -> np.ndarray:
    """Load an image as BGR uint8 (like Ultralytics) and resize so that its long side is `long_side`."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"Cannot read image {path}")
    h, w = im.shape[:2]
    nw, nh = canonical_size(w, h, long_side)
    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR
        im = cv2.resize(im, (nw, nh), interpolation=interp)
    return im


def md5sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dhash(path: str | Path, size: int = 8) -> str:
    """Difference hash (64 bit) of an image; identical for re-encoded / resized copies."""
    from PIL import Image
    with Image.open(path) as im:
        try:
            im.draft("L", (size * 8, size * 8))
        except Exception:
            pass
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


# ----------------------------------------------------------------------------
# CSV / JSON / environment
# ----------------------------------------------------------------------------
def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def json_default(o):
    """Make NumPy values (and anything else exotic) serialisable.

    pycocotools writes NumPy scalars and arrays into the dictionaries it is given
    (loadRes adds 'area' and 'bbox'), so anything that has passed through it needs this.
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    from importlib import metadata
    dists = {"numpy": "numpy", "torch": "torch", "torchvision": "torchvision", "ultralytics": "ultralytics",
             "cv2": "opencv-python", "pycocotools": "pycocotools", "transformers": "transformers"}
    for mod, dist in dists.items():
        try:
            m = __import__(mod)
        except Exception:
            continue
        v = getattr(m, "__version__", None)
        if v is None:
            for d in (dist, dist + "-headless", mod):
                try:
                    v = metadata.version(d)
                    break
                except Exception:
                    pass
        info[mod] = v or "unknown"
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    return info


def torch_load(path):
    """torch.load that works on PyTorch versions with and without the weights_only argument."""
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


# ----------------------------------------------------------------------------
# Experiment manifest
# ----------------------------------------------------------------------------
def read_manifest(path: str | Path) -> dict[str, dict]:
    """Read experiments.tsv (whitespace separated, '#' comments).

    Columns: id kind corpus model weights args
      kind    : yolo | maskrcnn
      corpus  : name of a YAML in the protocol directory (v1, v2, v3, v3_sizematched, ...)
      model   : initial weights for training (e.g. yolo11m-seg.pt) or '-' 
      weights : 'auto' (trained by these scripts under runs/<id>) or a path to an existing checkpoint
      args    : extra training arguments as key=value,key=value or '-'
    """
    exps = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Bad manifest line (need at least 5 columns): {raw}")
        eid, kind, corpus, model, weights = parts[:5]
        args = parts[5] if len(parts) > 5 else "-"
        extra = {}
        if args != "-":
            for kv in args.split(","):
                k, v = kv.split("=", 1)
                extra[k.strip()] = yaml.safe_load(v.strip())
        exps[eid] = {"id": eid, "kind": kind, "corpus": corpus, "model": model,
                     "weights": weights, "args": extra}
    return exps


def experiment_weights(exp: dict, runs_dir: str | Path) -> Path:
    """Checkpoint used for evaluation. An existing run may be given as its folder or as the file itself."""
    if exp["weights"] != "auto":
        p = Path(exp["weights"])
        if p.is_dir():
            for cand in (p / "weights" / "best.pt", p / "best.pt", p / "best.pth"):
                if cand.exists():
                    return cand
            return p / "weights" / "best.pt"
        return p
    if exp["kind"] == "maskrcnn":
        return Path(runs_dir) / exp["id"] / "best.pth"
    return Path(runs_dir) / exp["id"] / "weights" / "best.pt"
