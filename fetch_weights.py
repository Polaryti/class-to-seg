#!/usr/bin/env python3
"""Download every pretrained file the jobs need, for clusters whose compute nodes are offline.

Run once on a login node (with internet), with the same environment variables as
the jobs (config.sh sets TORCH_HOME and HF_HOME):

    source config.sh && activate_env && python fetch_weights.py --weights-dir "$WEIGHTS_DIR"
"""
import argparse
import os
from pathlib import Path

from common import read_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--birefnet", default=os.environ.get("BIREFNET", "ZhengPeng7/BiRefNet"))
    ap.add_argument("--skip-birefnet", action="store_true")
    ap.add_argument("--skip-maskrcnn", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="additional Ultralytics weights, e.g. yolo11n-cls.pt")
    a = ap.parse_args()
    wd = Path(a.weights_dir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    os.chdir(wd)
    from ultralytics import YOLO
    models = sorted({e["model"] for e in read_manifest(Path(__file__).resolve().parent / a.manifest).values()
                     if e["kind"] == "yolo" and e["model"] != "-"} | {"yolo11n.pt"} | set(a.extra))
    for m in models:
        print(f"ultralytics: {m}")
        YOLO(m)
    if not a.skip_maskrcnn:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        print("torchvision: Mask R-CNN R50-FPN v2 COCO weights ->", os.environ.get("TORCH_HOME", "~/.cache/torch"))
        MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict(progress=True)
    if not a.skip_birefnet:
        from transformers import AutoModelForImageSegmentation
        print(f"hugging face: {a.birefnet} ->", os.environ.get("HF_HOME", "~/.cache/huggingface"))
        AutoModelForImageSegmentation.from_pretrained(a.birefnet, trust_remote_code=True)
    print("done:", ", ".join(p.name for p in wd.glob("*.pt")))


if __name__ == "__main__":
    main()
