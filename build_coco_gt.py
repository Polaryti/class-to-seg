#!/usr/bin/env python3
"""Rasterise the YOLO polygon labels of one split into a COCO ground-truth file (canonical scale).

Example
-------
python build_coco_gt.py --data /scratch/protocol/v3.yaml --split test --out /scratch/results/gt_v3_test.json.gz
"""
import argparse
from collections import Counter
from pathlib import Path

from cocoutils import build_gt, save_json_gz
from common import class_names, load_yaml, split_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data YAML (with --split)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--images-list", help="text file with one image path per line (instead of --data/--split)")
    ap.add_argument("--names", help="data YAML providing the class names (with --images-list)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.images_list:
        if not a.names:
            raise SystemExit("--images-list needs --names (a data YAML with the class names)")
        names = class_names(load_yaml(a.names))
        images = [Path(x) for x in Path(a.images_list).read_text().split() if x]
    else:
        if not a.data:
            raise SystemExit("give --data or --images-list")
        names = class_names(load_yaml(a.data))
        images = split_images(a.data)[a.split]
    gt, stats = build_gt(images, names)
    per_img = Counter(x["image_id"] for x in gt["annotations"])
    stats["images"] = len(images)
    stats["annotations_kept"] = len(gt["annotations"])
    stats["max_instances_per_image"] = max(per_img.values()) if per_img else 0
    stats["images_over_100_instances"] = sum(1 for v in per_img.values() if v > 100)
    save_json_gz(gt, a.out)
    print(stats)
    if stats["images_over_100_instances"]:
        print("WARNING: some images have more than 100 instances; COCO evaluation caps detections at 100 per image.")
    if stats["box_lines"]:
        print(f"NOTE: {stats['box_lines']} label lines are boxes (5 values); they are evaluated as rectangles.")


if __name__ == "__main__":
    main()
