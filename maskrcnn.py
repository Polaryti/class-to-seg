#!/usr/bin/env python3
"""Mask R-CNN (ResNet-50-FPN v2, torchvision, COCO-pretrained) baseline on a YOLO-format corpus.

Standard two-stage recipe, not the YOLO augmentation pipeline: SGD, momentum 0.9,
weight decay 1e-4, base LR 0.02 x batch/16, linear warm-up over 1000 iterations,
LR x0.1 at 2/3 and 11/12 of the schedule, horizontal flip, mixed precision.
Images are resized so that their long side is 640 px (same scale as the YOLO runs
and as the evaluation). The checkpoint with the best validation mask mAP50-95 is
kept (evaluated every --eval-every epochs), mirroring the YOLO model selection.

Re-running the same command resumes from <out>/last.pth; a finished run exits.

Example
-------
python maskrcnn.py --data /scratch/protocol/v3.yaml --out /scratch/runs/E18
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (class_names, environment_info, label_path, load_canonical, load_yaml, read_label, split_images,
                    torch_load, write_json)


# ----------------------------------------------------------------------------
class YoloSegDataset(torch.utils.data.Dataset):
    def __init__(self, images, train: bool):
        self.images = list(images)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p = self.images[idx]
        im = load_canonical(p)[:, :, ::-1]  # BGR -> RGB
        h, w = im.shape[:2]
        masks, labels = [], []
        for c, pts, _ in read_label(label_path(p)):
            m = np.zeros((h, w), np.uint8)
            poly = np.round(pts * np.array([w, h])).astype(np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(m, [poly], 1)
            if m.any():
                masks.append(m)
                labels.append(c + 1)
        if self.train and np.random.rand() < 0.5:
            im = im[:, ::-1]
            masks = [m[:, ::-1] for m in masks]
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.as_tensor(np.asarray(boxes, np.float32).reshape(-1, 4)),
            "labels": torch.as_tensor(np.asarray(labels, np.int64)),
            "masks": torch.as_tensor(np.ascontiguousarray(np.stack(masks)) if masks else np.zeros((0, h, w), np.uint8)),
            "image_id": idx,
        }
        img = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
        return img, target


def make_grad_scaler(enabled: bool):
    """torch.amp.GradScaler on recent PyTorch, torch.cuda.amp.GradScaler on older releases."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, pretrained: bool = True):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    kw = dict(min_size=640, max_size=640, box_detections_per_img=100)
    if pretrained:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **kw)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kw)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes + 1)
    return model


def load_for_inference(weights: str | Path, device):
    ck = torch_load(weights)
    model = build_model(ck["num_classes"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 100
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_image(model, bgr: np.ndarray, device):
    img = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1).float().div(255).to(device)
    out = model([img])[0]
    masks = (out["masks"][:, 0] > 0.5).cpu().numpy()
    return masks, (out["labels"].cpu().numpy() - 1), out["scores"].cpu().numpy()


@torch.no_grad()
def evaluate_split(model, images, names, device, gt_cache: Path) -> float:
    from cocoutils import build_gt, coco_from_dict, load_json_gz, masks_to_detections, run_cocoeval, save_json_gz, summarise
    if gt_cache.exists():
        gt = load_json_gz(gt_cache)
    else:
        gt, _ = build_gt(images, names)
        save_json_gz(gt, gt_cache)
    coco_gt = coco_from_dict(gt)
    model.eval()
    old = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001
    dets = []
    for info in gt["images"]:
        im = load_canonical(info["file_name"])
        masks, cls, scores = predict_image(model, im, device)
        dets += masks_to_detections(info["id"], masks, cls, scores, (info["height"], info["width"]))
    model.roi_heads.score_thresh = old
    with contextlib.redirect_stdout(io.StringIO()):
        E = run_cocoeval(coco_gt, dets)
    return summarise(E, names)[0]["mAP50_95"]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data YAML (train/val splits are used)")
    ap.add_argument("--out", required=True, help="run directory (last.pth, best.pth, log)")
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default 0.02 * batch / 16")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)  # also used as the run's seed
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke test only)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-images", type=int, default=None, help="cap train/val images (smoke test only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = class_names(load_yaml(a.data))
    splits = split_images(a.data)
    train_imgs, val_imgs = splits["train"], splits["val"]
    if a.max_images:
        train_imgs, val_imgs = train_imgs[:a.max_images], val_imgs[:a.max_images]
    device = torch.device(a.device)
    use_amp = device.type == "cuda" and not a.no_amp
    lr0 = a.lr if a.lr is not None else 0.02 * a.batch / 16
    milestones = [int(round(a.epochs * 2 / 3)), int(round(a.epochs * 11 / 12))]

    last, best = out / "last.pth", out / "best.pth"
    state = torch_load(last) if last.exists() else None
    if state is not None and state.get("finished"):
        print(f"Already finished: {last}")
        return

    model = build_model(len(names), pretrained=not a.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr0, momentum=0.9, weight_decay=1e-4)
    scaler = make_grad_scaler(use_amp)
    start_epoch, best_map, history = 0, -1.0, []
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_map, history = state["epoch"] + 1, state["best_map"], state["history"]
        print(f"Resuming from epoch {start_epoch + 1}/{a.epochs} (best val mAP50-95 so far {best_map:.4f})")

    ds = YoloSegDataset(train_imgs, train=True)
    g = torch.Generator()
    g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     collate_fn=collate, drop_last=True, generator=g,
                                     persistent_workers=a.workers > 0)
    iters_per_epoch = len(dl)
    warmup = min(1000, iters_per_epoch * a.epochs // 4) or 1
    info = {"data": a.data, "epochs": a.epochs, "batch": a.batch, "lr0": lr0, "milestones": milestones,
            "warmup_iters": warmup, "train_images": len(train_imgs), "val_images": len(val_imgs),
            "amp": use_amp, "environment": environment_info()}
    write_json(out / "run_info.json", info)

    for epoch in range(start_epoch, a.epochs):
        # the loader order depends on the epoch only, so a resumed run sees the same batches
        g.manual_seed(a.seed + epoch)
        model.train()
        t0, run_loss = time.time(), 0.0
        decay = 0.1 ** sum(epoch >= m for m in milestones)
        for it, (imgs, targets) in enumerate(dl):
            step = epoch * iters_per_epoch + it
            wf = min(1.0, 0.001 + (1 - 0.001) * step / warmup)
            for pg in opt.param_groups:
                pg["lr"] = lr0 * decay * wf
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, iteration {it}: {losses}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            if it % 200 == 0:
                print(f"epoch {epoch + 1}/{a.epochs} iter {it}/{iters_per_epoch} loss {loss.item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.5f}", flush=True)
        rec = {"epoch": epoch + 1, "loss": run_loss / max(1, iters_per_epoch), "minutes": (time.time() - t0) / 60}
        if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
            m = evaluate_split(model, val_imgs, names, device, out / "gt_val.json.gz")
            rec["val_mAP50_95"] = m
            if m > best_map:
                best_map = m
                torch.save({"model": model.state_dict(), "num_classes": len(names), "names": names,
                            "epoch": epoch + 1, "epochs_configured": a.epochs, "val_mAP50_95": m}, str(best))
        history.append(rec)
        print(rec, flush=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_map": best_map, "history": history, "epochs_configured": a.epochs,
                    "finished": epoch + 1 == a.epochs}, str(last))
        write_json(out / "history.json", history)
    print(f"Finished. Best validation mask mAP50-95 {best_map:.4f} -> {best}")


if __name__ == "__main__":
    main()
