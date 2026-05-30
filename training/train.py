#!/usr/bin/env python3
"""Train a single-class QR detector (Faster R-CNN + MobileNetV3-Large-FPN).

License-clean: torchvision is BSD-3, so weights you train here are yours to
license (MIT/Apache) -- no Ultralytics/AGPL anywhere in the stack. The FPN
backbone handles the small-QR-in-big-photo regime that classical decoders miss.

Outputs a checkpoint at <out>/qr_frcnn.pth. Run export_onnx.py next.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import QrDataset, collate_fn


def build_model(num_classes: int = 2):
    # COCO-pretrained backbone+RPN gives a big head start; we only swap the
    # box predictor for our (background + QR) = 2 classes.
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        min_size=640, max_size=1024,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_hitrate(model, loader, device, iou_thr=0.5, score_thr=0.5):
    """Cheap proxy metric: fraction of GT boxes matched by a >score_thr pred."""
    model.eval()
    matched = total = 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            gts = tgt["boxes"]
            total += len(gts)
            keep = out["scores"] >= score_thr
            preds = out["boxes"][keep].cpu()
            for g in gts:
                if len(preds) and _max_iou(g, preds) >= iou_thr:
                    matched += 1
    return matched / max(1, total)


def _max_iou(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_p = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (inter / (area_b + area_p - inter + 1e-6)).max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", default=12, type=int)
    ap.add_argument("--batch-size", default=8, type=int)
    ap.add_argument("--lr", default=0.02, type=float,
                    help="reference LR for total batch 16; scaled by batch size")
    ap.add_argument("--clip", default=10.0, type=float, help="grad-norm clip")
    ap.add_argument("--workers", default=2, type=int)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    print(f"device: {device}")

    train_ds = QrDataset(args.data, "train")
    val_ds = QrDataset(args.data, "val")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # scale LR to batch size (reference recipe is tuned for total batch 16)
    lr = args.lr * args.batch_size / 16
    print(f"effective lr={lr:.5f} (base {args.lr} @ batch {args.batch_size})")
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        # linear LR warmup over the first epoch stabilizes the pretrained head
        warmup = None
        if epoch == 0:
            warmup_iters = min(1000, len(train_loader) - 1)
            if warmup_iters > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, total_iters=warmup_iters)
        for i, (images, targets) in enumerate(train_loader):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not math.isfinite(loss.item()):
                print("non-finite loss, skipping batch")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
            if warmup is not None:
                warmup.step()
            running += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch} [{i+1}/{len(train_loader)}] "
                      f"loss={running/(i+1):.3f}")
        scheduler.step()
        hit = evaluate_hitrate(model, val_loader, device)
        print(f"epoch {epoch}: loss={running/len(train_loader):.3f} "
              f"val_hitrate={hit:.3f} ({time.time()-t0:.0f}s)")
        torch.save(model.state_dict(), out / "qr_frcnn.pth")
        if hit >= best:
            best = hit
            torch.save(model.state_dict(), out / "qr_frcnn_best.pth")
    print(f"done. best val hitrate={best:.3f}. checkpoint: {out/'qr_frcnn_best.pth'}")


if __name__ == "__main__":
    main()
