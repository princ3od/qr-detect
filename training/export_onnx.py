#!/usr/bin/env python3
"""Export the trained Faster R-CNN checkpoint to ONNX (onnxruntime-safe).

torchvision's stock `RoIHeads.postprocess_detections` exports a broken graph:
`box_coder.decode` traces to `[N, num_classes*4]` (2D) instead of `[N, num_classes, 4]`,
and `clip_boxes_to_image` emits a dynamic `reshape(boxes.shape)`. The net effect is
a `Reshape {N,7} -> {-1,4}` that crashes at inference in onnxruntime
("input tensor cannot be reshaped"). We bind an export-safe postprocess that
(a) forces `reshape(-1, NUM_CLASSES, 4)` with NUM_CLASSES as a Python constant and
(b) clips with a shape-preserving clamp. Validated in onnxruntime across input sizes.

Exported graph:
  * INPUT  "images": float32 RGB in [0,1], shape [3, H, W] (dynamic H/W, NOT batched)
  * OUTPUT "boxes" [N,4] xyxy (input-pixel coords), "scores" [N], "labels" [N] (==1)
"""
from __future__ import annotations

import argparse
import types
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from train import build_model, NUM_CLASSES


def _clip_safe(boxes, size):
    """Shape-preserving clip (avoids torchvision's dynamic reshape(boxes.shape))."""
    h, w = size
    x1 = boxes[..., 0].clamp(min=0, max=w)
    y1 = boxes[..., 1].clamp(min=0, max=h)
    x2 = boxes[..., 2].clamp(min=0, max=w)
    y2 = boxes[..., 3].clamp(min=0, max=h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _export_safe_postprocess(self, class_logits, box_regression, proposals, image_shapes):
    device = class_logits.device
    boxes_per_image = [p.shape[0] for p in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)
    # FORCE [sum, NUM_CLASSES, 4]; NUM_CLASSES is a python constant -> export-safe.
    pred_boxes = pred_boxes.reshape(-1, NUM_CLASSES, 4)
    pred_scores = F.softmax(class_logits, -1)
    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = _clip_safe(boxes, image_shape)
        labels = torch.arange(NUM_CLASSES, device=device).view(1, -1).expand_as(scores)
        boxes = boxes[:, 1:].reshape(-1, 4)
        scores = scores[:, 1:].reshape(-1)
        labels = labels[:, 1:].reshape(-1)
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
    return all_boxes, all_scores, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/qr_frcnn_best.pth")
    ap.add_argument("--out", default="qr-detector.onnx")
    ap.add_argument("--opset", default=11, type=int)
    args = ap.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    # swap in the onnxruntime-safe postprocess
    model.roi_heads.postprocess_detections = types.MethodType(
        _export_safe_postprocess, model.roi_heads)
    model.eval()

    example = [torch.rand(3, 640, 640)]  # tuple-wrap so it's ONE positional arg
    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={"images": {1: "height", 2: "width"}, "boxes": {0: "num"},
                      "labels": {0: "num"}, "scores": {0: "num"}},
        do_constant_folding=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (example,), args.out, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, (example,), args.out, **export_kwargs)
    print(f"exported {args.out}")

    # sanity check + verify it actually RUNS in onnxruntime (the part that used to break)
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        for hw in [(480, 640), (800, 600), (1024, 768)]:
            dummy = np.random.rand(3, *hw).astype(np.float32)
            outs = sess.run(None, {"images": dummy})
            print(f"  onnxruntime OK @ {hw}: boxes {outs[0].shape}, scores {outs[2].shape}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: onnxruntime sanity check failed: {e}")

    print(f"\nDone. Drop {Path(args.out).name} into the package at "
          f"models/qr-detector.onnx and run `npm test`.")


if __name__ == "__main__":
    main()
