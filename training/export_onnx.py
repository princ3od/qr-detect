#!/usr/bin/env python3
"""Export the trained Faster R-CNN checkpoint to ONNX.

The exported graph bakes in torchvision's preprocessing (resize + ImageNet
normalize) AND postprocessing (NMS), so:
  * INPUT  "images": float32 RGB in [0,1], shape [3, H, W] (dynamic H/W, NOT batched)
  * OUTPUT "boxes":  [N,4] xyxy in input-pixel coords
           "labels": [N]  (all == 1 for QR)
           "scores": [N]  descending
This keeps the Node side trivial: feed RGB/255, read boxes+scores. No manual
anchor decode or NMS needed in TypeScript.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/qr_frcnn_best.pth")
    ap.add_argument("--out", default="qr-detector.onnx")
    ap.add_argument("--opset", default=11, type=int)
    args = ap.parse_args()

    model = build_model()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    example = [torch.rand(3, 640, 640)]
    # NOTE: wrap in a tuple so export passes `example` as a SINGLE positional arg
    # (the list of images) rather than unpacking it via model(*example).
    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "images": {1: "height", 2: "width"},
            "boxes": {0: "num"},
            "labels": {0: "num"},
            "scores": {0: "num"},
        },
        do_constant_folding=True,
    )
    # Force the legacy TorchScript exporter. Newer torch (2.9+) defaults to the
    # dynamo exporter, which needs `onnxscript` and does NOT reliably export
    # torchvision detection models (dynamic control flow + NMS). dynamo=False is
    # the validated path. Fall back for older torch that lacks the kwarg.
    try:
        torch.onnx.export(model, (example,), args.out, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, (example,), args.out, **export_kwargs)
    print(f"exported {args.out}")

    # sanity check with onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        dummy = np.random.rand(3, 480, 640).astype(np.float32)
        outs = sess.run(None, {"images": dummy})
        print("onnxruntime OK. outputs:")
        for name, arr in zip([o.name for o in sess.get_outputs()], outs):
            print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: onnxruntime sanity check failed: {e}")

    print(f"\nDone. Drop {Path(args.out).name} into the package at "
          f"models/qr-detector.onnx and run `npm test`.")


if __name__ == "__main__":
    main()
