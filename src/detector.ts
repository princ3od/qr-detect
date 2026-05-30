import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import sharp from "sharp";
import * as ort from "onnxruntime-node";

export type BBox = [x1: number, y1: number, x2: number, y2: number];

export interface Detection {
  bbox: BBox;
  confidence: number;
}

/**
 * A QR localizer. The decode pipeline depends only on this interface, so the
 * underlying model can be swapped without changing callers.
 *
 * Two implementations ship:
 *  - {@link FasterRcnnQrDetector} — torchvision (BSD-3), permissive, trainable
 *    via the `training/` pipeline. Preferred for commercial use.
 *  - {@link Yolov8QrDetector} — qrdet-s (Ultralytics YOLOv8, **AGPL-3.0**).
 */
export interface QrDetector {
  detect(image: Buffer): Promise<Detection[]>;
}

const here = dirname(fileURLToPath(import.meta.url));
const FASTERRCNN_MODEL = resolve(here, "../models/qr-detector.onnx");
const YOLOV8_MODEL = resolve(here, "../models/qrdet-s.onnx");

const SCORE_THRESHOLD = 0.3;

// ----------------------------------------------------------------------------
// FasterRCNN (torchvision) detector — permissive, the preferred model.
// ONNX bakes in resize+normalize and NMS:
//   input  "images": float32 RGB [0,1], dims [3,H,W]
//   output "boxes" [N,4] xyxy, "scores" [N] (and "labels" [N], ignored)
// ----------------------------------------------------------------------------

const FRCNN_MAX_SIDE = 1024; // cap input so inference stays fast on big photos

export class FasterRcnnQrDetector implements QrDetector {
  private session: ort.InferenceSession | null = null;
  constructor(private readonly modelPath: string = FASTERRCNN_MODEL) {}

  private async getSession(): Promise<ort.InferenceSession> {
    if (!this.session) {
      this.session = await ort.InferenceSession.create(this.modelPath);
    }
    return this.session;
  }

  async detect(image: Buffer): Promise<Detection[]> {
    const session = await this.getSession();
    const meta = await sharp(image).metadata();
    const origW = meta.width ?? 0;
    const origH = meta.height ?? 0;
    if (!origW || !origH) throw new Error("qr-detect: could not read image dimensions");

    const { data, info } = await sharp(image)
      .resize({
        width: FRCNN_MAX_SIDE,
        height: FRCNN_MAX_SIDE,
        fit: "inside",
        withoutEnlargement: true,
      })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const w = info.width;
    const h = info.height;
    const pixels = w * h;
    const chw = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      chw[i] = data[i * 3] / 255;
      chw[pixels + i] = data[i * 3 + 1] / 255;
      chw[2 * pixels + i] = data[i * 3 + 2] / 255;
    }
    const tensor = new ort.Tensor("float32", chw, [3, h, w]);

    const results = await session.run({ [session.inputNames[0]]: tensor });
    const boxes = (results["boxes"] ?? findByShape(results, 2)).data as Float32Array;
    const scores = (results["scores"] ?? findByShape(results, 1)).data as Float32Array;

    // map boxes from resized space back to original-image coordinates
    const sx = origW / w;
    const sy = origH / h;
    const dets: Detection[] = [];
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] < SCORE_THRESHOLD) continue;
      let x1 = boxes[i * 4] * sx;
      let y1 = boxes[i * 4 + 1] * sy;
      let x2 = boxes[i * 4 + 2] * sx;
      let y2 = boxes[i * 4 + 3] * sy;
      x1 = Math.max(0, Math.min(origW, x1));
      y1 = Math.max(0, Math.min(origH, y1));
      x2 = Math.max(0, Math.min(origW, x2));
      y2 = Math.max(0, Math.min(origH, y2));
      if (x2 - x1 < 1 || y2 - y1 < 1) continue;
      dets.push({ bbox: [x1, y1, x2, y2], confidence: scores[i] });
    }
    // torchvision already applied NMS; just order by confidence
    return dets.sort((a, b) => b.confidence - a.confidence);
  }
}

function findByShape(
  results: ort.InferenceSession.OnnxValueMapType,
  lastDim: 1 | 2
): ort.Tensor {
  for (const v of Object.values(results)) {
    const t = v as ort.Tensor;
    if (lastDim === 2 && t.dims.length === 2 && t.dims[1] === 4) return t;
    if (lastDim === 1 && t.dims.length === 1) return t;
  }
  throw new Error("qr-detect: expected FasterRCNN output not found");
}

// ----------------------------------------------------------------------------
// YOLOv8 (qrdet-s) detector — AGPL-3.0. Kept for parity / fallback.
// Segmentation model; we only read the detection head [1,37,8400].
// ----------------------------------------------------------------------------

const YOLO_INPUT = 640;
const PAD_VALUE = 114;
const NMS_IOU = 0.45;

interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
  origW: number;
  origH: number;
}

export class Yolov8QrDetector implements QrDetector {
  private session: ort.InferenceSession | null = null;
  constructor(private readonly modelPath: string = YOLOV8_MODEL) {}

  private async getSession(): Promise<ort.InferenceSession> {
    if (!this.session) {
      this.session = await ort.InferenceSession.create(this.modelPath);
    }
    return this.session;
  }

  async detect(image: Buffer): Promise<Detection[]> {
    const session = await this.getSession();
    const { tensor, letterbox } = await this.preprocess(image);
    const results = await session.run({ [session.inputNames[0]]: tensor });
    return postprocessYolo(pickYoloOutput(results), letterbox);
  }

  private async preprocess(
    image: Buffer
  ): Promise<{ tensor: ort.Tensor; letterbox: Letterbox }> {
    const meta = await sharp(image).metadata();
    const origW = meta.width ?? 0;
    const origH = meta.height ?? 0;
    if (!origW || !origH) throw new Error("qr-detect: could not read image dimensions");

    const scale = Math.min(YOLO_INPUT / origW, YOLO_INPUT / origH);
    const newW = Math.round(origW * scale);
    const newH = Math.round(origH * scale);
    const padLeft = Math.floor((YOLO_INPUT - newW) / 2);
    const padTop = Math.floor((YOLO_INPUT - newH) / 2);

    const { data } = await sharp(image)
      .resize(newW, newH, { fit: "fill" })
      .extend({
        top: padTop,
        bottom: YOLO_INPUT - newH - padTop,
        left: padLeft,
        right: YOLO_INPUT - newW - padLeft,
        background: { r: PAD_VALUE, g: PAD_VALUE, b: PAD_VALUE },
      })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixels = YOLO_INPUT * YOLO_INPUT;
    const chw = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      chw[i] = data[i * 3] / 255;
      chw[pixels + i] = data[i * 3 + 1] / 255;
      chw[2 * pixels + i] = data[i * 3 + 2] / 255;
    }
    return {
      tensor: new ort.Tensor("float32", chw, [1, 3, YOLO_INPUT, YOLO_INPUT]),
      letterbox: { scale, padX: padLeft, padY: padTop, origW, origH },
    };
  }
}

function pickYoloOutput(results: ort.InferenceSession.OnnxValueMapType): ort.Tensor {
  for (const v of Object.values(results)) {
    const t = v as ort.Tensor;
    if (t.dims.length === 3 && t.dims[2] === 8400) return t;
  }
  throw new Error("qr-detect: YOLOv8 detection output [1,37,8400] not found");
}

function postprocessYolo(output: ort.Tensor, lb: Letterbox): Detection[] {
  const data = output.data as Float32Array;
  const n = output.dims[2];
  const cands: Detection[] = [];
  for (let a = 0; a < n; a++) {
    const score = data[4 * n + a];
    if (score <= SCORE_THRESHOLD) continue;
    const cx = data[a];
    const cy = data[n + a];
    const bw = data[2 * n + a];
    const bh = data[3 * n + a];
    let x1 = (cx - bw / 2 - lb.padX) / lb.scale;
    let y1 = (cy - bh / 2 - lb.padY) / lb.scale;
    let x2 = (cx + bw / 2 - lb.padX) / lb.scale;
    let y2 = (cy + bh / 2 - lb.padY) / lb.scale;
    x1 = Math.max(0, Math.min(lb.origW, x1));
    y1 = Math.max(0, Math.min(lb.origH, y1));
    x2 = Math.max(0, Math.min(lb.origW, x2));
    y2 = Math.max(0, Math.min(lb.origH, y2));
    if (x2 - x1 < 1 || y2 - y1 < 1) continue;
    cands.push({ bbox: [x1, y1, x2, y2], confidence: score });
  }
  return nms(cands, NMS_IOU);
}

function iou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(dets: Detection[], iouThresh: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  for (const d of sorted) {
    if (kept.every((k) => iou(k.bbox, d.bbox) < iouThresh)) kept.push(d);
  }
  return kept;
}

// ----------------------------------------------------------------------------

/**
 * Pick the bundled detector: prefer the permissive torchvision model
 * (`models/qr-detector.onnx`) if present, else fall back to the AGPL qrdet-s.
 */
export function createDefaultDetector(): QrDetector {
  if (existsSync(FASTERRCNN_MODEL)) return new FasterRcnnQrDetector();
  if (existsSync(YOLOV8_MODEL)) return new Yolov8QrDetector();
  throw new Error(
    "qr-detect: no model found. Expected models/qr-detector.onnx (preferred) " +
      "or models/qrdet-s.onnx."
  );
}
