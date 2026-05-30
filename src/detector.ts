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
 * underlying model (currently AGPL YOLOv8/qrdet) can be swapped for an
 * Apache/MIT detector without changing callers. See README "License".
 */
export interface QrDetector {
  detect(image: Buffer): Promise<Detection[]>;
}

const MODEL_INPUT = 640;
const PAD_VALUE = 114;
const SCORE_THRESHOLD = 0.3;
const NMS_IOU = 0.45;

const here = dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL_PATH = resolve(here, "../models/qrdet-s.onnx");

interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
  origW: number;
  origH: number;
}

/** ONNX-backed YOLOv8 QR detector (qrdet-s). Session is created once and reused. */
export class OnnxQrDetector implements QrDetector {
  private session: ort.InferenceSession | null = null;
  private readonly modelPath: string;

  constructor(modelPath: string = DEFAULT_MODEL_PATH) {
    this.modelPath = modelPath;
  }

  private async getSession(): Promise<ort.InferenceSession> {
    if (!this.session) {
      this.session = await ort.InferenceSession.create(this.modelPath);
    }
    return this.session;
  }

  async detect(image: Buffer): Promise<Detection[]> {
    const session = await this.getSession();
    const { tensor, letterbox } = await this.preprocess(image);
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: tensor,
    };
    const results = await session.run(feeds);
    const output = pickDetectionOutput(results);
    return postprocess(output, letterbox);
  }

  private async preprocess(
    image: Buffer
  ): Promise<{ tensor: ort.Tensor; letterbox: Letterbox }> {
    const meta = await sharp(image).metadata();
    const origW = meta.width ?? 0;
    const origH = meta.height ?? 0;
    if (!origW || !origH) {
      throw new Error("qr-detect: could not read image dimensions");
    }

    const scale = Math.min(MODEL_INPUT / origW, MODEL_INPUT / origH);
    const newW = Math.round(origW * scale);
    const newH = Math.round(origH * scale);
    const padLeft = Math.floor((MODEL_INPUT - newW) / 2);
    const padTop = Math.floor((MODEL_INPUT - newH) / 2);
    const padRight = MODEL_INPUT - newW - padLeft;
    const padBottom = MODEL_INPUT - newH - padTop;

    const { data } = await sharp(image)
      .resize(newW, newH, { fit: "fill" })
      .extend({
        top: padTop,
        bottom: padBottom,
        left: padLeft,
        right: padRight,
        background: { r: PAD_VALUE, g: PAD_VALUE, b: PAD_VALUE },
      })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    // HWC RGB uint8 -> CHW RGB float32 normalized to [0,1]
    const pixels = MODEL_INPUT * MODEL_INPUT;
    const chw = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      chw[i] = data[i * 3] / 255; // R plane
      chw[pixels + i] = data[i * 3 + 1] / 255; // G plane
      chw[2 * pixels + i] = data[i * 3 + 2] / 255; // B plane
    }

    const tensor = new ort.Tensor("float32", chw, [1, 3, MODEL_INPUT, MODEL_INPUT]);
    return {
      tensor,
      letterbox: { scale, padX: padLeft, padY: padTop, origW, origH },
    };
  }
}

/** Pick output0 ([1,37,8400]); ignore the mask-prototype output ([1,32,160,160]). */
function pickDetectionOutput(
  results: ort.InferenceSession.OnnxValueMapType
): ort.Tensor {
  for (const value of Object.values(results)) {
    const t = value as ort.Tensor;
    if (t.dims.length === 3 && t.dims[2] === 8400) return t;
  }
  throw new Error("qr-detect: detection output [1,37,8400] not found");
}

/**
 * output0 is [1, 37, 8400]: element [0,c,a] = data[c*8400 + a].
 * cols 0..3 = cx,cy,w,h (640 letterboxed space); col 4 = score; cols 5..36 =
 * mask coefficients (ignored).
 */
function postprocess(output: ort.Tensor, lb: Letterbox): Detection[] {
  const data = output.data as Float32Array;
  const numAnchors = output.dims[2]; // 8400
  const cands: Detection[] = [];

  for (let a = 0; a < numAnchors; a++) {
    const score = data[4 * numAnchors + a];
    if (score <= SCORE_THRESHOLD) continue;

    const cx = data[a];
    const cy = data[numAnchors + a];
    const w = data[2 * numAnchors + a];
    const h = data[3 * numAnchors + a];

    // xywh -> xyxy in letterboxed 640 space
    let x1 = cx - w / 2;
    let y1 = cy - h / 2;
    let x2 = cx + w / 2;
    let y2 = cy + h / 2;

    // undo letterbox -> original-image coordinates
    x1 = (x1 - lb.padX) / lb.scale;
    y1 = (y1 - lb.padY) / lb.scale;
    x2 = (x2 - lb.padX) / lb.scale;
    y2 = (y2 - lb.padY) / lb.scale;

    // clip to image bounds
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
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(dets: Detection[], iouThresh: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  for (const d of sorted) {
    if (kept.every((k) => iou(k.bbox, d.bbox) < iouThresh)) {
      kept.push(d);
    }
  }
  return kept;
}
