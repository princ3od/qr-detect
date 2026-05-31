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
 * underlying model is replaceable — implement it to plug in your own detector.
 */
export interface QrDetector {
  detect(image: Buffer): Promise<Detection[]>;
}

const here = dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODEL = resolve(here, "../models/qr-detector.onnx");

const SCORE_THRESHOLD = 0.3;
const MAX_SIDE = 1024; // cap input so inference stays fast on big photos

/**
 * ONNX detector for the bundled torchvision Faster R-CNN + MobileNetV3-FPN model.
 * The exported graph bakes in resize+normalize and NMS:
 *   input  "images": float32 RGB [0,1], dims [3,H,W]
 *   output "boxes" [N,4] xyxy, "scores" [N] (and "labels" [N], ignored)
 * The session is created once and reused.
 */
export class FasterRcnnQrDetector implements QrDetector {
  private session: ort.InferenceSession | null = null;
  constructor(private readonly modelPath: string = DEFAULT_MODEL) {}

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
      .resize({ width: MAX_SIDE, height: MAX_SIDE, fit: "inside", withoutEnlargement: true })
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
    // NMS is already applied inside the model; just order by confidence
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
  throw new Error("qr-detect: expected detector output not found");
}

/** The bundled detector. Throws if `models/qr-detector.onnx` is missing. */
export function createDefaultDetector(): QrDetector {
  if (!existsSync(DEFAULT_MODEL)) {
    throw new Error(`qr-detect: model not found at ${DEFAULT_MODEL}`);
  }
  return new FasterRcnnQrDetector();
}
