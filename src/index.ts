import sharp from "sharp";
import {
  createDefaultDetector,
  FasterRcnnQrDetector,
  type BBox,
  type Detection,
  type QrDetector,
} from "./detector.js";
import { decodeCrop, decodeFull } from "./decode.js";

export type { BBox, Detection, QrDetector };
export { createDefaultDetector, FasterRcnnQrDetector };

export interface DetectResult {
  text: string;
  bbox: BBox;
  confidence: number;
  source: "crop" | "fallback";
}

export interface DetectOptions {
  /** Custom detector. Defaults to the bundled FasterRcnnQrDetector. */
  detector?: QrDetector;
  /** Max number of detections to try decoding via crop. Default 3. */
  maxCropAttempts?: number;
}

let sharedDetector: QrDetector | null = null;

/**
 * Detector-first QR pipeline:
 *  1. localize QR boxes with the bundled detector,
 *  2. crop+upscale the best boxes and decode with zxing,
 *  3. fall back to decoding the full image.
 */
export async function detectAndDecodeQr(
  input: Buffer,
  opts: DetectOptions = {}
): Promise<DetectResult | null> {
  const detector = opts.detector ?? (sharedDetector ??= createDefaultDetector());
  const maxCropAttempts = opts.maxCropAttempts ?? 3;

  // EXIF auto-orient once; reuse the oriented buffer everywhere.
  const oriented = await sharp(input).rotate().toBuffer();
  const meta = await sharp(oriented).metadata();
  const w = meta.width ?? 0;
  const h = meta.height ?? 0;
  if (!w || !h) return null;

  const detections = await detector.detect(oriented);

  for (const det of detections.slice(0, maxCropAttempts)) {
    const text = await decodeCrop(oriented, det.bbox, w, h);
    if (text) {
      return { text, bbox: det.bbox, confidence: det.confidence, source: "crop" };
    }
  }

  const fallbackText = await decodeFull(oriented);
  if (fallbackText) {
    return {
      text: fallbackText,
      bbox: detections[0]?.bbox ?? [0, 0, w, h],
      confidence: detections[0]?.confidence ?? 0,
      source: "fallback",
    };
  }

  return null;
}
