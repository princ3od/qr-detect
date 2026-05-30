import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import sharp from "sharp";
import type { ReaderOptions } from "zxing-wasm/reader";
import type { BBox } from "./detector.js";

const require = createRequire(import.meta.url);

const READ_OPTS: ReaderOptions = {
  formats: ["QRCode"],
  tryHarder: true,
  tryDownscale: true,
  maxNumberOfSymbols: 1,
};

const BBOX_PADDING = 0.15; // expand the detected box by 15% on each side
// zxing's binarizer is resolution-sensitive: a dense CCCD QR often decodes at
// one upscale but not a neighbouring one. Sweep a few short-side targets and
// stop at the first hit. Ordered by observed win-rate on real cards.
const UPSCALE_TARGETS = [900, 800, 1100, 1600, 2400];

let zxingPrepared = false;

/**
 * Feed zxing-wasm the bundled .wasm binary so decoding works offline (no CDN
 * fetch). Resolves the wasm next to the installed zxing-wasm package.
 */
async function ensureZxing(): Promise<void> {
  if (zxingPrepared) return;
  zxingPrepared = true;
  try {
    const { prepareZXingModule } = await import("zxing-wasm/reader");
    const pkgJson = require.resolve("zxing-wasm/package.json");
    const wasmPath = resolve(
      dirname(pkgJson),
      "dist/reader/zxing_reader.wasm"
    );
    if (existsSync(wasmPath)) {
      const buf = readFileSync(wasmPath);
      const wasmBinary = buf.buffer.slice(
        buf.byteOffset,
        buf.byteOffset + buf.byteLength
      );
      prepareZXingModule({
        overrides: { wasmBinary },
        fireImmediately: true,
      });
    }
  } catch {
    // fall back to zxing's default module loading
  }
}

async function decodePipe(pipe: sharp.Sharp): Promise<string | null> {
  await ensureZxing();
  try {
    const { readBarcodes } = await import("zxing-wasm/reader");
    const { data, info } = await pipe
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const imageData = {
      data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
      width: info.width,
      height: info.height,
    };
    const results = await readBarcodes(imageData, READ_OPTS);
    return results.find((r) => r.text)?.text ?? null;
  } catch {
    return null;
  }
}

/**
 * Crop the detected bbox (+15% padding), upscale the short side to ~1000px,
 * then decode with zxing. `image` must already be EXIF auto-oriented.
 */
export async function decodeCrop(
  image: Buffer,
  bbox: BBox,
  origW: number,
  origH: number
): Promise<string | null> {
  const [x1, y1, x2, y2] = bbox;
  const w = x2 - x1;
  const h = y2 - y1;
  const px = w * BBOX_PADDING;
  const py = h * BBOX_PADDING;

  const left = Math.max(0, Math.floor(x1 - px));
  const top = Math.max(0, Math.floor(y1 - py));
  const right = Math.min(origW, Math.ceil(x2 + px));
  const bottom = Math.min(origH, Math.ceil(y2 + py));
  const cropW = right - left;
  const cropH = bottom - top;
  if (cropW < 8 || cropH < 8) return null;
  const shortSide = Math.min(cropW, cropH);

  for (const target of UPSCALE_TARGETS) {
    const factor = shortSide > 0 ? target / shortSide : 1;
    const targetW = Math.max(cropW, Math.round(cropW * factor));
    const pipe = sharp(image)
      .extract({ left, top, width: cropW, height: cropH })
      .resize({ width: targetW, kernel: "cubic" });
    const text = await decodePipe(pipe);
    if (text) return text;
  }
  return null;
}

/** Fallback: decode the whole (auto-oriented) image once. */
export async function decodeFull(image: Buffer): Promise<string | null> {
  return decodePipe(sharp(image));
}
