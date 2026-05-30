# @homstera/qr-detect

Detector-first QR localization + decoding for **hard cases**: small, blurry, or
low-contrast QR codes such as the one printed on Vietnamese CCCD ID cards, where
naive full-image decoding fails.

Instead of brute-forcing the whole image, it:

1. Localizes the QR with a YOLOv8 detector (qrdet).
2. Crops the bounding box (+15% padding), upscales, and decodes the crop with `zxing-wasm`.
3. Falls back to decoding the full image if every crop fails.

## Install

```bash
npm install @homstera/qr-detect
```

Self-contained: the ONNX model is bundled (`models/qrdet-s.onnx`) and the zxing
WASM is loaded from the installed package, so it works offline after install.
Requires Node 18+.

## Usage

```ts
import { detectAndDecodeQr } from "@homstera/qr-detect";
import { readFileSync } from "node:fs";

const result = await detectAndDecodeQr(readFileSync("cccd.jpg"));
// => { text, bbox: [x1,y1,x2,y2], confidence, source: "crop" | "fallback" } | null

if (result) {
  console.log(result.text); // "052206008097||Tạ Quốc Thưởng|19102006|Nam|…"
}
```

### API

```ts
function detectAndDecodeQr(
  input: Buffer,
  opts?: {
    detector?: QrDetector;     // swap the model; defaults to bundled ONNX qrdet-s
    maxCropAttempts?: number;  // how many detections to try via crop (default 3)
  }
): Promise<DetectResult | null>;

type DetectResult = {
  text: string;
  bbox: [number, number, number, number]; // x1,y1,x2,y2 in the EXIF-oriented image
  confidence: number;
  source: "crop" | "fallback";
};
```

The detector lives behind a `QrDetector` interface:

```ts
interface QrDetector {
  detect(image: Buffer): Promise<Detection[]>; // Detection = { bbox, confidence }
}
```

The bundled implementation is `OnnxQrDetector`. Because the decode pipeline only
depends on this interface, the model can be swapped (see **License** below)
without changing callers.

### CCCD payload

The decoded string is pipe-delimited:

```
CCCD_number | old_id | name | dob(ddmmyyyy) | sex | address | issue_date(ddmmyyyy)
```

## Benchmark

Measured on 6 real CCCD photos (Apple Silicon, CPU, after warmup):

| Approach                              | Decoded | Notes                                            |
| ------------------------------------- | ------- | ------------------------------------------------ |
| zxing brute-force tiling (old)        | 4/6     | 2 misses cost 1.6s and 3.5s                      |
| qrdet **localization** only           | 6/6     | confidence 0.89–0.97, ~120–230ms/image           |
| qrdet crop → decode (this package)    | 6/6     | crop hit 6/6; full-image fallback covers any miss |
| OpenCV `QRCodeDetector` (classical)   | 0/6     | classical detectors are too weak — the DL detector is the point |

Per-image total runtime is ~210–320ms after warmup (detection + crop-decode).

## How decode works

zxing's binarizer is resolution-sensitive: a dense CCCD QR often decodes at one
upscale but not a neighbouring one. The crop is therefore decoded across a short
sweep of short-side targets (`[900, 800, 1100, 1600, 2400]`px) with early exit.
The decode feeds zxing a raw RGBA buffer produced by `sharp`:

```ts
const { data, info } = await pipe.ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const imageData = {
  data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
  width: info.width,
  height: info.height,
};
await readBarcodes(imageData, { formats: ["QRCode"], tryHarder: true, tryDownscale: true, maxNumberOfSymbols: 1 });
```

Input is EXIF auto-oriented once (`sharp(input).rotate()`) and the oriented
buffer is reused for detection, cropping, and the fallback.

## Test / acceptance

Fixtures are read from `/tmp/cccd-*.jpg`.

```bash
npm install
npm run build
npx tsx scripts/test.ts                       # acceptance over all 6 fixtures
npx tsx scripts/test.ts /tmp/cccd-front-orig.jpg   # single image -> { text, bbox, confidence, ms }
```

Acceptance: localize 6/6, crop-decode ≥5/6, total 6/6 including the fallback.

## Regenerating the model

The bundled `models/qrdet-s.onnx` was exported once from qrdet-s weights:

```python
# pip install ultralytics
from ultralytics import YOLO
YOLO("qrdet-s.pt").export(format="onnx", imgsz=640, opset=12)
# weights: https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-s.pt
```

ONNX I/O (segmentation model — only the boxes are used):

- **Input** `images` `[1,3,640,640]` float32, RGB, `/255`, CHW, letterboxed (gray 114 pad).
- **Output 0** `[1,37,8400]` → cols 0..3 = `cx,cy,w,h` (640 space), col 4 = score,
  cols 5..36 = 32 mask coefficients (**ignored**).
- **Output 1** mask prototypes `[1,32,160,160]` (**ignored**).

## ⚠️ License

`qrdet`'s weights are trained with **Ultralytics YOLOv8**, which is licensed
**AGPL-3.0**. AGPL is viral and treats *network use as distribution* — i.e.
offering this over a network service can trigger the obligation to release your
corresponding source under AGPL. **This is a real risk for a commercial SaaS.**

Mitigation / swap path: the model sits behind the `QrDetector` interface, so an
Apache/MIT-licensed detector can drop in behind the same `detectAndDecodeQr` API
without touching callers — e.g. OpenCV `WeChatQRCode`, or a self-trained
MIT/Apache YOLO. Only the detector module changes; the crop/decode/fallback stay.

This package is published as `AGPL-3.0-only` to reflect the bundled weights.
Replace the detector to relicense.
