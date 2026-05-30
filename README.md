# @homstera/qr-detect

Detector-first QR localization + decoding for **hard cases**: small, blurry, or
low-contrast QR codes such as the one printed on Vietnamese CCCD ID cards, where
naive full-image decoding fails.

Instead of brute-forcing the whole image, it:

1. Localizes the QR with a small object-detection model (ONNX).
2. Crops the bounding box (+15% padding), upscales, and decodes the crop with `zxing-wasm`.
3. Falls back to decoding the full image if every crop fails.

## Models (and why there are two)

The localizer sits behind a `QrDetector` interface and is **auto-selected** at
runtime from whichever ONNX file is bundled in `models/`:

| File | Detector | License | Status |
| --- | --- | --- | --- |
| `models/qr-detector.onnx` | `FasterRcnnQrDetector` (torchvision) | **BSD-3 / yours** | **preferred** — train it yourself (see [`training/`](training/)) |
| `models/qrdet-s.onnx` | `Yolov8QrDetector` (qrdet) | **AGPL-3.0** | fallback — works great but viral license (see [License](#-license)) |

`createDefaultDetector()` prefers `qr-detector.onnx` when present, so the moment
you drop in your trained model the package becomes fully permissive — **no code
change required**. The whole point of the interface is this swap.

## Install

```bash
npm install @homstera/qr-detect
```

Self-contained: the ONNX model is bundled and the zxing WASM is loaded from the
installed package, so it works offline after install. Requires Node 18+.

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

## Training your own permissive detector (`training/`)

To get a model with **no AGPL strings attached**, train the bundled torchvision
detector (Faster R-CNN + MobileNetV3-Large-FPN, BSD-3) on **synthetic data** —
QR detection is an easy single-class task where synthetic data works great and
gives pixel-perfect labels for free. Open [`training/qr_detector_colab.ipynb`](training/qr_detector_colab.ipynb)
in Google Colab (GPU runtime) and **Run all**:

1. `generate_dataset.py` — renders QRs (CCCD-like density) onto varied backgrounds
   at small scale with blur/rotation/perspective/low-contrast/JPEG augmentation.
2. `train.py` — fine-tunes the COCO-pretrained detector (LR warmup + grad clipping).
3. `export_onnx.py` — exports `qr-detector.onnx`.

Then drop the file into `models/qr-detector.onnx` and run `npm test`. The package
auto-switches to it. Locally you can run the same scripts (see `training/requirements.txt`);
CPU works but is slow — a GPU is strongly recommended.

The exported torchvision ONNX bakes in resize+normalize **and** NMS:

- **Input** `images`: float32 RGB `[0,1]`, dims `[3,H,W]` (dynamic H/W, not batched).
- **Outputs** `boxes` `[N,4]` xyxy (input-pixel coords), `scores` `[N]`, `labels` `[N]` (ignored).

<details>
<summary>AGPL fallback model (<code>qrdet-s.onnx</code>) ONNX I/O</summary>

Exported once from qrdet-s (`pip install ultralytics`; `YOLO("qrdet-s.pt").export(format="onnx", imgsz=640, opset=12)`).
Segmentation model — only the detection head is read:

- **Input** `images` `[1,3,640,640]` float32, RGB, `/255`, CHW, letterboxed (gray 114 pad).
- **Output 0** `[1,37,8400]`: cols 0..3 = `cx,cy,w,h` (640 space), col 4 = score, cols 5..36 mask coeffs (**ignored**).
- **Output 1** mask prototypes `[1,32,160,160]` (**ignored**).
</details>

## ⚠️ License

This repo currently ships **two** detector options (see [Models](#models-and-why-there-are-two)):

- **`qr-detector.onnx`** (torchvision Faster R-CNN, **BSD-3**) — the model and the
  whole training stack are permissive. Weights you train are **yours** to license
  (MIT/Apache). This is the recommended path for a commercial SaaS.
- **`qrdet-s.onnx`** (Ultralytics YOLOv8 / qrdet, **AGPL-3.0**) — the fallback. AGPL
  is viral and treats *network use as distribution*: offering it over a network
  service can trigger the obligation to release your corresponding source under
  AGPL. **A real risk for a commercial SaaS.** We benchmarked the obvious Apache
  swap (OpenCV `WeChatQRCode`) and it only decoded 1/6 of real CCCD photos, which
  is why "train your own" is the recommended permissive route rather than a
  drop-in pretrained model.

If you ship **only** `qr-detector.onnx` (delete `qrdet-s.onnx`), nothing AGPL is
distributed or run, and you can relicense this package to BSD/MIT. While the AGPL
model is bundled, the package is published as `AGPL-3.0-only`.
