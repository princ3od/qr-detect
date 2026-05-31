# @homstera/qr-detect

Detector-first QR localization + decoding for **hard cases** — small, blurry, or
low-contrast QR codes such as the one printed on Vietnamese CCCD ID cards, where
decoding the whole image fails.

Naïve full-image decoders (and even brute-force tiling) miss the small dense QR on
a card photo. This package instead:

1. **Localizes** the QR with a small object-detection model (ONNX).
2. **Crops** the box (+15% padding), upscales, and decodes the crop with [`zxing-wasm`](https://github.com/Sec-ant/zxing-wasm).
3. **Falls back** to decoding the full image (with an upscale sweep) if no crop decodes.

Runs on CPU via `onnxruntime-node`. Self-contained and offline after install.

```ts
import { detectAndDecodeQr } from "@homstera/qr-detect";
import { readFileSync } from "node:fs";

const result = await detectAndDecodeQr(readFileSync("cccd.jpg"));
// => { text, bbox: [x1,y1,x2,y2], confidence, source: "crop" | "fallback" } | null
```

## Install

```bash
npm install @homstera/qr-detect
```

Dependencies: `onnxruntime-node`, `sharp`, `zxing-wasm`. Requires **Node 18+**.
The ONNX model and the zxing WASM are bundled, so it works offline after install.

## API

```ts
function detectAndDecodeQr(
  input: Buffer,
  opts?: {
    detector?: QrDetector;     // override the model (defaults to the bundled one)
    maxCropAttempts?: number;  // detections to try via crop before fallback (default 3)
  }
): Promise<DetectResult | null>;

type DetectResult = {
  text: string;
  bbox: [number, number, number, number]; // x1,y1,x2,y2 in the EXIF-oriented image
  confidence: number;                      // detector score (0 for fallback)
  source: "crop" | "fallback";
};

interface QrDetector {
  detect(image: Buffer): Promise<{ bbox: [number, number, number, number]; confidence: number }[]>;
}
```

The input is EXIF auto-oriented once (`sharp(input).rotate()`) and the oriented
buffer is reused for detection, cropping, and the fallback.

### CCCD payload

A decoded Vietnamese CCCD QR is a pipe-delimited string:

```
CCCD_number | old_id | name | dob(ddmmyyyy) | sex | address | issue_date(ddmmyyyy)
```

This package only returns the raw decoded `text`; parsing is up to the caller.

## Models

The localizer sits behind the `QrDetector` interface and is **auto-selected** at
runtime from whichever ONNX file is present in `models/`:

| File | Detector class | License | Status |
| --- | --- | --- | --- |
| `models/qr-detector.onnx` | `FasterRcnnQrDetector` (torchvision) | **BSD-3 / yours** | **preferred** — trainable in [`training/`](training/) |
| `models/qrdet-s.onnx` | `Yolov8QrDetector` ([qrdet](https://github.com/Eric-Canas/qrdet)) | **AGPL-3.0** | optional fallback (see [License](#license)) |

`createDefaultDetector()` prefers `qr-detector.onnx` when present, so dropping in a
freshly trained model makes the package fully permissive with **no code change**.
You can also pass your own `QrDetector` via `opts.detector`.

## Benchmark

Measured on 6 real CCCD photos (Apple Silicon, CPU, steady-state after warmup):

| Approach | License | Decoded | Notes |
| --- | --- | --- | --- |
| **`qr-detector.onnx`** (this package) | **BSD-3** | **6/6** | 1 clean box @ 0.99–1.00, no false positives; ~120–260 ms/img |
| `qrdet-s.onnx` (qrdet YOLOv8) | AGPL-3.0 | 6/6 | conf 0.89–0.97; ~120–230 ms/img |
| zxing brute-force tiling | Apache-2.0 | 4/6 | slow — 2 misses cost 1.6 s and 3.5 s |
| OpenCV `WeChatQRCode` | Apache-2.0 | 1/6 | only the already-cropped image |
| ZBar | LGPL-2.1 | 0/6 | can't localize small QRs |
| OpenCV `QRCodeDetector` | BSD | 0/6 | classical detectors are too weak |

The DL detector is the whole point: classical/permissive-by-default decoders can't
localize a small dense QR inside a big photo. The decode step itself (zxing) is fine
once the QR is cropped out.

## Training your own detector (`training/`)

To get a model with **no AGPL strings attached**, train the bundled torchvision
detector (Faster R-CNN + MobileNetV3-Large-FPN, BSD-3) on **synthetic data** — QR
detection is an easy single-class task where synthetic data works well and gives
pixel-perfect labels for free.

Open [`training/qr_detector_colab.ipynb`](training/qr_detector_colab.ipynb) in Google
Colab (GPU runtime) and **Run all**, or run the scripts locally
(`pip install -r training/requirements.txt`):

```bash
python generate_dataset.py --out data --train 8000 --val 500 --debug-grid
python train.py --data data --out runs --epochs 12 --batch-size 8 --lr 0.02
python export_onnx.py --checkpoint runs/qr_frcnn_best.pth --out qr-detector.onnx
```

Then copy `qr-detector.onnx` into `models/` and run `npm test`.

Notes:
- The generator injects **hard negatives** (checkerboards, binary-noise squares,
  barcode stripes, portrait rectangles, text, CCCD-like document cards) so the model
  learns QR *finder patterns* instead of "dense square = QR" — this is what kills
  false positives on faces/text/emblems.
- Training reports **`fp/img`** (false positives per image), not just recall — recall
  alone is misleading (a model that fires everywhere scores perfect recall).
- `export_onnx.py` binds an **onnxruntime-safe postprocess**: stock torchvision exports
  a broken `roi_heads` reshape (`{N,7} -> {-1,4}`) that crashes at inference; the fix
  forces `reshape(-1, num_classes, 4)` with a constant and uses a shape-preserving clip.

Exported ONNX I/O: input `images` float32 RGB `[0,1]`, shape `[3,H,W]` (dynamic H/W,
not batched); outputs `boxes [N,4]` xyxy, `scores [N]`, `labels [N]` — NMS baked in.

## Project layout

```
src/
  index.ts      detectAndDecodeQr + types
  detector.ts   QrDetector interface, FasterRcnnQrDetector, Yolov8QrDetector, createDefaultDetector
  decode.ts     crop+upscale+zxing decode, full-image fallback
scripts/test.ts CLI / smoke test
models/         bundled ONNX model(s)
training/       synthetic data generator, training, ONNX export, Colab notebook
```

## Development

```bash
npm install
npm run build           # tsc -> dist/
npm test                # tsx scripts/test.ts (point it at your own images)

# run on your images:
tsx scripts/test.ts path/to/photo.jpg          # single -> JSON
tsx scripts/test.ts path/to/folder             # batch -> table + summary
QR_FIXTURES=/path/to/images tsx scripts/test.ts
```

No sample cards are shipped — real ID photos are personal data. Bring your own.

## License

The package code is open source. The bundled detector models differ:

- **`qr-detector.onnx`** (torchvision Faster R-CNN) and the entire `training/` stack
  are **BSD-3** — weights you train are yours to license (MIT/Apache).
- **`qrdet-s.onnx`** (Ultralytics YOLOv8 / qrdet) is **AGPL-3.0**. AGPL is viral and
  treats *network use as distribution*: serving it over a network can oblige you to
  release your corresponding source under AGPL — a real risk for a commercial SaaS.

While the AGPL model is bundled, this package is published as **`AGPL-3.0-only`**. To
make it fully permissive, **delete `models/qrdet-s.onnx`** (the package then runs on
the BSD model only) and relicense. We benchmarked the obvious Apache swap, OpenCV
`WeChatQRCode`, and it only decoded 1/6 of real CCCD photos — which is why "train your
own" is the recommended permissive route rather than a drop-in pretrained model.

## Acknowledgements

- [qrdet](https://github.com/Eric-Canas/qrdet) — the YOLOv8 QR detector (AGPL model).
- [zxing-wasm](https://github.com/Sec-ant/zxing-wasm) — QR decoding.
- [torchvision](https://github.com/pytorch/vision) — the permissive detector backbone.
- [sharp](https://github.com/lovell/sharp) — image I/O.
