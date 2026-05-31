# @homstera/qr-detect

Detector-first QR localization + decoding for **hard cases** — small, blurry, or
low-contrast QR codes in photos, where decoding the whole image fails.

Naïve full-image decoders (and even brute-force tiling) miss a small dense QR in a
big photo. This package instead:

1. **Localizes** the QR with a small object-detection model (ONNX).
2. **Crops** the box (+15% padding), upscales, and decodes the crop with [`zxing-wasm`](https://github.com/Sec-ant/zxing-wasm).
3. **Falls back** to decoding the full image (with an upscale sweep) if no crop decodes.

Runs on CPU via `onnxruntime-node`. Self-contained and offline after install.

```ts
import { detectAndDecodeQr } from "@homstera/qr-detect";
import { readFileSync } from "node:fs";

const result = await detectAndDecodeQr(readFileSync("photo.jpg"));
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
buffer is reused for detection, cropping, and the fallback. The result's `text` is
the raw decoded QR payload — interpreting it is up to the caller.

## Model

The bundled detector is `FasterRcnnQrDetector` — a torchvision Faster R-CNN +
MobileNetV3-Large-FPN exported to ONNX (`models/qr-detector.onnx`). It sits behind
the `QrDetector` interface, so you can retrain it (see [`training/`](training/)) or
swap in your own model via `opts.detector` with no other changes. The whole stack is
permissively licensed — no AGPL anywhere.

## Benchmark

Measured on 6 real-world photos with small/blurry QR codes (Apple Silicon, CPU,
steady-state after warmup):

| Approach | License | Decoded | Notes |
| --- | --- | --- | --- |
| **this package** (`qr-detector.onnx`) | **MIT / BSD** | **6/6** | 1 clean box @ 0.99–1.00, no false positives; ~120–260 ms/img |
| qrdet (Ultralytics YOLOv8) | AGPL-3.0 | 6/6 | the approach this was validated against; not bundled (viral license) |
| zxing brute-force tiling | Apache-2.0 | 4/6 | slow — 2 misses cost 1.6 s and 3.5 s |
| OpenCV `WeChatQRCode` | Apache-2.0 | 1/6 | only the already-cropped image |
| ZBar | LGPL-2.1 | 0/6 | can't localize small QRs |
| OpenCV `QRCodeDetector` | BSD | 0/6 | classical detectors are too weak |

The DL detector is the whole point: classical decoders can't localize a small dense
QR inside a big photo. The decode step itself (zxing) is fine once the QR is cropped
out. The only off-the-shelf permissive DL alternative we found (`WeChatQRCode`) got
1/6 — which is why this package ships a model trained from scratch on synthetic data.

## Training your own detector (`training/`)

Train the bundled torchvision detector (Faster R-CNN + MobileNetV3-Large-FPN,
BSD-3) on **synthetic data** — QR detection is an easy single-class task where
synthetic data works well and gives pixel-perfect labels for free.

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
  barcode stripes, portrait rectangles, text, document cards) so the model learns QR
  *finder patterns* instead of "dense square = QR" — this is what kills false
  positives on faces/text/logos.
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
  detector.ts   QrDetector interface, FasterRcnnQrDetector, createDefaultDetector
  decode.ts     crop+upscale+zxing decode, full-image fallback
scripts/test.ts CLI / smoke test
models/         bundled ONNX model
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

No sample images are shipped — point the test at your own photos.

## License

**MIT** — see [LICENSE](LICENSE). The bundled model and the `training/` stack are
permissively licensed too (torchvision, BSD-3).

## Acknowledgements

- [zxing-wasm](https://github.com/Sec-ant/zxing-wasm) — QR decoding.
- [torchvision](https://github.com/pytorch/vision) — the detector backbone.
- [sharp](https://github.com/lovell/sharp) — image I/O.
- [qrdet](https://github.com/Eric-Canas/qrdet) — the detector-first approach this was validated against.
