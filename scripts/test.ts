import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { detectAndDecodeQr, createDefaultDetector } from "../src/index.js";

const FIXTURES = [
  "/tmp/cccd-1500.jpg",
  "/tmp/cccd-cardfill.jpg",
  "/tmp/cccd-front-1200.jpg",
  "/tmp/cccd-front-orig.jpg",
  "/tmp/cccd-front-topright.jpg",
  "/tmp/cccd-qr.jpg",
];

function fmtBBox(b: [number, number, number, number]): string {
  return `[${b.map((n) => Math.round(n)).join(", ")}]`;
}

async function single(path: string): Promise<void> {
  const buf = readFileSync(path);
  const t0 = performance.now();
  const res = await detectAndDecodeQr(buf);
  const ms = Math.round(performance.now() - t0);
  console.log(
    JSON.stringify(
      {
        text: res?.text ?? null,
        bbox: res?.bbox ?? null,
        confidence: res?.confidence ?? null,
        source: res?.source ?? null,
        ms,
      },
      null,
      2
    )
  );
}

async function acceptance(): Promise<void> {
  const detector = createDefaultDetector();
  let localized = 0;
  let cropHits = 0;
  let totalHits = 0;

  // warm up the ONNX session so the first timing isn't skewed
  await detector.detect(readFileSync(FIXTURES[0]));

  for (const path of FIXTURES) {
    const buf = readFileSync(path);
    const name = path.split("/").pop();

    const tDet = performance.now();
    const dets = await detector.detect(buf);
    const detMs = Math.round(performance.now() - tDet);
    if (dets.length > 0) localized++;
    const topConf = dets[0]?.confidence ?? 0;

    const t0 = performance.now();
    const res = await detectAndDecodeQr(buf, { detector });
    const ms = Math.round(performance.now() - t0);

    if (res) {
      totalHits++;
      if (res.source === "crop") cropHits++;
    }

    const status = res ? "HIT " : "MISS";
    const src = res ? res.source.padEnd(8) : "-       ";
    const text = res ? ` "${res.text.slice(0, 42)}${res.text.length > 42 ? "…" : ""}"` : "";
    console.log(
      `${status} ${name?.padEnd(24)} det=${dets.length}@${topConf.toFixed(2)} ${fmtBBox(
        dets[0]?.bbox ?? [0, 0, 0, 0]
      ).padEnd(28)} src=${src} det=${detMs}ms total=${ms}ms${text}`
    );
  }

  console.log("\n--- summary ---");
  console.log(`localized:    ${localized}/${FIXTURES.length}`);
  console.log(`crop-decoded: ${cropHits}/${FIXTURES.length}`);
  console.log(`total-decoded ${totalHits}/${FIXTURES.length} (incl. full-image fallback)`);

  const ok = localized === 6 && cropHits >= 5 && totalHits === 6;
  console.log(`\nACCEPTANCE: ${ok ? "PASS" : "FAIL"} (need localize 6/6, crop >=5/6, total 6/6)`);
  if (!ok) process.exitCode = 1;
}

const arg = process.argv[2];
if (arg) {
  await single(arg);
} else {
  await acceptance();
}
