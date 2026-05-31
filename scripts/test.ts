/**
 * CLI / smoke test for @homstera/qr-detect.
 *
 *   tsx scripts/test.ts <image>          # one image -> { text, bbox, confidence, source, ms }
 *   tsx scripts/test.ts <directory>      # every image in the dir -> table + summary
 *   tsx scripts/test.ts                  # uses $QR_FIXTURES or ./fixtures
 *
 * No sample images ship with the repo. Point it at your own folder of photos.
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { detectAndDecodeQr } from "../src/index.js";

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp"]);

function collectImages(target: string): string[] {
  if (!existsSync(target)) return [];
  if (statSync(target).isDirectory()) {
    return readdirSync(target)
      .filter((f) => IMAGE_EXTS.has(extname(f).toLowerCase()))
      .map((f) => join(target, f))
      .sort();
  }
  return [target];
}

function round(b: readonly number[]): string {
  return `[${b.map((n) => Math.round(n)).join(", ")}]`;
}

async function single(path: string): Promise<void> {
  const t0 = performance.now();
  const res = await detectAndDecodeQr(readFileSync(path));
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

async function batch(paths: string[]): Promise<void> {
  let decoded = 0;
  for (const path of paths) {
    const name = path.split("/").pop() ?? path;
    const t0 = performance.now();
    const res = await detectAndDecodeQr(readFileSync(path));
    const ms = Math.round(performance.now() - t0);
    if (res) decoded++;
    const status = res ? "HIT " : "MISS";
    const src = (res?.source ?? "-").padEnd(8);
    const conf = res ? res.confidence.toFixed(2) : "----";
    const text = res ? ` "${res.text.slice(0, 40)}${res.text.length > 40 ? "…" : ""}"` : "";
    console.log(
      `${status} ${name.padEnd(26)} ${conf} ${round(res?.bbox ?? [0, 0, 0, 0]).padEnd(26)} ` +
        `src=${src} ${String(ms).padStart(4)}ms${text}`
    );
  }
  console.log(`\ndecoded ${decoded}/${paths.length}`);
  if (decoded === 0) process.exitCode = 1;
}

const arg = process.argv[2];
const target = arg ?? process.env.QR_FIXTURES ?? "fixtures";
const images = collectImages(target);

if (images.length === 0) {
  console.error(
    `No images found at "${target}".\n` +
      `Usage: tsx scripts/test.ts <image|directory>\n` +
      `   or: QR_FIXTURES=/path/to/images tsx scripts/test.ts\n` +
      `(No sample images ship with the repo.)`
  );
  process.exit(1);
}

if (images.length === 1) {
  await single(images[0]);
} else {
  await batch(images);
}
