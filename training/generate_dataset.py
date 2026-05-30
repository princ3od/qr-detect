#!/usr/bin/env python3
"""Synthetic QR-detection dataset generator.

QR-code detection is an easy single-class task where synthetic data works very
well: we place the QR ourselves, so every bounding box is pixel-perfect and free.
The generator targets the *hard* regime classical decoders miss (small QRs in big
photos; blur, low contrast, rotation, perspective, JPEG) AND -- crucially --
trains the model to *reject* look-alikes by injecting unlabeled distractors:

  * hard negatives: checkerboards, binary-noise squares, barcode stripes, photo
    blobs (square high-frequency texture that is NOT a QR -> no finder pattern),
  * text clutter and CCCD-like "document cards" (header + fields + portrait rect
    + a QR in a corner) so the model sees QRs surrounded by text, like real IDs.

Without these, a detector trained only on QR-on-plain-bg learns "square texture =
QR" and fires on emblems, portraits and text blocks. Payloads mimic CCCD density.

Output:
    <out>/images/<split>/{i:06d}.jpg
    <out>/labels_<split>.json   # [{"file": "...", "boxes": [[x1,y1,x2,y2], ...]}]
Only QRs are labelled (single class). No external assets, no licensing concerns.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
from pathlib import Path

import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# ----------------------------- fonts ------------------------------------------

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def load_font(size: int):
    size = max(8, int(size))
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def _rand_text(min_w: int = 3, max_w: int = 12) -> str:
    words = []
    for _ in range(random.randint(min_w, max_w)):
        words.append("".join(random.choices(
            string.ascii_letters + string.digits, k=random.randint(2, 9))))
    return " ".join(words)


# ----------------------------- QR payloads -----------------------------------

def _rand_digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))


def _rand_name() -> str:
    syll = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Dang", "Bui",
            "Do", "Ho", "Quoc", "Thuong", "Minh", "Anh", "Thi", "Van", "Duc"]
    return " ".join(random.choices(syll, k=random.randint(2, 4)))


def _rand_address() -> str:
    parts = ["Xom", "Thon", "Xa", "Huyen", "Tinh", "Phuong", "Quan", "TP"]
    return ", ".join(f"{random.choice(parts)} {random.randint(1, 30)}"
                     for _ in range(random.randint(3, 6)))


def random_payload() -> str:
    r = random.random()
    if r < 0.6:
        return "|".join([
            _rand_digits(12),
            _rand_digits(9) if random.random() < 0.5 else "",
            _rand_name(), _rand_digits(8),
            random.choice(["Nam", "Nu"]), _rand_address(), _rand_digits(8),
        ])
    if r < 0.8:
        return "https://example.com/" + _rand_digits(random.randint(4, 20))
    n = random.randint(8, 140)
    return "".join(random.choices(string.ascii_letters + string.digits + " :/.-|", k=n))


def make_qr_image() -> Image.Image:
    """QR as RGBA with transparent margin so rotation/perspective keeps a tight
    alpha mask for exact bbox recovery."""
    ec = random.choice([qrcode.constants.ERROR_CORRECT_L, qrcode.constants.ERROR_CORRECT_M,
                        qrcode.constants.ERROR_CORRECT_Q, qrcode.constants.ERROR_CORRECT_H])
    qr = qrcode.QRCode(error_correction=ec, box_size=random.randint(4, 10),
                       border=random.randint(1, 4))
    qr.add_data(random_payload())
    qr.make(fit=True)
    dark = random.choice([(0, 0, 0), (20, 20, 30), (10, 30, 10)])
    light = random.choice([(255, 255, 255), (245, 245, 240), (250, 248, 255)])
    return qr.make_image(fill_color=dark, back_color=light).convert("RGBA")


# ----------------------------- backgrounds ------------------------------------

def random_background(w: int, h: int) -> Image.Image:
    kind = random.random()
    if kind < 0.25:
        base = Image.new("RGB", (w, h), tuple(random.randint(0, 255) for _ in range(3)))
    elif kind < 0.55:
        c1 = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
        vertical = random.random() < 0.5
        t = np.linspace(0, 1, h if vertical else w)[:, None]
        line = (c1[None, :] * (1 - t) + c2[None, :] * t).astype(np.uint8)
        arr = (np.repeat(line[:, None, :], w, axis=1) if vertical
               else np.repeat(line[None, :, :], h, axis=0))
        base = Image.fromarray(arr)
    elif kind < 0.8:
        small = np.random.randint(0, 255, (max(2, h // 40), max(2, w // 40), 3), np.uint8)
        base = Image.fromarray(small).resize((w, h), Image.BICUBIC)
        base = base.filter(ImageFilter.GaussianBlur(random.uniform(1, 4)))
    else:
        base = Image.new("RGB", (w, h), tuple(random.randint(0, 255) for _ in range(3)))
        d = ImageDraw.Draw(base)
        for _ in range(random.randint(3, 12)):
            x0, y0 = random.randint(0, w), random.randint(0, h)
            x1, y1 = random.randint(0, w), random.randint(0, h)
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                        fill=tuple(random.randint(0, 255) for _ in range(3)))
    # background text clutter (like a desk mat / surrounding scene)
    if random.random() < 0.5:
        draw_text_clutter(base, w, h, random.randint(3, 12),
                          color_dark=random.random() < 0.5)
    return base


def draw_text_clutter(img: Image.Image, w: int, h: int, n: int, color_dark=True):
    d = ImageDraw.Draw(img)
    for _ in range(n):
        size = random.randint(max(10, h // 60), max(14, h // 18))
        font = load_font(size)
        col = (random.randint(0, 60),) * 3 if color_dark else (random.randint(200, 255),) * 3
        d.text((random.randint(0, w), random.randint(0, h)), _rand_text(),
               fill=col, font=font)


# ----------------------------- hard negatives ---------------------------------

def _checkerboard(side: int) -> Image.Image:
    cells = random.randint(4, 16)
    g = (np.indices((cells, cells)).sum(0) % 2 * 255).astype(np.uint8)
    im = Image.fromarray(g).resize((side, side), Image.NEAREST).convert("RGB")
    return im


def _binary_noise(side: int) -> Image.Image:
    # QR-like dense square texture but WITHOUT finder patterns
    cells = random.randint(15, 40)
    g = (np.random.rand(cells, cells) < 0.5).astype(np.uint8) * 255
    im = Image.fromarray(g).resize((side, side), Image.NEAREST).convert("RGB")
    return im


def _stripes(side: int) -> Image.Image:
    arr = np.zeros((side, side), np.uint8)
    x = 0
    while x < side:
        bw = random.randint(2, max(3, side // 20))
        if random.random() < 0.5:
            arr[:, x:x + bw] = 255
        x += bw
    return Image.fromarray(arr).convert("RGB")


def _photo_blob(side: int) -> Image.Image:
    small = np.random.randint(0, 255, (random.randint(3, 8), random.randint(3, 8), 3), np.uint8)
    return Image.fromarray(small).resize((side, side), Image.BICUBIC)


def add_hard_negatives(img: Image.Image, w: int, h: int, n: int):
    makers = [_checkerboard, _binary_noise, _binary_noise, _stripes, _photo_blob]
    for _ in range(n):
        side = max(20, int(min(w, h) * random.uniform(0.04, 0.30)))
        patch = random.choice(makers)(side).convert("RGBA")
        # mild rotation so they aren't all axis-aligned
        patch = patch.rotate(random.uniform(-20, 20), expand=True, resample=Image.BICUBIC)
        if patch.width >= w or patch.height >= h:
            continue
        ox = random.randint(0, w - patch.width)
        oy = random.randint(0, h - patch.height)
        img.paste(patch, (ox, oy), patch)


# ----------------------------- document card ----------------------------------

def draw_document_card(bg: Image.Image, w: int, h: int):
    """Light card with header + field lines + portrait rect; returns the corner
    rect (x1,y1,x2,y2) where a QR should be placed, mimicking a CCCD."""
    cw = random.randint(int(w * 0.55), int(w * 0.95))
    ch = random.randint(int(cw * 0.55), int(cw * 0.72))
    ch = min(ch, int(h * 0.95))
    cx = random.randint(0, max(1, w - cw))
    cy = random.randint(0, max(1, h - ch))
    shade = random.randint(200, 250)
    d = ImageDraw.Draw(bg)
    d.rectangle([cx, cy, cx + cw, cy + ch], fill=(shade, shade, shade - random.randint(0, 15)))
    # portrait rectangle (lower-left) — a common false positive source
    pw = int(cw * random.uniform(0.18, 0.26))
    ph = int(pw * 1.3)
    px, py = cx + int(cw * 0.04), cy + int(ch * 0.4)
    d.rectangle([px, py, px + pw, py + ph], fill=tuple(random.randint(90, 180) for _ in range(3)))
    # header + field text lines
    fs = max(10, int(ch * 0.05))
    for i in range(random.randint(5, 9)):
        font = load_font(int(fs * random.uniform(0.8, 1.3)))
        d.text((cx + int(cw * 0.30), cy + int(ch * 0.08) + i * int(ch * 0.10)),
               _rand_text(2, 7), fill=(random.randint(0, 70),) * 3, font=font)
    # QR target corner (top-right or bottom-right of card)
    qs = int(min(cw, ch) * random.uniform(0.16, 0.30))
    if random.random() < 0.5:
        qx, qy = cx + cw - qs - int(cw * 0.03), cy + int(ch * 0.06)
    else:
        qx, qy = cx + cw - qs - int(cw * 0.03), cy + ch - qs - int(ch * 0.06)
    return (qx, qy, qx + qs, qy + qs)


# ----------------------------- geometry ---------------------------------------

def _perspective_coeffs(src, dst):
    matrix = []
    for s, dd in zip(src, dst):
        matrix.append([dd[0], dd[1], 1, 0, 0, 0, -s[0] * dd[0], -s[0] * dd[1]])
        matrix.append([0, 0, 0, dd[0], dd[1], 1, -s[1] * dd[0], -s[1] * dd[1]])
    A = np.array(matrix, np.float64)
    B = np.array(src, np.float64).reshape(8)
    return np.linalg.solve(A, B).tolist()


def warp_qr(qr: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        w, h = qr.size
        m = min(w, h) * random.uniform(0.0, 0.18)
        dst = [(random.uniform(0, m), random.uniform(0, m)),
               (w - random.uniform(0, m), random.uniform(0, m)),
               (w - random.uniform(0, m), h - random.uniform(0, m)),
               (random.uniform(0, m), h - random.uniform(0, m))]
        try:
            qr = qr.transform((w, h), Image.PERSPECTIVE,
                              _perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)], dst),
                              resample=Image.BICUBIC)
        except np.linalg.LinAlgError:
            pass
    angle = (random.uniform(-15, 15) if random.random() < 0.8
             else random.choice([random.uniform(-45, 45), 90, 180, 270]))
    return qr.rotate(angle, expand=True, resample=Image.BICUBIC)


# ----------------------------- compositing ------------------------------------

def degrade(img: Image.Image) -> Image.Image:
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.2)))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.BoxBlur(random.uniform(0.5, 2.0)))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.45, 1.15))
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.4))
    if random.random() < 0.4:
        arr = np.asarray(img).astype(np.int16)
        arr += np.random.normal(0, random.uniform(3, 18), arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def _place_qr(bg, w, h, target_rect=None):
    qr = warp_qr(make_qr_image())
    if target_rect is not None:
        tx1, ty1, tx2, ty2 = target_rect
        target = max(16, min(tx2 - tx1, ty2 - ty1))
        scale = target / max(qr.size)
        qr = qr.resize((max(8, int(qr.width * scale)), max(8, int(qr.height * scale))),
                       Image.BICUBIC)
        ox = min(max(0, tx1), w - qr.width)
        oy = min(max(0, ty1), h - qr.height)
    else:
        frac = random.uniform(0.04, 0.38)
        target = max(16, int(min(w, h) * frac))
        scale = target / max(qr.size)
        qr = qr.resize((max(8, int(qr.width * scale)), max(8, int(qr.height * scale))),
                       Image.BICUBIC)
        if qr.width >= w or qr.height >= h:
            return None
        ox = random.randint(0, w - qr.width)
        oy = random.randint(0, h - qr.height)
    bg.paste(qr, (ox, oy), qr)
    bb = qr.getbbox()
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return [ox + x1, oy + y1, ox + x2, oy + y2]


def make_sample(min_size: int, max_size: int):
    w = random.randint(min_size, max_size)
    h = random.randint(min_size, max_size)
    bg = random_background(w, h).convert("RGB")
    boxes = []

    is_card = random.random() < 0.5
    has_qr = random.random() > 0.08  # ~8% pure negatives (no QR at all)

    target_rect = draw_document_card(bg, w, h) if is_card else None

    if has_qr:
        n_qr = 2 if (not is_card and random.random() < 0.12) else 1
        for k in range(n_qr):
            box = _place_qr(bg, w, h, target_rect if k == 0 else None)
            if box:
                boxes.append(box)

    # unlabeled distractors: hard negatives + extra text clutter
    add_hard_negatives(bg, w, h, random.randint(1, 5))
    if random.random() < 0.5:
        draw_text_clutter(bg, w, h, random.randint(2, 8), color_dark=random.random() < 0.5)

    return degrade(bg), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", type=str)
    ap.add_argument("--train", default=8000, type=int)
    ap.add_argument("--val", default=500, type=int)
    ap.add_argument("--min-size", default=700, type=int)
    ap.add_argument("--max-size", default=1800, type=int)
    ap.add_argument("--jpeg-min", default=45, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--debug-grid", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)

    debug_samples = []
    for split, count in (("train", args.train), ("val", args.val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        labels = []
        for i in range(count):
            img, boxes = make_sample(args.min_size, args.max_size)
            fname = f"images/{split}/{i:06d}.jpg"
            img.save(out / fname, "JPEG", quality=random.randint(args.jpeg_min, 95))
            labels.append({"file": fname, "boxes": boxes})
            if args.debug_grid and split == "train" and len(debug_samples) < 16:
                debug_samples.append((img.copy(), boxes))
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{count}")
        with open(out / f"labels_{split}.json", "w") as f:
            json.dump(labels, f)
        print(f"wrote {count} {split} images + labels_{split}.json")

    if args.debug_grid and debug_samples:
        cell, cols = 256, 4
        rows = (len(debug_samples) + cols - 1) // cols
        grid = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
        for idx, (im, bxs) in enumerate(debug_samples):
            sx, sy = cell / im.width, cell / im.height
            im2 = im.convert("RGB").resize((cell, cell))
            d = ImageDraw.Draw(im2)
            for (x1, y1, x2, y2) in bxs:
                d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(0, 255, 0), width=3)
            grid.paste(im2, ((idx % cols) * cell, (idx // cols) * cell))
        grid.save(out / "debug_grid.jpg", "JPEG", quality=90)
        print(f"wrote {out / 'debug_grid.jpg'}")


if __name__ == "__main__":
    main()
