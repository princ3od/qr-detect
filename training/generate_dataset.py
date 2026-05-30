#!/usr/bin/env python3
"""Synthetic QR-detection dataset generator.

QR-code detection is an easy single-class task where synthetic data works very
well: we place the QR ourselves, so every bounding box is pixel-perfect and free.
The generator deliberately targets the *hard* regime that classical decoders miss:
small QRs inside large photos, with blur, low contrast, rotation, perspective and
JPEG artifacts. Payloads mimic real CCCD density (long byte-mode strings -> high
QR versions with small modules).

Output:
    <out>/images/<split>/{i:06d}.jpg
    <out>/labels_<split>.json   # [{"file": "...", "boxes": [[x1,y1,x2,y2], ...]}]

All boxes are a single class ("QR"). No external assets, no licensing concerns.
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


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
    """Mix CCCD-like records with generic payloads of varied length/density."""
    r = random.random()
    if r < 0.6:
        # CCCD-like: number|old|name|dob|sex|address|issue
        return "|".join([
            _rand_digits(12),
            _rand_digits(9) if random.random() < 0.5 else "",
            _rand_name(),
            _rand_digits(8),
            random.choice(["Nam", "Nu"]),
            _rand_address(),
            _rand_digits(8),
        ])
    if r < 0.8:
        # URL-ish
        return "https://example.com/" + _rand_digits(random.randint(4, 20))
    # arbitrary text of varied length -> varied QR version/density
    n = random.randint(8, 140)
    return "".join(random.choices(string.ascii_letters + string.digits + " :/.-|",
                                  k=n))


def make_qr_image() -> Image.Image:
    """Render a QR as RGBA with a transparent margin so rotation/perspective
    keeps a tight alpha mask for exact bbox recovery."""
    ec = random.choice([
        qrcode.constants.ERROR_CORRECT_L,
        qrcode.constants.ERROR_CORRECT_M,
        qrcode.constants.ERROR_CORRECT_Q,
        qrcode.constants.ERROR_CORRECT_H,
    ])
    qr = qrcode.QRCode(error_correction=ec, box_size=random.randint(4, 10),
                       border=random.randint(1, 4))
    qr.add_data(random_payload())
    qr.make(fit=True)
    dark = random.choice([(0, 0, 0), (20, 20, 30), (10, 30, 10)])
    light = random.choice([(255, 255, 255), (245, 245, 240), (250, 248, 255)])
    img = qr.make_image(fill_color=dark, back_color=light).convert("RGBA")
    return img


# ----------------------------- backgrounds ------------------------------------

def random_background(w: int, h: int) -> Image.Image:
    kind = random.random()
    if kind < 0.25:
        base = Image.new("RGB", (w, h), tuple(random.randint(0, 255) for _ in range(3)))
    elif kind < 0.55:
        # vertical/horizontal gradient
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        t = np.linspace(0, 1, h if random.random() < 0.5 else w)[:, None]
        line = (c1[None, :] * (1 - t) + c2[None, :] * t).astype(np.uint8)
        if line.shape[0] == h:
            arr = np.repeat(line[:, None, :], w, axis=1)
        else:
            arr = np.repeat(line[None, :, :], h, axis=0)
        base = Image.fromarray(arr)
    elif kind < 0.8:
        # smooth noise (card/paper-ish texture)
        small = np.random.randint(0, 255, (max(2, h // 40), max(2, w // 40), 3), np.uint8)
        base = Image.fromarray(small).resize((w, h), Image.BICUBIC)
        base = base.filter(ImageFilter.GaussianBlur(random.uniform(1, 4)))
    else:
        # random colored rectangles (cluttered scene)
        base = Image.new("RGB", (w, h), tuple(random.randint(0, 255) for _ in range(3)))
        d = ImageDraw.Draw(base)
        for _ in range(random.randint(3, 12)):
            x0, y0 = random.randint(0, w), random.randint(0, h)
            x1, y1 = random.randint(0, w), random.randint(0, h)
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                        fill=tuple(random.randint(0, 255) for _ in range(3)))
    # sometimes draw a light "card" rectangle to host the QR (CCCD-like)
    if random.random() < 0.5:
        d = ImageDraw.Draw(base)
        cw, ch = random.randint(w // 3, w), random.randint(h // 3, h)
        cx, cy = random.randint(0, max(1, w - cw)), random.randint(0, max(1, h - ch))
        shade = random.randint(180, 255)
        d.rectangle([cx, cy, cx + cw, cy + ch], fill=(shade, shade, shade))
    return base


# ----------------------------- geometry ---------------------------------------

def _perspective_coeffs(src, dst):
    """PIL PERSPECTIVE coeffs mapping dst->src (8 params)."""
    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0] * d[0], -s[0] * d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1] * d[0], -s[1] * d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def warp_qr(qr: Image.Image) -> Image.Image:
    """Apply rotation + mild perspective; keep alpha so bbox = getbbox()."""
    # mild perspective
    if random.random() < 0.5:
        w, h = qr.size
        m = min(w, h) * random.uniform(0.0, 0.18)
        dst = [(random.uniform(0, m), random.uniform(0, m)),
               (w - random.uniform(0, m), random.uniform(0, m)),
               (w - random.uniform(0, m), h - random.uniform(0, m)),
               (random.uniform(0, m), h - random.uniform(0, m))]
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        try:
            coeffs = _perspective_coeffs(src, dst)
            qr = qr.transform((w, h), Image.PERSPECTIVE, coeffs,
                              resample=Image.BICUBIC)
        except np.linalg.LinAlgError:
            pass
    # rotation (mostly small; occasionally large)
    angle = random.uniform(-15, 15) if random.random() < 0.8 else random.choice(
        [random.uniform(-45, 45), 90, 180, 270])
    qr = qr.rotate(angle, expand=True, resample=Image.BICUBIC)
    return qr


# ----------------------------- compositing ------------------------------------

def degrade(img: Image.Image) -> Image.Image:
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.2)))
    if random.random() < 0.3:  # motion-ish blur
        img = img.filter(ImageFilter.BoxBlur(random.uniform(0.5, 2.0)))
    if random.random() < 0.8:  # contrast (push toward low)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.45, 1.15))
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.4))
    if random.random() < 0.4:
        arr = np.asarray(img).astype(np.int16)
        arr += np.random.normal(0, random.uniform(3, 18), arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def make_sample(min_size: int, max_size: int):
    w = random.randint(min_size, max_size)
    h = random.randint(min_size, max_size)
    bg = random_background(w, h).convert("RGB")
    boxes = []

    n_qr = 0 if random.random() < 0.08 else (2 if random.random() < 0.12 else 1)
    for _ in range(n_qr):
        qr = warp_qr(make_qr_image())
        # target size: QR short side between 4% and 38% of image min dim (small!)
        frac = random.uniform(0.04, 0.38)
        target = max(16, int(min(w, h) * frac))
        scale = target / max(qr.size)
        qr = qr.resize((max(8, int(qr.width * scale)),
                        max(8, int(qr.height * scale))), Image.BICUBIC)
        if qr.width >= w or qr.height >= h:
            continue
        ox = random.randint(0, w - qr.width)
        oy = random.randint(0, h - qr.height)
        bg.paste(qr, (ox, oy), qr)
        bb = qr.getbbox()  # tight bbox of non-transparent pixels
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        boxes.append([ox + x1, oy + y1, ox + x2, oy + y2])

    img = degrade(bg)
    return img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", type=str)
    ap.add_argument("--train", default=8000, type=int)
    ap.add_argument("--val", default=500, type=int)
    ap.add_argument("--min-size", default=700, type=int)
    ap.add_argument("--max-size", default=1800, type=int)
    ap.add_argument("--jpeg-min", default=45, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--debug-grid", action="store_true",
                    help="also write data/debug_grid.jpg with drawn boxes")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)

    debug_samples = []
    for split, count in (("train", args.train), ("val", args.val)):
        img_dir = out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        labels = []
        for i in range(count):
            img, boxes = make_sample(args.min_size, args.max_size)
            fname = f"images/{split}/{i:06d}.jpg"
            img.save(out / fname, "JPEG",
                     quality=random.randint(args.jpeg_min, 95))
            labels.append({"file": fname, "boxes": boxes})
            if args.debug_grid and split == "train" and len(debug_samples) < 16:
                debug_samples.append((img.copy(), boxes))
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{count}")
        with open(out / f"labels_{split}.json", "w") as f:
            json.dump(labels, f)
        print(f"wrote {count} {split} images + labels_{split}.json")

    if args.debug_grid and debug_samples:
        cell = 256
        cols = 4
        rows = (len(debug_samples) + cols - 1) // cols
        grid = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
        for idx, (im, bxs) in enumerate(debug_samples):
            im = im.convert("RGB")
            sx, sy = cell / im.width, cell / im.height
            im2 = im.resize((cell, cell))
            d = ImageDraw.Draw(im2)
            for (x1, y1, x2, y2) in bxs:
                d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                            outline=(0, 255, 0), width=3)
            grid.paste(im2, ((idx % cols) * cell, (idx // cols) * cell))
        grid.save(out / "debug_grid.jpg", "JPEG", quality=90)
        print(f"wrote {out / 'debug_grid.jpg'}")


if __name__ == "__main__":
    main()
