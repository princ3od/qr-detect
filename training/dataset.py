"""torchvision-style detection dataset reading the generator's labels_*.json."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F


class QrDataset(torch.utils.data.Dataset):
    """Single-class ("QR" = label 1) detection dataset.

    Returns (image_tensor[0..1, CHW, RGB], target) where target has the keys
    torchvision detection models expect: boxes [N,4] xyxy, labels [N].
    """

    def __init__(self, root: str, split: str):
        self.root = Path(root)
        with open(self.root / f"labels_{split}.json") as f:
            self.items = json.load(f)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = Image.open(self.root / item["file"]).convert("RGB")
        boxes = item["boxes"]
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        target = {
            "boxes": boxes_t,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return F.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))
