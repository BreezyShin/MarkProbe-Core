from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, image_size=128, limit=None):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in self.extensions)
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError("No supported images found under: {}".format(root))
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = ImageOps.fit(
            Image.open(self.paths[index]).convert("RGB"),
            (self.image_size, self.image_size),
            method=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class SyntheticImages(Dataset):
    def __init__(self, length=64, image_size=64, seed=0):
        self.length, self.image_size, self.seed = length, image_size, seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        image = rng.random((self.image_size, self.image_size, 3), dtype=np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)


def random_bits(batch, bit_count, device):
    return torch.randint(0, 2, (batch, bit_count), device=device).float()
