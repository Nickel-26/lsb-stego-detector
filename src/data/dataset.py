"""PyTorch dataset / dataloaders built from the manifest.

Only ``torch``-dependent code lives here; import this module only when training
or running the CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import ROOT, load_config, resolve
from ..imageio import center_crop_or_pad, load_image


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = resolve(path or cfg["paths"]["manifest"])
    if not Path(path).exists():
        raise FileNotFoundError(f"manifest not found: {path}. Run src.data.download_check first.")
    df = pd.read_csv(path)
    df["abspath"] = df["path"].map(lambda p: str(resolve(p)))
    return df


def _to_tensor(img: np.ndarray, size: int, train: bool, augment: bool) -> torch.Tensor:
    if train and augment:
        h, w = img.shape[:2]
        if h >= size and w >= size:
            top = np.random.randint(0, h - size + 1)
            left = np.random.randint(0, w - size + 1)
            img = img[top:top + size, left:left + size]
        else:
            img = center_crop_or_pad(img, size)
        if np.random.random() < 0.5:
            img = img[:, ::-1]
    else:
        img = center_crop_or_pad(img, size)
    x = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    if x.ndim == 3:                    # (H, W, C) -> (C, H, W)
        x = np.transpose(x, (2, 0, 1))
    else:                              # (H, W) -> (1, H, W)
        x = x[None, ...]
    return torch.from_numpy(x)


class StegoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = False, size: int = 256,
                 augment: bool = True, in_channels: int = 1) -> None:
        self.paths = df["abspath"].tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        self.train = train
        self.size = size
        self.augment = augment
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        mode = "rgb" if self.in_channels == 3 else "gray"
        img = load_image(self.paths[idx], mode=mode)
        x = _to_tensor(img, self.size, self.train, self.augment)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def build_dataloaders(manifest: str | Path | None = None, size: int | None = None,
                      batch_size: int | None = None, num_workers: int | None = None,
                      augment: bool | None = None, in_channels: int | None = None):
    cfg = load_config()
    cnn = cfg["cnn"]
    size = size or int(cnn["image_size"])
    batch_size = batch_size or int(cnn["batch_size"])
    num_workers = num_workers if num_workers is not None else int(cnn["num_workers"])
    augment = augment if augment is not None else bool(cnn["augment"])
    in_channels = in_channels if in_channels is not None else int(cnn.get("in_channels", 1))

    df = load_manifest(manifest)
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        if part.empty:
            loaders[split] = None
            continue
        ds = StegoDataset(part, train=(split == "train"), size=size, augment=augment,
                          in_channels=in_channels)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False, persistent_workers=num_workers > 0,
        )
    return loaders


def class_balance(df: pd.DataFrame, split: str = "train") -> dict[int, int]:
    part = df[df["split"] == split]
    return {int(k): int(v) for k, v in part["label"].value_counts().sort_index().items()}
