"""Lossless image loading helpers.

LSB steganography lives in the least-significant bits, so images must be read
without any lossy conversion. We never resize inside feature extraction; the
CNN applies random *crops* (not resize) during training.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".pgm", ".ppm", ".webp"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


def load_image(path: str | Path, mode: str = "gray") -> np.ndarray:
    """Load an image as a numpy array.

    mode="gray" -> uint8 (H, W); mode="rgb" -> uint8 (H, W, 3) in RGB order.
    Uses OpenCV first (fast), falls back to Pillow.
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.array(Image.open(path))
    if img.ndim == 2:
        gray = img
    else:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 3:
            pass
        else:  # single-channel but 3-d
            img = img[:, :, 0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if mode == "gray":
        return np.ascontiguousarray(gray.astype(np.uint8))
    if mode in ("rgb", "color"):
        if img.ndim == 2:
            rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)
    raise ValueError(f"unknown mode: {mode}")


def center_crop_or_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Deterministic center crop (or zero-pad) to size x size."""
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        if img.ndim == 3:
            pad += ((0, 0),)
        img = np.pad(img, pad, mode="edge")
        h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]
