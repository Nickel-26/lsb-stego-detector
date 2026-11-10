from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_cover(rng: np.random.Generator, size: int) -> np.ndarray:
    coarse = rng.normal(0, 1, (max(size // 16, 2), max(size // 16, 2)))
    base = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-8) * 200 + 20
    img = np.clip(base + rng.normal(0, 4, (size, size)), 0, 255).astype(np.uint8)
    return ((img & 0xFE) | ((img >> 1) & 1)).astype(np.uint8)


def _make_stego(cover: np.ndarray, rng: np.random.Generator, payload: float = 0.5) -> np.ndarray:
    out = cover.copy()
    mask = rng.random(cover.shape) < payload
    out[mask] = (out[mask] & 0xFE) | rng.integers(0, 2, int(mask.sum())).astype(np.uint8)
    return out


@pytest.fixture(scope="session")
def paired_images():
    rng = np.random.default_rng(42)
    cover = _make_cover(rng, 128)
    stego = _make_stego(cover, rng, 0.5)
    return cover, stego


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Create a small on-disk dataset and return (manifest_path, raw_dir, manifest_df)."""
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(7)
    counts = {"train": 40, "val": 12, "test": 12}
    size = 128
    for split, n in counts.items():
        for cls in ("clean", "stego"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cover = _make_cover(rng, size)
            stego = _make_stego(cover, rng, 0.5)
            name = f"{split}_img_{i:05d}.png"
            cv2.imwrite(str(root / split / "clean" / name), cover)
            cv2.imwrite(str(root / split / "stego" / name), stego)

    from src.data.download_check import build_manifest

    out = root / "manifest.csv"
    df = build_manifest(raw_dir=root, out=out, quick=True)
    return out, root, df
