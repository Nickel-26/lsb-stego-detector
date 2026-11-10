"""Create a tiny synthetic dataset to smoke-test the whole pipeline.

Produces cover/stego pairs in the same layout as the real dataset:
    data/raw/{split}/{clean,stego}/img_00000.png

Cover LSBs are correlated with higher bits (cover-like); stego LSBs are
randomised over a fraction of pixels (payload). Use this to verify code paths
before the real data is downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_cover(rng: np.random.Generator, size: int) -> np.ndarray:
    coarse = rng.normal(0, 1, (max(size // 16, 2), max(size // 16, 2)))
    base = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-8) * 200 + 20
    texture = rng.normal(0, 4, (size, size))
    img = np.clip(base + texture, 0, 255).astype(np.uint8)
    # Correlate LSB with bit 1 -> cover-like even/odd histogram asymmetry.
    return ((img & 0xFE) | ((img >> 1) & 1)).astype(np.uint8)


def make_stego(cover: np.ndarray, rng: np.random.Generator, payload: float) -> np.ndarray:
    out = cover.copy()
    mask = rng.random(cover.shape) < payload
    out[mask] = (out[mask] & 0xFE) | rng.integers(0, 2, int(mask.sum())).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--payload", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        for cls in ("clean", "stego"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cover = make_cover(rng, args.size)
            stego = make_stego(cover, rng, args.payload)
            name = f"{split}_img_{i:05d}.png"
            cv2.imwrite(str(root / split / "clean" / name), cover)
            cv2.imwrite(str(root / split / "stego" / name), stego)
        print(f"{split}: {n} cover/stego pairs under {root / split}")
    print("done. next: python -m src.data.download_check")


if __name__ == "__main__":
    main()
