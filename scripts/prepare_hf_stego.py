"""Prepare a compact cover/stego dataset from the HF Multilingual LSB Stego Dataset.

The HuggingFace repo ``emirozturk/LSBStegoDataset`` ships one tar per
resolution/language.  The base ``<res>.tar`` holds the original covers, while
``<res>_<lang>.tar`` holds stego variants named::

    <cover>_hr_<ratio>_hb_<bits>_<policy>.png

This script pairs each cover with a few randomly chosen stego variants and
materialises the ``data/raw/{train,val,test}/{clean,stego}`` layout expected by
``python -m src.data.download_check``.  Splits are assigned per cover so a
cover and all of its stego copies always stay together (no leakage).

Example::

    tar -xf 128x128.tar      -C /tmp/stego_hf/covers
    tar -xf 128x128_en.tar   -C /tmp/stego_hf/en
    python scripts/prepare_hf_stego.py \
        --covers-dir /tmp/stego_hf/covers/128x128 \
        --stego-dir  /tmp/stego_hf/en/128x128_en \
        --out data/raw --variants 2 --max-covers 1000
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
STEGO_RE = re.compile(r"^(?P<gid>.+?)_hr_", re.IGNORECASE)
SPLITS = ("train", "val", "test")
CLASSES = ("clean", "stego")


def _images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pair HF covers with LSB stego variants")
    ap.add_argument("--covers-dir", required=True, help="extracted <res>.tar directory")
    ap.add_argument("--stego-dir", required=True, help="extracted <res>_<lang>.tar directory")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--variants", type=int, default=2, help="stego variants per cover")
    ap.add_argument("--max-covers", type=int, default=None)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    covers_dir = Path(args.covers_dir)
    stego_dir = Path(args.stego_dir)
    out = Path(args.out)
    if not covers_dir.is_dir() or not stego_dir.is_dir():
        raise SystemExit(f"missing input dir(s): {covers_dir} / {stego_dir}")

    covers: dict[str, Path] = {}
    for p in _images(covers_dir):
        covers.setdefault(p.stem, p)
    if not covers:
        raise SystemExit(f"no cover images under {covers_dir}")

    variants: dict[str, list[Path]] = defaultdict(list)
    for p in _images(stego_dir):
        m = STEGO_RE.match(p.stem)
        if m and m.group("gid") in covers:
            variants[m.group("gid")].append(p)

    gids = sorted(g for g in covers if variants.get(g))
    if not gids:
        raise SystemExit("no stego files matched cover ids; check naming / directories")

    rng = random.Random(args.seed)
    rng.shuffle(gids)
    if args.max_covers:
        gids = gids[: args.max_covers]

    n = len(gids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    split_of = {
        g: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, g in enumerate(gids)
    }

    if out.exists() and any(out.glob("*/*/*")) and not args.overwrite:
        raise SystemExit(f"{out} already contains files; pass --overwrite")
    if args.overwrite:
        for split in SPLITS:
            for cls in CLASSES:
                d = out / split / cls
                if d.exists():
                    shutil.rmtree(d)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for g in gids:
        split = split_of[g]
        (out / split / "clean").mkdir(parents=True, exist_ok=True)
        (out / split / "stego").mkdir(parents=True, exist_ok=True)
        cover = covers[g]
        shutil.copy2(cover, out / split / "clean" / f"{g}{cover.suffix.lower()}")
        counts[(split, "clean")] += 1
        picks = rng.sample(variants[g], k=min(args.variants, len(variants[g])))
        for i, stego in enumerate(picks):
            shutil.copy2(stego, out / split / "stego" / f"{g}__v{i}.png")
            counts[(split, "stego")] += 1

    print(f"wrote dataset -> {out.resolve()}")
    for split in SPLITS:
        print(f"  {split}: clean={counts[(split, 'clean')]} stego={counts[(split, 'stego')]}")
    print(f"  groups: {n} (train={n_train} val={n_val} test={n - n_train - n_val}), "
          f"variants/cover={args.variants}")


if __name__ == "__main__":
    main()
