"""Validate a manually downloaded dataset and build a manifest.

Expected layout (Stego-Images-Dataset):

    data/raw/
      train/{clean,stego}/
      val/{clean,stego}/
      test/{clean,stego}/

The manifest is a CSV with columns:
    path, label, split, cls, group_id, md5, size_bytes

``group_id`` links a cover and its stego counterpart so that group-aware
splitting cannot leak the same source image across train/test.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_config, resolve
from ..imageio import is_image
from ..utils import get_logger

log = get_logger()

SPLITS = ("train", "val", "test")
CLASSES = ("clean", "stego")
LABELS = {"clean": 0, "stego": 1}

_GROUP_STRIP = re.compile(
    r"(?:^|[_\-.])"
    r"(?:clean|cover|stego|steg|lsb|hidden|secret|original|"
    r"embed(?:ded)?|p\d+(?:\.\d+)?|bpp\d*(?:\.\d+)?)"
    r"(?:$|[_\-.])",
    re.IGNORECASE,
)

# HF "Multilingual LSB Stego Dataset" names: <cover>_hr_<ratio>_hb_<bits>_<policy>.
# Locally prepared variants use <cover>__v<n>.
_VARIANT_SUFFIX = re.compile(r"(?:_hr_.*|__v\d+)$", re.IGNORECASE)


def derive_group_id(stem: str) -> str:
    """Best-effort mapping from filename stem to a shared cover/stego group id."""
    stem = _VARIANT_SUFFIX.sub("", stem)
    g = _GROUP_STRIP.sub("_", stem).strip("_-.")
    return (g or stem).lower()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan_raw(raw_dir: Path, splits=SPLITS, classes=CLASSES,
             limit_per_class: int | None = None, quick: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for split in splits:
        for cls in classes:
            folder = raw_dir / split / cls
            if not folder.is_dir():
                log.warning("missing folder: %s", folder)
                continue
            files = sorted(p for p in folder.rglob("*") if p.is_file() and is_image(p))
            if limit_per_class is not None:
                files = files[:limit_per_class]
            log.info("%s/%s: %d images", split, cls, len(files))
            for p in files:
                stat = p.stat()
                try:
                    rel = str(p.relative_to(ROOT))
                except ValueError:  # outside repo root (e.g. temp dirs in tests)
                    rel = str(p)
                rows.append({
                    "path": rel,
                    "label": LABELS[cls],
                    "split": split,
                    "cls": cls,
                    "group_id": derive_group_id(p.stem),
                    "md5": "" if quick else _md5(p),
                    "size_bytes": stat.st_size,
                })
    return pd.DataFrame(rows, columns=["path", "label", "split", "cls",
                                       "group_id", "md5", "size_bytes"])


def leakage_report(df: pd.DataFrame) -> dict:
    report: dict = {}
    if df.empty:
        return {"empty": True}
    # Identical content appearing in more than one split.
    if df["md5"].astype(bool).any():
        per_hash = df[df["md5"].astype(bool)].groupby("md5")["split"].nunique()
        report["duplicate_content_across_splits"] = int((per_hash > 1).sum())
        report["duplicate_rows"] = int(df[df["md5"].astype(bool)].duplicated("md5").sum())
    # Same group id appearing in more than one split.
    per_group = df.groupby("group_id")["split"].nunique()
    report["groups_across_splits"] = int((per_group > 1).sum())
    report["n_groups"] = int(df["group_id"].nunique())
    report["counts"] = df.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict()
    return report


def build_manifest(raw_dir: str | Path | None = None, out: str | Path | None = None,
                   limit_per_class: int | None = None, quick: bool = False) -> pd.DataFrame:
    cfg = load_config()
    raw_dir = resolve(raw_dir or cfg["paths"]["raw_dir"])
    out = resolve(out or cfg["paths"]["manifest"])
    df = scan_raw(raw_dir, limit_per_class=limit_per_class, quick=quick)
    if df.empty:
        raise SystemExit(
            f"No images found under {raw_dir}.\n"
            f"Expected {raw_dir}/{{train,val,test}}/{{clean,stego}}/*.png"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    report = leakage_report(df)
    log.info("wrote manifest: %s (%d rows)", out, len(df))
    log.info("counts: %s", df.groupby(["split", "cls"]).size().to_dict())
    log.info("leakage report: %s", {k: v for k, v in report.items() if k != "counts"})
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate dataset and build manifest")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="skip content hashing")
    args = ap.parse_args()
    build_manifest(args.raw_dir, args.out, args.limit_per_class, args.quick)


if __name__ == "__main__":
    main()
