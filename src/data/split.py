"""Group-aware (leakage-free) split assignment and leakage checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import resolve
from ..utils import get_logger

log = get_logger()


def regroup(df: pd.DataFrame, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
            seed: int = 1337) -> pd.DataFrame:
    """Reassign train/val/test by whole ``group_id`` so no source image leaks.

    Classes are split independently to preserve balance.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    # Assign whole groups (cover + its stego copy) to a single split.
    groups = np.array(sorted(out["group_id"].unique()))
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    mapping = {g: "train" for g in groups[:n_train]}
    mapping.update({g: "val" for g in groups[n_train:n_train + n_val]})
    mapping.update({g: "test" for g in groups[n_train + n_val:]})
    out["split"] = out["group_id"].map(mapping)
    return out


def leakage_summary(df: pd.DataFrame) -> dict:
    per_group = df.groupby("group_id")["split"].nunique()
    summary = {"groups_across_splits": int((per_group > 1).sum())}
    if df["md5"].astype(bool).any():
        per_hash = df[df["md5"].astype(bool)].groupby("md5")["split"].nunique()
        summary["hashes_across_splits"] = int((per_hash > 1).sum())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Regroup a manifest to avoid leakage")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()
    path = resolve(args.manifest)
    df = pd.read_csv(path)
    before = leakage_summary(df)
    new = regroup(df, (args.train, args.val, 1 - args.train - args.val), args.seed)
    after = leakage_summary(new)
    out = resolve(args.out) if args.out else path.with_name(path.stem + "_grouped.csv")
    new.to_csv(out, index=False)
    log.info("before=%s after=%s", before, after)
    log.info("wrote %s; counts=%s", out, new.groupby(["split", "cls"]).size().to_dict())


if __name__ == "__main__":
    main()
