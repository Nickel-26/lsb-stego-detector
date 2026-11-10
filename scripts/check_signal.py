"""Sanity-check that statistical attacks separate cover from stego.

If the chi-square / RS features cannot separate the classes on your dataset,
something is wrong with the labels or the embedding. Run this before training.

    python scripts/check_signal.py --split train --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve
from src.features.statistical import statistical_features
from src.imageio import load_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=300, help="samples per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    manifest = resolve(args.manifest or cfg["paths"]["manifest"])
    df = pd.read_csv(manifest)
    df = df[df["split"] == args.split]
    if df.empty:
        raise SystemExit(f"split '{args.split}' empty in {manifest}")

    parts = []
    for label, part in df.groupby("label"):
        parts.append(part.sample(min(args.n, len(part)), random_state=args.seed))
    sample = pd.concat(parts)

    rows = []
    for _, row in sample.iterrows():
        gray = load_image(resolve(row["path"]), mode="gray")
        feats = statistical_features(gray)
        feats["label"] = int(row["label"])
        rows.append(feats)
    data = pd.DataFrame(rows)

    from sklearn.metrics import roc_auc_score

    print(f"\nsamples: {data.groupby('label').size().to_dict()}\n")
    print(f"{'feature':28s} {'clean mean':>12s} {'stego mean':>12s} {'AUC':>7s}")
    print("-" * 64)
    for col in data.columns:
        if col == "label":
            continue
        y = data["label"].to_numpy()
        x = data[col].to_numpy()
        try:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)  # direction-agnostic
        except ValueError:
            auc = float("nan")
        print(f"{col:28s} {data.loc[y == 0, col].mean():12.4f} "
              f"{data.loc[y == 1, col].mean():12.4f} {auc:7.4f}")
    print("\nAUC ~0.5 = no separation; >0.8 = strong signal (expected for LSB).")


if __name__ == "__main__":
    main()
