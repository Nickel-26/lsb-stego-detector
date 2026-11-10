"""Standalone evaluation for a saved classical or CNN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .utils import (
    compute_metrics,
    ensure_dir,
    format_metrics,
    get_logger,
    load_json,
    plot_confusion,
    plot_roc,
    save_json,
)

log = get_logger()


def _eval_classical(args, cfg):
    from .models import classical as C
    from .train_classical import extract_split

    out = resolve(args.artifacts)
    model, scaler = C.load(out)
    meta = load_json(out / "meta.json")
    manifest = resolve(args.manifest or cfg["paths"]["manifest"])
    df = pd.read_csv(manifest)
    split = args.split
    X, y, _ = extract_split(df, split, meta["spam_t"], meta["srm_t"],
                            int(cfg["classical"]["n_jobs"]),
                            resolve(cfg["paths"]["features_dir"]),
                            color=bool(meta.get("color", False)))
    prob = C.predict_proba(model, scaler, X)
    return y, prob, meta


def _eval_cnn(args, cfg):
    import torch

    from .data.dataset import build_dataloaders
    from .models.factory import build_cnn
    from .train_cnn import evaluate as eval_loader

    out = resolve(args.artifacts)
    meta = load_json(out / "meta.json")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    in_channels = int(ckpt.get("in_channels", meta.get("in_channels", 1)))
    model = build_cnn(ckpt["arch"], pretrained=False,
                      in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loaders = build_dataloaders(args.manifest, size=int(meta["image_size"]),
                                batch_size=int(cfg["cnn"]["batch_size"]),
                                num_workers=int(cfg["cnn"]["num_workers"]),
                                in_channels=in_channels)
    loader = loaders[args.split]
    if loader is None:
        raise SystemExit(f"split '{args.split}' is empty")
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss()
    loss, prob, y = eval_loader(model, loader, device, ckpt["arch"], criterion)
    meta["loss"] = loss
    return y, prob, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a saved model")
    ap.add_argument("--model", required=True, choices=["classical", "cnn"])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config()
    threshold = args.threshold if args.threshold is not None else float(cfg["eval"]["threshold"])
    if args.model == "classical":
        y, prob, meta = _eval_classical(args, cfg)
    else:
        y, prob, meta = _eval_cnn(args, cfg)

    metrics = compute_metrics(y, prob, threshold, cfg["eval"]["target_tpr"])
    out = ensure_dir(resolve(args.artifacts))
    save_json(metrics, out / f"metrics_{args.split}.json")
    plot_confusion(metrics, out / f"confusion_{args.split}.png", f"{args.model} {args.split}")
    plot_roc(metrics, out / f"roc_{args.split}.png", f"{args.model} {args.split}")
    log.info("%s/%s: %s", args.model, args.split, format_metrics(metrics))


if __name__ == "__main__":
    main()
