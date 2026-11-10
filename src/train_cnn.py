"""Train the CNN (SRNet-lite) or a transfer-learning baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import ROOT, ensure_dirs, load_config, resolve
from .data.dataset import build_dataloaders, load_manifest
from .models.cnn import n_params
from .models.factory import build_cnn, is_transfer
from .utils import (
    compute_metrics,
    ensure_dir,
    format_metrics,
    get_logger,
    plot_confusion,
    plot_history,
    plot_roc,
    save_json,
    set_seed,
)

log = get_logger()


def apply_limit(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n:
        return df
    out = []
    for _, part in df.groupby(["split", "label"]):
        out.append(part.sample(min(n, len(part)), random_state=seed))
    return pd.concat(out).reset_index(drop=True)


def forward(model: nn.Module, x: torch.Tensor, arch: str) -> torch.Tensor:
    if is_transfer(arch) and x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return model(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, arch: str, criterion: nn.Module):
    model.eval()
    losses, probs, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = forward(model, x, arch)
        loss = criterion(logits, y)
        losses.append(loss.item() * len(y))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.float().cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    loss = float(np.sum(losses) / max(len(labels), 1))
    return loss, probs, labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CNN LSB steganalysis detector")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="artifacts/cnn")
    ap.add_argument("--arch", default=None, choices=["srnet_lite", "efficientnet_v2_s", "resnet18"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--in-channels", type=int, default=None, choices=[1, 3])
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    set_seed(seed)
    cnn = cfg["cnn"]
    arch = args.arch or cnn["arch"]
    img_size = args.img_size or int(cnn["image_size"])
    epochs = args.epochs or int(cnn["epochs"])
    batch_size = args.batch_size or int(cnn["batch_size"])
    lr = args.lr or float(cnn["learning_rate"])
    num_workers = args.num_workers if args.num_workers is not None else int(cnn["num_workers"])
    patience = args.patience if args.patience is not None else int(cnn["patience"])
    amp = (not args.no_amp) and bool(cnn["amp"])
    in_channels = args.in_channels or int(cnn.get("in_channels", 1))
    if is_transfer(arch):
        in_channels = 3

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.warn("CUDA not available -> training on CPU will be slow")

    manifest = resolve(args.manifest or cfg["paths"]["manifest"])
    df = load_manifest(manifest)
    limit = args.limit_per_class if args.limit_per_class is not None else cfg["data"]["limit_per_class"]
    if limit:
        df = apply_limit(df, limit, seed)
        tmp = resolve("data/manifest_limit.csv")
        df.drop(columns=["abspath"]).to_csv(tmp, index=False)
        manifest = tmp

    loaders = build_dataloaders(manifest, size=img_size, batch_size=batch_size,
                                num_workers=num_workers, in_channels=in_channels)
    if loaders["train"] is None or loaders["val"] is None:
        raise SystemExit("manifest must contain non-empty train and val splits")
    log.info("batches: train=%d val=%d test=%s", len(loaders["train"]), len(loaders["val"]),
             len(loaders["test"]) if loaders["test"] is not None else "none")

    model = build_cnn(arch, pretrained=not args.no_pretrained,
                      in_channels=in_channels).to(device)
    log.info("arch=%s params=%s", arch, f"{n_params(model):,}")

    train_labels = load_manifest(manifest)
    train_labels = train_labels[train_labels["split"] == "train"]["label"].to_numpy()
    pos = float(train_labels.sum())
    neg = float(len(train_labels) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(cnn["weight_decay"]))
    total_steps = max(len(loaders["train"]) * epochs, 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = bool(amp and device.type == "cuda")

    class _NoScaler:
        def scale(self, x):
            return x

        def step(self, opt):
            opt.step()

        def update(self):
            pass

    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else _NoScaler()

    out = ensure_dir(resolve(args.out))
    best_auc, best_loss, bad = -1.0, float("inf"), 0
    history: list[dict] = []
    best_path = out / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                logits = forward(model, x, arch)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
        train_loss = running / len(loaders["train"].dataset)

        val_loss, val_prob, val_y = evaluate(model, loaders["val"], device, arch, criterion)
        vm = compute_metrics(val_y, val_prob, 0.5, cfg["eval"]["target_tpr"])
        val_auc = vm["roc_auc"] if vm["roc_auc"] is not None else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_auc": val_auc, "val_acc": vm["accuracy"], "lr": scheduler.get_last_lr()[0]})
        log.info("epoch %d | train_loss=%.4f val_loss=%.4f val_auc=%.4f acc=%.4f (%.1fs)",
                 epoch, train_loss, val_loss, val_auc, vm["accuracy"], time.time() - t0)

        improved = (val_auc > best_auc) if val_auc > 0 else (val_loss < best_loss)
        if improved:
            best_auc = max(best_auc, val_auc)
            best_loss = min(best_loss, val_loss)
            bad = 0
            torch.save({
                "arch": arch,
                "state_dict": model.state_dict(),
                "image_size": img_size,
                "in_channels": in_channels,
                "epoch": epoch,
                "val_auc": val_auc,
                "val_loss": val_loss,
            }, best_path)
        else:
            bad += 1
            if bad >= patience:
                log.info("early stopping at epoch %d", epoch)
                break

    save_json(history, out / "history.json")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_history(history, out / "history.png")

    meta = {
        "kind": "cnn",
        "arch": arch,
        "image_size": img_size,
        "in_channels": in_channels,
        "threshold": float(cfg["eval"]["threshold"]),
        "needs_3ch": is_transfer(arch),
        "n_params": int(n_params(model)),
        "best_val_auc": float(best_auc),
    }
    save_json(meta, out / "meta.json")

    # Final evaluation with the best checkpoint.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    eval_loaders = {k: v for k, v in loaders.items() if v is not None and k != "train"}
    for split, loader in eval_loaders.items():
        loss, prob, y = evaluate(model, loader, device, arch, criterion)
        metrics = compute_metrics(y, prob, cfg["eval"]["threshold"], cfg["eval"]["target_tpr"])
        metrics["loss"] = loss
        save_json(metrics, out / f"metrics_{split}.json")
        plot_confusion(metrics, out / f"confusion_{split}.png", f"{arch} {split}")
        plot_roc(metrics, out / f"roc_{split}.png", f"{arch} {split}")
        log.info("%s metrics: %s", split, format_metrics(metrics))
    log.info("saved CNN artifacts -> %s", out)


if __name__ == "__main__":
    main()
