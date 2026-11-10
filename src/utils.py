"""Shared utilities: IO, seeding, metrics and plotting."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER_NAME = "stegdet"


def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional torch
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5,
                    target_tpr: float = 0.95) -> dict[str, Any]:
    """Binary metrics. Positive class (1) = stego."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    # AUC / thresholds need both classes present.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        # smallest threshold achieving target TPR -> corresponding FPR
        idx = int(np.searchsorted(tpr, target_tpr))
        idx = min(idx, len(fpr) - 1)
        metrics[f"fpr_at_tpr_{target_tpr:.2f}"] = float(fpr[idx])
        metrics["threshold_at_target_tpr"] = float(thr[idx])
        metrics["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        metrics["roc_auc"] = None
    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    parts = [f"{k}={metrics.get(k):.4f}" if isinstance(metrics.get(k), float) else f"{k}={metrics.get(k)}" for k in keys]
    fpr_key = next((k for k in metrics if k.startswith("fpr_at_tpr_")), None)
    if fpr_key:
        parts.append(f"{fpr_key}={metrics[fpr_key]:.4f}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Plotting (matplotlib imported lazily)
# --------------------------------------------------------------------------- #
def plot_confusion(metrics: dict[str, Any], path: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["clean", "stego"])
    ax.set_yticks([0, 1], labels=["clean", "stego"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc(metrics: dict[str, Any], path: str | Path, title: str = "") -> None:
    if "roc" not in metrics:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr = np.asarray(metrics["roc"]["fpr"])
    tpr = np.asarray(metrics["roc"]["tpr"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={metrics.get('roc_auc', float('nan')):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or "ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_auc"] for h in history], label="val AUC")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
