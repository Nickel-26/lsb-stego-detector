"""Shared inference API + CLI for the LSB steganalysis detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve
from .features.extract import extract_features, extract_features_color
from .features.lsb import lsb_features
from .features.statistical import statistical_features
from .imageio import center_crop_or_pad, load_image
from .utils import get_logger

log = get_logger()

CLASSES = {0: "clean", 1: "stego"}


# --------------------------------------------------------------------------- #
# Artifact loading (cached in-process)
# --------------------------------------------------------------------------- #
_CACHE: dict[str, Any] = {}


def _classical_paths(artifacts_dir: str | Path | None) -> Path:
    cfg = load_config()
    return resolve(artifacts_dir or Path(cfg["paths"]["artifacts_dir"]) / "classical")


def _cnn_paths(artifacts_dir: str | Path | None) -> Path:
    cfg = load_config()
    return resolve(artifacts_dir or Path(cfg["paths"]["artifacts_dir"]) / "cnn")


def load_classical(artifacts_dir: str | Path | None = None):
    from .models import classical as C
    from .utils import load_json

    root = _classical_paths(artifacts_dir)
    key = f"classical:{root}"
    if key not in _CACHE:
        if not (root / "model.joblib").exists():
            raise FileNotFoundError(f"classical model not found in {root}")
        model, scaler = C.load(root)
        meta = load_json(root / "meta.json") if (root / "meta.json").exists() else {}
        _CACHE[key] = (model, scaler, meta)
    return _CACHE[key]


def load_cnn(artifacts_dir: str | Path | None = None, device: str | None = None):
    import torch

    from .models.factory import build_cnn
    from .utils import load_json

    root = _cnn_paths(artifacts_dir)
    key = f"cnn:{root}:{device}"
    if key not in _CACHE:
        ckpt_path = root / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"CNN checkpoint not found in {root}")
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        meta = load_json(root / "meta.json") if (root / "meta.json").exists() else {}
        meta.setdefault("arch", ckpt["arch"])
        meta.setdefault("image_size", ckpt.get("image_size", 256))
        meta.setdefault("in_channels", ckpt.get("in_channels", 1))
        model = build_cnn(ckpt["arch"], pretrained=False,
                          in_channels=int(meta["in_channels"])).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _CACHE[key] = (model, meta, dev)
    return _CACHE[key]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score_classical(gray: np.ndarray, rgb: np.ndarray | None, artifacts_dir) -> float:
    from .models import classical as C

    model, scaler, meta = load_classical(artifacts_dir)
    if meta.get("color"):
        image = rgb if rgb is not None else np.stack([gray, gray, gray], axis=-1)
        feats = extract_features_color(image, int(meta.get("spam_t", 3)),
                                       int(meta.get("srm_t", 2)))
    else:
        feats = extract_features(gray, int(meta.get("spam_t", 3)), int(meta.get("srm_t", 2)))
    return float(C.predict_proba(model, scaler, feats["vector"][None, :])[0])


def _score_cnn(gray: np.ndarray, rgb: np.ndarray | None, artifacts_dir, device) -> float:
    import torch

    from .models.factory import is_transfer

    model, meta, dev = load_cnn(artifacts_dir, device)
    size = int(meta["image_size"])
    arch = meta["arch"]
    in_channels = int(meta.get("in_channels", 1))
    if in_channels == 3:
        img = rgb if rgb is not None else np.stack([gray, gray, gray], axis=-1)
    else:
        img = gray if gray is not None else (rgb[:, :, 0] if rgb is not None and rgb.ndim == 3 else rgb)
    img = center_crop_or_pad(img, size)
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1) if x.ndim == 3 else x.unsqueeze(0)
    x = x.unsqueeze(0).to(dev)
    if is_transfer(arch) and x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    with torch.no_grad():
        logit = model(x)
    return float(torch.sigmoid(logit).item())


def predict_gray(gray: np.ndarray, model: str = "ensemble", threshold: float = 0.5,
                 classical_dir: str | Path | None = None, cnn_dir: str | Path | None = None,
                 device: str | None = None, rgb: np.ndarray | None = None) -> dict[str, Any]:
    """Predict from a uint8 array (gray or, ideally, the original RGB image)."""
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    want = {"classical", "cnn"} if model == "ensemble" else {model}
    if "classical" in want:
        try:
            scores["classical"] = _score_classical(gray, rgb, classical_dir)
        except Exception as exc:  # keep the other model usable
            errors["classical"] = str(exc)
    if "cnn" in want:
        try:
            scores["cnn"] = _score_cnn(gray, rgb, cnn_dir, device)
        except Exception as exc:
            errors["cnn"] = str(exc)

    if not scores:
        raise RuntimeError(f"no model could be run: {errors}")
    prob = float(np.mean(list(scores.values())))
    is_stego = prob >= threshold
    display = rgb[:, :, 1] if rgb is not None and rgb.ndim == 3 else gray
    return {
        "label": CLASSES[int(is_stego)],
        "is_stego": bool(is_stego),
        "probability": prob,
        "threshold": float(threshold),
        "per_model": scores,
        "errors": errors,
        "statistical": statistical_features(display),
        "lsb": lsb_features(display),
    }


def predict_image(path: str | Path, model: str = "ensemble", threshold: float = 0.5,
                  classical_dir=None, cnn_dir=None, device: str | None = None) -> dict[str, Any]:
    gray = load_image(path, mode="gray")
    rgb = load_image(path, mode="rgb")
    result = predict_gray(gray, model, threshold, classical_dir, cnn_dir, device, rgb=rgb)
    result["path"] = str(path)
    return result


def predict_paths(paths, **kwargs) -> list[dict[str, Any]]:
    return [predict_image(p, **kwargs) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect LSB steganography in images")
    ap.add_argument("image", nargs="+", help="image path(s)")
    ap.add_argument("--model", default="ensemble", choices=["classical", "cnn", "ensemble"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--classical-dir", default=None)
    ap.add_argument("--cnn-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()

    cfg = load_config()
    threshold = args.threshold if args.threshold is not None else float(cfg["eval"]["threshold"])
    results = []
    for path in args.image:
        res = predict_image(path, args.model, threshold, args.classical_dir,
                            args.cnn_dir, args.device)
        results.append(res)
        if not args.json:
            print(f"{path}: {res['label'].upper()}  p(stego)={res['probability']:.4f} "
                  f"(threshold {threshold:.2f})  per_model={res['per_model']}")
    if args.json:
        print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
