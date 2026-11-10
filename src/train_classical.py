"""Train the classical (SPAM+SRM+LSB -> XGBoost/SVM) detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ensure_dirs, load_config, resolve
from .features.extract import (
    extract_features,
    extract_features_color,
    feature_names,
    feature_names_color,
)
from .imageio import load_image
from .models import classical as C
from .utils import (
    compute_metrics,
    ensure_dir,
    format_metrics,
    get_logger,
    plot_confusion,
    plot_roc,
    save_json,
    set_seed,
)

log = get_logger()


def _extract_one(args: tuple[str, int, int, bool]) -> np.ndarray:
    path, spam_t, srm_t, color = args
    if color:
        rgb = load_image(path, mode="rgb")
        return extract_features_color(rgb, spam_t, srm_t)["vector"]
    gray = load_image(path, mode="gray")
    return extract_features(gray, spam_t, srm_t)["vector"]


def apply_limit(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n:
        return df
    out = []
    for _, part in df.groupby(["split", "label"]):
        out.append(part.sample(min(n, len(part)), random_state=seed))
    return pd.concat(out).reset_index(drop=True)


def extract_split(df: pd.DataFrame, split: str, spam_t: int, srm_t: int,
                  n_jobs: int, cache_dir: Path, use_cache: bool = True,
                  color: bool = False):
    part = df[df["split"] == split].reset_index(drop=True)
    if part.empty:
        raise SystemExit(f"split '{split}' is empty")
    paths = [str(resolve(p)) for p in part["path"]]
    y = part["label"].to_numpy(np.int64)
    suffix = "_color" if color else ""
    cache = cache_dir / f"{split}_spam{spam_t}_srm{srm_t}{suffix}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        same = list(z["paths"]) == paths
        if same and z["X"].shape[0] == len(paths):
            log.info("loaded cached %s features %s from %s", split, z["X"].shape, cache.name)
            return z["X"], y, paths

    from joblib import Parallel, delayed

    log.info("extracting %s features for %d images in %s", split, len(paths), split)
    vecs = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_extract_one)((p, spam_t, srm_t, color)) for p in tqdm(paths, desc=split)
    )
    X = np.stack(vecs).astype(np.float32)
    ensure_dir(cache_dir)
    np.savez_compressed(cache, X=X, y=y, paths=np.asarray(paths, dtype=object))
    log.info("cached -> %s", cache)
    return X, y, paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Train classical LSB steganalysis model")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="artifacts/classical")
    ap.add_argument("--model", default=None, choices=["xgboost", "svm"])
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap images per (split,class) for quick runs")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--spam-t", type=int, default=None)
    ap.add_argument("--srm-t", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    model_name = args.model or cfg["classical"]["model"]
    spam_t = args.spam_t if args.spam_t is not None else int(cfg["features"]["spam"]["threshold"])
    srm_t = args.srm_t if args.srm_t is not None else int(cfg["features"]["srm"]["threshold"])
    n_jobs = args.n_jobs if args.n_jobs is not None else int(cfg["classical"]["n_jobs"])
    limit = args.limit_per_class if args.limit_per_class is not None else cfg["data"]["limit_per_class"]
    color = bool(cfg["features"].get("color", False))

    manifest = resolve(args.manifest or cfg["paths"]["manifest"])
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}. Run: python -m src.data.download_check")
    df = pd.read_csv(manifest)
    df = apply_limit(df, limit, cfg["seed"])

    cache_dir = resolve(cfg["paths"]["features_dir"])
    Xtr, ytr, _ = extract_split(df, "train", spam_t, srm_t, n_jobs, cache_dir,
                                not args.no_cache, color)
    Xva, yva, _ = extract_split(df, "val", spam_t, srm_t, n_jobs, cache_dir,
                                not args.no_cache, color)
    Xte, yte, _ = extract_split(df, "test", spam_t, srm_t, n_jobs, cache_dir,
                                not args.no_cache, color)
    log.info("feature dim=%d train=%s val=%s test=%s", Xtr.shape[1], Xtr.shape, Xva.shape, Xte.shape)

    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    spw = max(neg / max(pos, 1.0), 1e-6)
    model = C.build_model(cfg, model_name, scale_pos_weight=spw)
    model, scaler = C.fit(model, model_name, Xtr, ytr, Xva, yva)

    out = ensure_dir(resolve(args.out))
    names = feature_names_color(spam_t, srm_t) if color else feature_names(spam_t, srm_t)
    meta = {
        "kind": "classical",
        "model": model_name,
        "spam_t": spam_t,
        "srm_t": srm_t,
        "feature_dim": int(Xtr.shape[1]),
        "color": bool(color),
        "n_train": int(len(ytr)),
        "threshold": float(cfg["eval"]["threshold"]),
    }
    C.save(model, scaler, meta, out)
    save_json({"top_features": C.feature_importances(model, names, 40)}, out / "feature_importance.json")

    for split, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        prob = C.predict_proba(model, scaler, X)
        metrics = compute_metrics(y, prob, cfg["eval"]["threshold"], cfg["eval"]["target_tpr"])
        save_json(metrics, out / f"metrics_{split}.json")
        plot_confusion(metrics, out / f"confusion_{split}.png", f"{model_name} {split}")
        plot_roc(metrics, out / f"roc_{split}.png", f"{model_name} {split}")
        log.info("%s metrics: %s", split, format_metrics(metrics))

    log.info("saved classical artifacts -> %s", out)


if __name__ == "__main__":
    main()
