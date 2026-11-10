"""Classical steganalysis models: XGBoost / SVM on SPAM+SRM+LSB features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..utils import ensure_dir, get_logger

log = get_logger()


def build_model(cfg: dict[str, Any], model: str = "xgboost",
                scale_pos_weight: float = 1.0) -> Any:
    if model == "xgboost":
        from xgboost import XGBClassifier

        params = dict(cfg["classical"]["xgboost"])
        params.setdefault("random_state", cfg.get("seed", 1337))
        params.setdefault("n_jobs", cfg["classical"].get("n_jobs", -1))
        params["scale_pos_weight"] = scale_pos_weight
        params.setdefault("early_stopping_rounds", 30)
        try:
            return XGBClassifier(**params)
        except TypeError:  # older/newer API without early_stopping_rounds kwarg
            params.pop("early_stopping_rounds", None)
            return XGBClassifier(**params)
    if model == "svm":
        from sklearn.svm import SVC

        params = cfg["classical"]["svm"]
        return SVC(C=params["C"], gamma=params["gamma"], kernel="rbf",
                   probability=True, class_weight="balanced", cache_size=1024)
    raise ValueError(f"unknown classical model: {model}")


def needs_scaler(model: str) -> bool:
    return model == "svm"


def fit(model_obj: Any, model: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
    """Fit model and return (model, scaler_or_None)."""
    scaler = None
    if needs_scaler(model):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = scaler.transform(X_val)

    if model == "xgboost":
        if X_val is None:
            try:
                model_obj.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            model_obj.fit(X_train, y_train, verbose=False)
        else:
            try:
                model_obj.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                model_obj.fit(X_train, y_train, verbose=False)
    else:
        model_obj.fit(X_train, y_train)

    if X_val is not None:
        prob = model_obj.predict_proba(X_val)[:, 1]
        from ..utils import compute_metrics

        log.info("val metrics: %s", compute_metrics(y_val, prob))
    return model_obj, scaler


def predict_proba(model_obj: Any, scaler: Any, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    return model_obj.predict_proba(X)[:, 1]


def save(model_obj: Any, scaler: Any, meta: dict[str, Any], out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    joblib.dump(model_obj, out / "model.joblib")
    if scaler is not None:
        joblib.dump(scaler, out / "scaler.joblib")
    from ..utils import save_json

    save_json(meta, out / "meta.json")
    return out


def load(out_dir: str | Path):
    out = Path(out_dir)
    model_obj = joblib.load(out / "model.joblib")
    scaler_path = out / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model_obj, scaler


def feature_importances(model_obj: Any, names: list[str], top: int = 30) -> list[dict]:
    try:
        imp = np.asarray(model_obj.feature_importances_)
    except Exception:
        return []
    order = np.argsort(imp)[::-1][:top]
    return [{"feature": names[i], "importance": float(imp[i])} for i in order]
