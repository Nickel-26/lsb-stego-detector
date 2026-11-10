from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import classical as C
from src.train_classical import extract_split
from src.utils import compute_metrics


def test_xgboost_learns_synthetic_lsb(synthetic_dataset, tmp_path):
    manifest, _, df = synthetic_dataset
    df = pd.read_csv(manifest)
    cache = tmp_path / "features"
    Xtr, ytr, _ = extract_split(df, "train", 3, 2, 1, cache)
    Xte, yte, _ = extract_split(df, "test", 3, 2, 1, cache)
    assert Xtr.shape[1] == Xte.shape[1]
    assert len(np.unique(ytr)) == 2

    cfg = load_config()
    cfg["classical"]["xgboost"].update({"n_estimators": 80, "max_depth": 4, "learning_rate": 0.15})
    model = C.build_model(cfg, "xgboost", scale_pos_weight=1.0)
    model, scaler = C.fit(model, "xgboost", Xtr, ytr)

    prob = C.predict_proba(model, scaler, Xte)
    metrics = compute_metrics(yte, prob)
    assert metrics["roc_auc"] is not None
    assert metrics["roc_auc"] > 0.8
