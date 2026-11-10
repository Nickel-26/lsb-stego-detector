from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.models import classical as C
from src.predict import predict_image
from src.train_classical import extract_split


def test_predict_classical_artifact(synthetic_dataset, tmp_path):
    manifest, _, _ = synthetic_dataset
    df = pd.read_csv(manifest)
    cache = tmp_path / "features"
    Xtr, ytr, _ = extract_split(df, "train", 3, 2, 1, cache)
    Xte, yte, _ = extract_split(df, "test", 3, 2, 1, cache)

    cfg = load_config()
    cfg["classical"]["xgboost"].update({"n_estimators": 60, "max_depth": 4})
    model = C.build_model(cfg, "xgboost", scale_pos_weight=1.0)
    model, scaler = C.fit(model, "xgboost", Xtr, ytr)

    art = tmp_path / "artifacts"
    C.save(model, scaler, {"kind": "classical", "model": "xgboost", "spam_t": 3,
                           "srm_t": 2, "feature_dim": Xtr.shape[1]}, art)

    test_df = df[df["split"] == "test"]
    clean = test_df[test_df["label"] == 0].iloc[0]["path"]
    stego = test_df[test_df["label"] == 1].iloc[0]["path"]

    res_clean = predict_image(clean, model="classical", classical_dir=art)
    res_stego = predict_image(stego, model="classical", classical_dir=art)

    for res in (res_clean, res_stego):
        assert 0.0 <= res["probability"] <= 1.0
        assert res["label"] in ("clean", "stego")
        assert "statistical" in res
        assert "lsb" in res
    assert res_stego["probability"] >= res_clean["probability"]
