"""Streamlit web UI for the LSB steganalysis detector.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve
from src.features.lsb import lsb_plane
from src.imageio import load_image
from src.predict import predict_image

st.set_page_config(page_title="LSB Steganography Detector", page_icon="🕵️", layout="wide")

cfg = load_config()
CLASSICAL_DIR = resolve(Path(cfg["paths"]["artifacts_dir"]) / "classical")
CNN_DIR = resolve(Path(cfg["paths"]["artifacts_dir"]) / "cnn")

MODEL_LABELS = {
    "ensemble": "Ensemble (classical + CNN)",
    "classical": "Classical (SPAM+SRM → XGBoost/SVM)",
    "cnn": "CNN (SRNet-lite)",
}


def _available() -> dict[str, bool]:
    return {
        "classical": (CLASSICAL_DIR / "model.joblib").exists(),
        "cnn": (CNN_DIR / "best.pt").exists(),
    }


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _run(path: str, model: str, threshold: float, device: str) -> dict:
    return predict_image(path, model=model, threshold=threshold,
                         classical_dir=CLASSICAL_DIR, cnn_dir=CNN_DIR, device=device)


def _lsb_panel(gray: np.ndarray) -> np.ndarray:
    plane = lsb_plane(gray) * 255
    scale = max(1, 512 // max(gray.shape))
    return np.kron(plane, np.ones((scale, scale), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Controls")
avail = _available()
model = st.sidebar.selectbox(
    "Detector", list(MODEL_LABELS), format_func=lambda k: MODEL_LABELS[k],
    help="Ensemble averages available models.",
)
threshold = st.sidebar.slider("Decision threshold p(stego)", 0.0, 1.0, float(cfg["eval"]["threshold"]), 0.01)
device = st.sidebar.selectbox("Device", ["cuda", "cpu"],
                              index=0 if _gpu_available() else 1,
                              help="Only affects the CNN.")
st.sidebar.markdown("**Model status**")
for name, ok in avail.items():
    st.sidebar.write(f"{'✅' if ok else '❌'} {name}")
if not any(avail.values()):
    st.sidebar.warning("No trained model found. Train one first:\n\n"
                       "`python -m src.train_classical`\n\n"
                       "`python -m src.train_cnn`")
st.sidebar.markdown("---")
st.sidebar.caption("Known-embedder detector: trained on a fixed LSB embedding scheme. "
                   "It will not reliably catch adaptive/unknown embedders.")
if device == "cuda" and not _gpu_available():
    st.sidebar.error("CUDA requested but not available; CNN will fail.")

st.title("🕵️ LSB Steganography Detector")
st.caption("Upload an image to estimate whether it contains a hidden LSB payload.")

tab_detect, tab_batch, tab_about = st.tabs(["Detect", "Batch", "About"])

# --------------------------------------------------------------------------- #
# Single image
# --------------------------------------------------------------------------- #
with tab_detect:
    uploaded = st.file_uploader("Image", type=["png", "bmp", "tif", "tiff", "pgm", "ppm", "webp"])
    if uploaded is not None:
        path = _save_upload(uploaded)
        try:
            gray = load_image(path, mode="gray")
        except Exception as exc:
            st.error(f"Could not read image: {exc}")
            st.stop()

        col_img, col_lsb = st.columns(2)
        with col_img:
            st.subheader("Image")
            st.image(gray, clamp=True, use_container_width=True)
        with col_lsb:
            st.subheader("LSB plane")
            st.image(_lsb_panel(gray), clamp=True, use_container_width=True,
                     caption="Random-looking LSB plane is consistent with embedding")

        with st.spinner("Running detector..."):
            try:
                res = _run(path, model, threshold, device)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", res["label"].upper())
        c2.metric("p(stego)", f"{res['probability']:.4f}")
        c3.metric("Threshold", f"{threshold:.2f}")
        st.progress(min(max(res["probability"], 0.0), 1.0))
        if res["is_stego"]:
            st.error("LSB steganography likely present.")
        else:
            st.success("No LSB steganography detected at this threshold.")

        if res["per_model"]:
            st.write("**Per-model probabilities**")
            st.dataframe(pd.DataFrame(
                [{"model": k, "p(stego)": round(v, 4)} for k, v in res["per_model"].items()]),
                hide_index=True, use_container_width=True)
        if res["errors"]:
            st.info(f"Models skipped: {res['errors']}")

        with st.expander("Statistical analysis (no ML)"):
            stat = res["statistical"]
            pretty = {
                "chi² p-value (global)": stat.get("chi2_p_value"),
                "chi² p-value (block mean)": stat.get("chi2_p_value_blockmean"),
                "RS score": stat.get("rs_score"),
                "RS estimated rate": stat.get("rs_estimated_rate"),
                "SPA close-pair ratio": stat.get("spa_close_ratio"),
                "SPA same-parity ratio": stat.get("spa_same_parity_ratio"),
            }
            st.dataframe(pd.DataFrame(
                [{"measure": k, "value": round(float(v), 5)} for k, v in pretty.items() if v is not None]),
                hide_index=True, use_container_width=True)
            st.caption("High chi² p-value is stego-like (even/odd histogram pairs equalised). "
                       "RS score / estimated rate increase with embedding.")

        with st.expander("LSB-plane features"):
            st.dataframe(pd.DataFrame(
                [{"feature": k, "value": round(float(v), 5)} for k, v in res["lsb"].items()]),
                hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Batch detection")
    files = st.file_uploader("Images", type=["png", "bmp", "tif", "tiff", "pgm", "ppm", "webp"],
                             accept_multiple_files=True, key="batch")
    if files and st.button("Run batch", type="primary"):
        rows = []
        progress = st.progress(0.0)
        for i, f in enumerate(files, 1):
            path = _save_upload(f)
            try:
                res = _run(path, model, threshold, device)
                rows.append({
                    "file": f.name,
                    "verdict": res["label"],
                    "p_stego": round(res["probability"], 5),
                    "chi2_p": round(res["statistical"].get("chi2_p_value", float("nan")), 5),
                    "rs_score": round(res["statistical"].get("rs_score", float("nan")), 5),
                })
            except Exception as exc:
                rows.append({"file": f.name, "verdict": f"error: {exc}", "p_stego": None,
                             "chi2_p": None, "rs_score": None})
            progress.progress(i / len(files))
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           file_name="lsb_detection_results.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
### How it works
1. **Classical detector** — extracts SPAM + SRM-style residual + LSB-plane features
   and classifies with XGBoost (or SVM). Fast, CPU-friendly, strong baseline.
2. **CNN detector** — a compact SRNet-lite network with a fixed high-pass first layer
   that preserves the weak LSB signal.
3. **Ensemble** — averages the probabilities of whichever models are trained.

### Interpreting results
- **p(stego)** is calibrated only loosely; use the threshold that matches your
  acceptable false-positive rate.
- The **chi² p-value** jumps toward 1 when even/odd histogram pairs are equalised,
  the signature of LSB replacement.

### Limitations
- Trained for a **known embedder** on the Stego-Images-Dataset. It generalises
  poorly to adaptive schemes (S-UNIWARD, HILL) or to JPEG recompression.
- Never re-compress or resize before detection — that destroys the LSB signal.
        """
    )
    for name, path in (("Classical", CLASSICAL_DIR), ("CNN", CNN_DIR)):
        mp = path / "metrics_test.json"
        if mp.exists():
            import json

            m = json.loads(mp.read_text())
            st.write(f"**{name} — test metrics**")
            st.json({k: v for k, v in m.items() if k not in ("roc", "confusion_matrix")})
