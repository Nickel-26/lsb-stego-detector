# LSB Steganography Detector

Binary detector for **LSB (least-significant-bit) image steganography**, built for a
**known embedder**. Two detectors are trained and compared, then exposed through a CLI
and a Streamlit web UI:

| Track | Features | Model |
|---|---|---|
| Classical | SPAM + SRM-style residuals + LSB-plane stats | XGBoost (or SVM) |
| Deep | raw grayscale pixels, high-pass first layer | SRNet-lite CNN |
| Ensemble | average of the two probabilities | — |

A statistical sanity suite (chi-square, RS analysis, sample-pair statistics) is included
both as a model-free baseline and as a dataset validator.

> **Scope / limitation:** the detector is trained against a fixed LSB embedding scheme.
> It will not reliably catch adaptive embedders (S-UNIWARD, HILL, MiPOD) or images that
> have been JPEG-recompressed after embedding. Never resize/recompress before detection —
> it destroys the LSB signal.

---

## 1. Environment

Tested layout: Linux, Python 3.12 (recommended for PyTorch), NVIDIA GPU for the CNN.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install PyTorch (GPU)

If the default `pip install torch` is CPU-only, install a CUDA build for your driver:

```bash
# Example: CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

If `nvidia-smi` fails, fix the driver / container GPU passthrough before training the CNN.

---

## 2. Dataset (manual download)

Download **Stego-Images-Dataset** from Kaggle and extract it into this layout:

```
data/raw/
  train/{clean,stego}/
  val/{clean,stego}/
  test/{clean,stego}/
```

Then validate and build the manifest:

```bash
python -m src.data.download_check
```

This checks counts, hashes images to detect duplicates/leakage, and writes
`data/manifest.csv`. Use `--limit-per-class 2000` for a quick subset and `--quick`
to skip content hashing.

If cover/stego pairs leak across splits, regroup them without leakage:

```bash
python -m src.data.split --manifest data/manifest.csv
```

### Try the pipeline without downloading anything

```bash
python scripts/make_synthetic_dataset.py     # writes a tiny synthetic dataset
python -m src.data.download_check            # validates it
python scripts/check_signal.py               # chi-square/RS separation report
```

---

## 3. Check that the signal is real

```bash
python scripts/check_signal.py --split train --n 300
```

AUC ≈ 0.5 means no separation (labels/dataset problem). LSB data should show strong
separation (> 0.8) for chi-square / RS features.

---

## 4. Train

### Classical (CPU, fast)

```bash
python -m src.train_classical                 # XGBoost, per configs/default.yaml
python -m src.train_classical --model svm
python -m src.train_classical --limit-per-class 3000   # quick run
```

Artifacts → `artifacts/classical/` (`model.joblib`, `meta.json`, metrics + plots).

### CNN (GPU recommended)

```bash
python -m src.train_cnn                       # SRNet-lite
python -m src.train_cnn --arch efficientnet_v2_s
python -m src.train_cnn --limit-per-class 3000 --epochs 15
```

Artifacts → `artifacts/cnn/` (`best.pt`, `meta.json`, history, metrics + plots).

---

## 5. Evaluate

```bash
python -m src.evaluate --model classical --artifacts artifacts/classical --split test
python -m src.evaluate --model cnn       --artifacts artifacts/cnn       --split test
```

Reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix and
**FPR at 95% TPR** (`fpr_at_tpr_0.95`), which matters more than raw accuracy.

---

## 6. Predict

```bash
python -m src.predict photo.png                       # ensemble
python -m src.predict photo.png --model classical
python -m src.predict a.png b.png --json
```

Python API:

```python
from src.predict import predict_image
res = predict_image("photo.png", model="ensemble", threshold=0.5)
print(res["label"], res["probability"], res["per_model"])
```

---

## 7. Web UI

```bash
streamlit run app/streamlit_app.py
```

Features: drag-and-drop detection, LSB-plane visualisation, per-model probabilities,
statistical analysis, batch mode with CSV export, adjustable threshold.

---

## 8. Test

```bash
python -m pytest -q
```

The synthetic dataset fixtures let the feature, data, classical-training and inference
tests run with no external data. CNN tests are skipped automatically if PyTorch is absent.

---

## Project layout

```
configs/default.yaml       tunables
src/
  config.py  utils.py  imageio.py
  data/      download_check.py  split.py  dataset.py
  features/  statistical.py  lsb.py  spam.py  srm.py  extract.py
  models/    classical.py  cnn.py  transfer.py  factory.py
  train_classical.py  train_cnn.py  evaluate.py  predict.py
app/streamlit_app.py
scripts/   make_synthetic_dataset.py  check_signal.py
tests/
```

## Notes on the method

- **SPAM** = second-order Markov transition probabilities of clipped pixel differences (686 dims).
- **SRM-subset** = bank of high-pass residuals + quantised co-occurrence histograms.
- **LSB stats** = entropy, run length, even/odd histogram asymmetry, spatial autocorrelation.
- **SRNet-lite** = frozen SRM high-pass first layer, residual conv blocks, global average
  pooling. No early pooling so the weak LSB signal survives.
- Splits are **group-aware**: a cover and its stego copy never land in different splits.
# lsb-stego-detector
# lsb-stego-detector
# lsb-stego-detector
# lsb-stego-detector
# lsb-stego-detector
