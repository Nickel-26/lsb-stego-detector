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
`evaluate.py` reloads the saved model, recomputes the metrics and overwrites the
metrics JSON / plots — it never changes the model weights.

### 5.1 What each metric means

Assume the positive class is `stego` (label `1`):

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | `(TP + TN) / N` | overall fraction of correct decisions |
| Precision | `TP / (TP + FP)` | of the images we flagged, how many are truly stego |
| Recall / TPR | `TP / (TP + FN)` | of the real stego images, how many we caught |
| F1 | `2·P·R / (P + R)` | harmonic mean of precision and recall |
| ROC-AUC | area under TPR-vs-FPR curve | ranking quality: 0.5 = random, 1.0 = perfect |
| FPR @ 95% TPR | FP rate where recall first reaches 95% | false alarms when catching 95% of stego |
| Threshold @ 95% TPR | probability cutoff at that operating point | how to set the UI slider operationally |

Confusion matrices below use this layout:

```text
                Predicted
              clean   stego
Actual clean     TN      FP
Actual stego     FN      TP
```

Each split has **300 images: 100 clean + 200 stego**, so the classes are not
balanced (1:2). Metrics are reported on that same distribution.

### 5.2 Classical model — XGBoost (2,976-dim RGB features)

| Split | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 95% | Thresh @ 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 300 | 0.9267 | 0.9785 | 0.9100 | 0.9430 | 0.98145 | 0.09 | 0.2200 |
| Test | 300 | **0.9467** | **0.9894** | **0.9300** | **0.9588** | **0.98450** | 0.10 | 0.2347 |

```text
Validation:  [[96,   4],     Test:  [[98,   2],
              [18, 182]]             [14, 186]]
```

### 5.3 CNN model — SRNet-lite (3 × 128 × 128 RGB)

| Split | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 95% | Thresh @ 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 300 | 0.9333 | 0.9891 | 0.9100 | 0.9479 | **0.99065** | **0.03** | 0.2948 |
| Test | 300 | 0.9133 | 0.9677 | 0.9000 | 0.9326 | 0.98360 | 0.09 | 0.3128 |

```text
Validation:  [[98,   2],     Test:  [[94,   6],
              [18, 182]]             [20, 180]]
```

Test set side by side at threshold 0.5:

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 98 | 2 | 14 | 186 | 0.947 | 0.989 | 0.930 |
| CNN | 94 | 6 | 20 | 180 | 0.913 | 0.968 | 0.900 |

### 5.4 Where the ROC curve data comes from

The curve needs only two values per image: the true label (`0`/`1`, from the
manifest) and the model's continuous `p(stego)` score. `src/utils.py` passes the
300 pairs from a split to scikit-learn:

```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
```

`roc_curve` sweeps the decision threshold; at each step it reports the true
positive rate and false positive rate. Plotting TPR against FPR gives the curve
and the area under it is the AUC. The full `roc.fpr` / `roc.tpr` arrays are saved
in `metrics_{val,test}.json` and rendered into `roc_{val,test}.png`.

Replot from the saved JSON without reloading a model:

```python
import json
import matplotlib.pyplot as plt

for model in ("classical", "cnn"):
    d = json.load(open(f"artifacts/{model}/metrics_test.json"))
    plt.plot(d["roc"]["fpr"], d["roc"]["tpr"],
             label=f"{model} (AUC={d['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

### 5.5 How to read these results

- Both models are strong on the held-out test split (ROC-AUC ≈ 0.984).
- The classical model has better test accuracy and precision at threshold 0.5
  (2 false positives vs. 6 for the CNN).
- The CNN has the best validation AUC (0.9907) and a much better validation
  FPR@95 (0.03 vs. 0.09); on the test set its FPR@95 is comparable.
- At 95% recall, both models still flag roughly **9–10 clean images per 100** —
  that is the honest operating cost on this dataset. Use the "Threshold @ 95%"
  values above if that operating point matters more than raw accuracy.
- No separate ensemble metrics file is saved: the ensemble is a runtime average
  of the two probabilities in `src/predict.py`.

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
