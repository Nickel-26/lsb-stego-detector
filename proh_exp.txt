# PROJECT.md — LSB Steganography Detector

Complete technical write-up of what was built, why it was built that way, what data
was used, how the models were trained, how they performed, and how to reproduce or
operate the project.

---

## 1. What the project is

A binary image-steganalysis system that decides whether an image contains a hidden
payload embedded with **LSB (least-significant-bit) replacement**.

| Input | Output |
|---|---|
| An image file | `clean` or `stego`, plus `p(stego)` in `[0, 1]` |

Two independent detectors are trained and exposed through a CLI and a Streamlit UI:

| Track | Representation | Model |
|---|---|---|
| Classical | per-channel SPAM + SRM-style residuals + LSB-plane statistics | XGBoost (SVM alternative implemented) |
| Deep | raw RGB pixels → frozen high-pass residual layer | SRNet-lite CNN |
| Ensemble | unweighted mean of available model probabilities | — |

The project also includes model-free statistical steganalysis (chi-square, RS
analysis, sample-pair analysis) used as a sanity check and shown in the UI.

---

## 2. Scope and threat model

### In scope

- **LSB replacement / LSB matching-style embedding** in raster images.
- **Known embedder**: the detector is trained against the embedding scheme and
  payload variants present in the selected dataset.
- **Binary detection**: clean vs. stego.
- **Lossless image paths**: PNG/BMP/TIFF and similar formats where the LSB plane
  survives.
- **Both RGB and grayscale support in the core code**, with RGB as the actual
  trained configuration.

### Explicitly out of scope

- Adaptive content-adaptive schemes such as S-UNIWARD, HILL, MiPOD.
- Robust watermarking, DCT-domain steganography, spread-spectrum methods.
- Payload extraction, payload-length estimation, or locating the hidden message.
- Detection after JPEG recompression, resizing, or other lossy post-processing.
- General forensic-grade attribution. This is a trained detector for a known
  distribution, not a universal steganalysis engine.

The key limitation is repeated throughout the code and UI: **never resize or
recompress an image before detection**, because that destroys the LSB signal.

---

## 3. Final results at a glance

### Classical model — XGBoost, 2,976-dim RGB features

| Split | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 95% TPR | Threshold at 95% TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 300 | 0.9267 | 0.9785 | 0.9100 | 0.9430 | 0.98145 | 0.09 | 0.2200 |
| Test | 300 | **0.9467** | **0.9894** | **0.9300** | **0.9588** | **0.98450** | **0.10** | 0.2347 |

Confusion matrices (rows = actual clean/stego, columns = predicted clean/stego):

```text
Validation: [[96,  4],
             [18, 182]]

Test:       [[98,  2],
             [14, 186]]
```

### CNN model — SRNet-lite, 3×128×128 RGB input

| Split | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 95% TPR | Threshold at 95% TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 300 | 0.9333 | 0.9891 | 0.9100 | 0.9479 | **0.99065** | 0.03 | 0.2948 |
| Test | 300 | 0.9133 | 0.9677 | 0.9000 | 0.9326 | **0.98360** | 0.09 | 0.3128 |

```text
Validation: [[98,  2],
             [18, 182]]

Test:       [[94,  6],
             [20, 180]]
```

### Interpretation

- Both models are strong on the held-out test split: AUC ≈ 0.984.
- The classical model has slightly better test accuracy and precision (fewer false
  positives at threshold 0.5: 2 vs. 6 clean images flagged).
- The CNN has a slightly better validation AUC and a much better validation
  FPR@95 (3% vs. 9%), but its test FPR@95 is comparable.
- At 95% recall, both produce roughly **9–10 false positives per 100 clean
  images**. That is the honest operating point for this dataset; raw accuracy at
  0.5 is not the whole story.
- The ensemble is an unweighted average. A separate ensemble metrics file was not
  generated; during smoke testing it corrected a borderline CNN false positive
  (see §11).

---

## 4. Environment and hardware

### Training-time environment (historical)

| Item | Value |
|---|---|
| OS | Fedora 44 |
| Kernel | `7.1.10-200.fc44.x86_64` |
| Python | 3.14.7 |
| CPU | 16 cores |
| GPU | NVIDIA RTX 3050 6 GB Laptop |
| GPU driver | 610.57.04 |
| CUDA UMD | 13.3 |
| Compute capability | 8.6 |
| PyTorch | `2.14.1+cu130` |
| torchvision | `0.29.1+cu130` |
| NumPy | 2.5.4 |
| OpenCV | 5.0.0 |
| XGBoost | 3.4.0 |
| scikit-learn | 1.9.0 |
| Streamlit | 1.67.0 |

The GPU was enabled after disabling Secure Boot and rebooting so the proprietary
NVIDIA kernel driver could bind (nouveau was no longer using the device).
`nvidia-smi` and a CUDA matmul test both succeeded at that point.

There was no CUDA-enabled PyTorch build for Python 3.14 at version 2.15, so the
latest available CUDA wheel was used:

```bash
python3 -m pip install --user --force-reinstall \
  torch==2.14.1+cu130 torchvision==0.29.1+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

Verification used:

```bash
python3 -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item())"
```

### Current operational note

At the time this document was generated, the host GPU stack was reporting a
driver/library mismatch again:

```text
nvidia-smi: Failed to initialize NVML: Driver/library version mismatch
NVML library version: 620.70
PyTorch: CUDA unknown error / torch.cuda.is_available() == False
```

This does **not** invalidate the trained models. The classical model is CPU-only.
The CNN can run on CPU (slower). GPU inference/training resumes after the driver
and library are realigned, typically a reboot or driver reload. The saved
`best.pt` was produced earlier while the RTX 3050 was working.

---

## 5. Dataset: from the original plan to the real data

### 5.1 Planned path

The initial plan was the Kaggle dataset
`marcozuppelli/stegoimagesdataset`, using the documented
`data/raw/{train,val,test}/{clean,stego}` layout. Kaggle credentials were not
available, so this path remained optional/manual.

### 5.2 Actual dataset used

The project switched to the real Hugging Face dataset:

**`emirozturk/LSBStegoDataset`** — “Multilingual LSB Stego Dataset”.

- Cover archive: `128x128.tar` (~33 MB, 1,000 covers)
- English stego archive: `128x128_en.tar` (~766 MB, 18,000 stego PNGs plus
  ~19,000 metadata JSON files)

Download URLs:

```text
https://huggingface.co/datasets/emirozturk/LSBStegoDataset/resolve/main/128x128.tar
https://huggingface.co/datasets/emirozturk/LSBStegoDataset/resolve/main/128x128_en.tar
```

The stego filenames encode the embedding variant:

```text
<cover>_hr_<ratio>_hb_<bits>_<policy>.png
```

For example, a cover named `1000` can have variants such as
`1000_hr_0.25_hb_1_random.png`, representing a payload ratio, bits per channel,
and embedding policy. There are up to 18 variants per cover in the archive.

### 5.3 What was actually materialised

`scripts/prepare_hf_stego.py` pairs each cover with a random sample of stego
variants and writes the layout expected by the rest of the pipeline.

The final dataset:

- **1,000 covers**
- **2 stego variants per cover**
- **3,000 images total**: 1,000 clean + 2,000 stego
- **1,000 groups** (one group = one cover plus all its stego variants)
- **~101.5 MB** of image data
- **27 JPG covers, 2,973 PNG files** overall

The mixed formats are worth noting: the clean covers include 27 JPEGs, while the
stego variants are PNGs. JPEG covers are lossy-compressed source images; the
stego images are written losslessly so the embedded LSBs survive.

### 5.4 Split

Splits are assigned **by group**, never by individual file:

| Split | Covers / groups | Clean | Stego | Total |
|---|---:|---:|---:|---:|
| Train | 800 | 800 | 1,600 | 2,400 |
| Val | 100 | 100 | 200 | 300 |
| Test | 100 | 100 | 200 | 300 |
| **Total** | **1,000** | **1,000** | **2,000** | **3,000** |

Class balance is **1 clean : 2 stego**, not 1:1. This is handled with class
weighting in both models.

Leakage checks:

- `groups_across_splits = 0`
- duplicate MD5 content across splits = `0`
- every cover and all of its stego variants stay in the same split

This matters enormously: if a cover appears in train and its stego copy appears in
test, the model can memorize the source image and report unrealistically high
accuracy.

### 5.5 Manifest

`data/manifest.csv` has 3,000 rows and these columns:

```text
path, label, split, cls, group_id, md5, size_bytes
```

- `label`: `0 = clean`, `1 = stego`
- `split`: `train`, `val`, `test`
- `cls`: string version of the label
- `group_id`: cover identity shared by a cover and all its variants
- `md5`: content hash for duplicate/leakage detection
- `size_bytes`: file size

Example group-id derivation:

```text
1000                       -> 1000
1000_hr_0.25_hb_1_random   -> 1000
1000__v1                   -> 1000
```

The rules live in `src/data/download_check.py`:

```python
_VARIANT_SUFFIX = re.compile(r"(?:_hr_.*|__v\d+)$", re.IGNORECASE)
```

`derive_group_id()` first strips the variant suffix, then strips common label
tokens such as `clean`, `cover`, `stego`, `lsb`, `hidden`, etc.

---

## 6. The critical discovery: grayscale destroys the signal

This was the most important technical finding in the project.

The first implementation converted every image to grayscale and extracted
992 features from that single plane. The dataset labels looked correct, but the
signal was weak. A sanity check using model-free RS analysis produced roughly:

| Input representation | RS-based separation (AUC) |
|---|---:|
| Grayscale conversion | ~0.51 (essentially random) |
| Per-channel RGB | ~0.94 (strong) |

Why: LSB replacement embeds bits across the R, G, and B channels independently.
Converting to grayscale computes a weighted sum of the channels and rounds to
8 bits. That operation mixes the bit planes and effectively destroys the
lowest-bit structure the detector depends on.

The pipeline was therefore converted to **RGB / per-channel analysis**:

- classical feature extraction: run SPAM, SRM, and LSB statistics separately on
  R, G, and B, then concatenate the vectors;
- CNN: `in_channels=3`;
- dataloaders: load RGB and return CHW tensors;
- prediction: load both grayscale (for display/statistics) and RGB (for models).

This single change is the difference between a near-random model and a strong
detector. It is also why the README and UI insist on not converting or
recompressing images before detection.

---

## 7. End-to-end pipeline

```text
HF tars
  -> extract covers + stego variants
  -> scripts/prepare_hf_stego.py: pair and split by cover
  -> data/raw/{train,val,test}/{clean,stego}/
  -> src.data.download_check: scan, hash, group, write manifest
  -> data/manifest.csv
  -> model-free sanity checks (chi-square / RS / SPA)
  -> TWO training tracks:
       (a) src.train_classical: SPAM+SRM+LSB -> XGBoost
       (b) src.train_cnn: RGB -> SRNet-lite CNN
  -> artifacts/classical/ and artifacts/cnn/
  -> src.evaluate: metrics + ROC/confusion plots
  -> src.predict: classical / CNN / ensemble inference
  -> app/streamlit_app.py: web UI
```

---

## 8. Image loading and the “no lossy preprocessing” rule

`src/imageio.py` handles loading:

- OpenCV first (`cv2.imread` with `IMREAD_UNCHANGED`), Pillow fallback.
- `mode="gray"` → uint8 `(H, W)`.
- `mode="rgb"` → uint8 `(H, W, 3)` in **RGB order** (OpenCV loads BGR, so the
  code converts).
- Handles 4-channel images by dropping alpha.
- Handles 2D/3D shapes consistently.

`center_crop_or_pad()` is deterministic:

- If the image is smaller than the target, it pads with edge pixels.
- Otherwise it center-crops to `size × size`.
- It handles both 2D and 3D arrays.

No resizing is performed anywhere in feature extraction or inference. Resizing
would interpolate pixels and destroy the LSB plane.

CNN training uses random **crops** and horizontal flips, not resize. At the
actual training resolution (128×128 images, `--img-size 128`), the random crop is
effectively a no-op because the source is already the target size; the meaningful
augmentation is the horizontal flip. Validation/test use deterministic center
crop/pad.

---

## 9. Classical features (2,976 dimensions)

The classical model sees a flat vector built from three feature families, applied
per channel.

### 9.1 SPAM — 686 features per channel

Source: `src/features/spam.py`.

SPAM = **Subtractive Pixel Adjacency Model**.

For an image, compute horizontal and vertical neighbour differences:

```text
d_h[i, j] = x[i, j+1] - x[i, j]
d_v[i, j] = x[i+1, j] - x[i, j]
```

Clip and round each difference to the range `[-T, T]` (default `T = 3`), then
compute second-order Markov transition probabilities over triples of consecutive
differences:

```text
p(a, b, c) = count(d_i = a, d_{i+1} = b, d_{i+2} = c) / total
```

Number of states per difference: `2T + 1 = 7`.
Features per direction: `7^3 = 343`.
Two directions (horizontal, vertical): `2 × 343 = 686`.

SPAM is good at detecting local residual irregularities introduced by LSB
replacement.

### 9.2 SRM-style residuals — 300 features per channel

Source: `src/features/srm.py`.

The full Spatial Rich Model has ~34k features. This project uses a compact subset
of **12 high-pass filters**:

1. first-order horizontal derivative `[-1, 1]`
2. first-order vertical derivative
3. second-order horizontal `[1, -2, 1]`
4. second-order vertical
5. third-order horizontal `[-1, 3, -3, 1]`
6. third-order vertical
7. 3×3 square residual
8. 3×3 horizontal edge residual
9. 3×3 vertical edge residual
10. 3×3 Laplacian
11. 3×3 KB-like residual
12. 1×5 residual

For each filter:

```text
r = filter2D(image, kernel)
q = clip(round(r), -T, T) + T       # T = 2, so q in [0, 4]
```

Then build a normalized 2D co-occurrence histogram of horizontally and
vertically adjacent quantized residuals. With `T = 2`, each histogram has
`(2T + 1)^2 = 25` bins.

Total: `12 filters × 25 bins = 300` features per channel.

SRM residuals are the strongest family in the trained model; the top feature
importances are dominated by `srm_*` entries.

### 9.3 LSB-plane statistics — 6 features per channel

Source: `src/features/lsb.py`.

For the LSB plane `p = image & 1`:

| Feature | Meaning |
|---|---|
| `lsb_mean` | fraction of pixels with LSB = 1 |
| `lsb_entropy` | binary entropy of the LSB plane |
| `lsb_even_odd_asymmetry` | normalized sum of `|hist[2k] - hist[2k+1]|`; drops toward 0 under LSB replacement |
| `lsb_run_length` | number of runs / pixel count; randomized planes have more runs |
| `lsb_autocorr_h` | horizontal spatial autocorrelation of the LSB plane |
| `lsb_autocorr_v` | vertical spatial autocorrelation |

### 9.4 Total dimensionality

Per channel:

```text
SPAM 686 + SRM 300 + LSB 6 = 992
```

Per RGB image:

```text
992 × 3 channels = 2,976
```

Feature naming is channel-prefixed, e.g. `R_srm_112`, `G_spam_511`,
`B_lsb_mean`. The vector passed to XGBoost is the concatenation of the three
channel vectors. The richer dictionary (`stat`, `lsb` with per-channel keys) is
retained for the UI but is not what the model consumes.

### 9.5 Feature importance observations

`artifacts/classical/feature_importance.json` stores the top 40 features. The
leading entries are:

| Feature | Importance |
|---|---:|
| `R_srm_112` | 0.0410 |
| `R_srm_212` | 0.0260 |
| `R_srm_268` | 0.0211 |
| `G_srm_258` | 0.0153 |
| `B_srm_212` | 0.0152 |
| `G_spam_511` | 0.0129 |
| `G_lsb_mean` | 0.0125 |

SRM residuals dominate, with SPAM and a few LSB statistics contributing. All
three color channels appear, which is consistent with per-channel LSB embedding.

---

## 10. Model-free statistical steganalysis

Source: `src/features/statistical.py`.

These methods require no training and are used both as a baseline and as a
dataset validator. They are also shown in the web UI.

### 10.1 Chi-square attack (Westfeld & Pfitzmann)

LSB replacement equalizes the counts of even/odd histogram pairs. For each pair:

```text
expected = (hist[2k] + hist[2k+1]) / 2
chi2 = sum((even - expected)^2 / expected + (odd - expected)^2 / expected)
```

A small p-value is cover-like; a large p-value is stego-like. The code computes:

- global chi-square statistic, degrees of freedom, p-value;
- a block-wise mean p-value over 32×32 blocks (`chi2_p_value_blockmean`), which
  is more stable for a single image.

### 10.2 RS analysis (Fridrich & Goljan)

Groups of four adjacent pixels are classified as regular or singular before and
after two flipping operations:

- `F1`: LSB flip (`x ^ 1`)
- `F-1`: dual flip (even → `-1`, odd → `+1`, clamped)

The discriminant is the sum of absolute differences inside each group. The code
returns `rs_Rp`, `rs_Sp`, `rs_Rm`, `rs_Sm`, an `rs_score`, and a clipped
`rs_estimated_rate`. The score increases with embedding.

### 10.3 Sample-pair analysis (SPA heuristic)

For horizontally and vertically adjacent pixel pairs:

- `spa_close_ratio`: fraction with `|x - y| == 1`
- `spa_same_parity_ratio`: fraction with equal parity

LSB embedding tends to increase close-pair statistics among otherwise smooth
neighbours.

These features are what `scripts/check_signal.py` prints, with direction-agnostic
AUC per feature. A feature AUC near 0.5 means no separation; above ~0.8 is strong
evidence that the labels/embedding are meaningful.

---

## 11. Classical model

Source: `src/train_classical.py`, `src/models/classical.py`.

### 11.1 Model

Chosen model: **XGBoost** (`XGBClassifier`).

Default configuration (`configs/default.yaml`):

```yaml
n_estimators: 400
max_depth: 6
learning_rate: 0.08
subsample: 0.9
colsample_bytree: 0.8
tree_method: hist
eval_metric: auc
```

Additional behavior:

- `random_state = 1337`
- `n_jobs = -1`
- `early_stopping_rounds = 30` with the validation set
- `scale_pos_weight = neg / pos`, computed from the training labels

With 800 clean and 1,600 stego training images, that weight is `800 / 1600 = 0.5`,
which downweights the majority stego class and keeps the decision boundary from
collapsing toward “predict stego everywhere.”

An SVM alternative (`SVC` with RBF kernel, `C=5`, `gamma="scale"`,
`probability=True`, `class_weight="balanced"`) is implemented and selectable with
`--model svm`, but was not the final chosen model. SVM uses `StandardScaler`;
XGBoost does not.

### 11.2 Training flow

1. Load `data/manifest.csv`.
2. Optionally cap per `(split, class)` with `--limit-per-class`.
3. Extract features for train/val/test in parallel with `joblib`.
4. Cache each split as an `.npz` under `data/features/`:
   - `train_spam3_srm2_color.npz`
   - `val_spam3_srm2_color.npz`
   - `test_spam3_srm2_color.npz`
   - The cache stores `X`, `y`, and the exact path list.
   - On reuse, the path list is compared; a stale cache is ignored.
5. Fit XGBoost with the validation set for early stopping.
6. Save artifacts.
7. Compute validation and test metrics, ROC curves, and confusion matrices.

### 11.3 Artifacts

`artifacts/classical/`:

| File | Purpose |
|---|---|
| `model.joblib` | trained XGBoost model (~452 KB) |
| `meta.json` | model type, feature settings, dim, color flag, training size |
| `metrics_val.json` | validation metrics + full ROC arrays |
| `metrics_test.json` | test metrics + full ROC arrays |
| `confusion_val.png` | validation confusion matrix |
| `confusion_test.png` | test confusion matrix |
| `roc_val.png` | validation ROC curve |
| `roc_test.png` | test ROC curve |
| `feature_importance.json` | top 40 features by XGBoost importance |

Current `meta.json`:

```json
{
  "kind": "classical",
  "model": "xgboost",
  "spam_t": 3,
  "srm_t": 2,
  "feature_dim": 2976,
  "color": true,
  "n_train": 2400,
  "threshold": 0.5
}
```

### 11.4 Why it works well

LSB replacement changes local residual statistics subtly but consistently. SPAM
and SRM were designed for exactly this kind of detection, and a gradient-boosted
tree model can combine hundreds of weak residual cues. The per-channel
representation preserves the signal that grayscale conversion would have mixed
away.

The classical model is also fast at inference: feature extraction is CPU-only and
the model is a few hundred kilobytes.

---

## 12. CNN model

Sources: `src/models/cnn.py`, `src/models/factory.py`,
`src/data/dataset.py`, `src/train_cnn.py`.

### 12.1 Architecture: SRNet-lite

Input: `3 × 128 × 128` RGB, scaled to `[0, 1]`.

**Frozen high-pass first layer**

- A bank of 5 high-pass 3×3 kernels taken from the SRM filter list.
- Kernel weights are normalized by their absolute sum.
- The bank is replicated independently for each input channel.
- Implemented as a grouped convolution:
  - `in_channels = C`
  - `out_channels = 5 × C`
  - `groups = C`
- The high-pass weights are **frozen** (`requires_grad = False`).

For RGB, this produces `5 × 3 = 15` residual maps. The frozen layer ensures the
network starts from residual-like representations rather than learning arbitrary
low-level filters that might overfit to image content.

**Residual convolutional trunk**

Eight `ConvBlock`s. Each block is:

```text
Conv3×3 → BatchNorm → LeakyReLU(0.1)
Conv3×3 → BatchNorm → LeakyReLU(0.1)
```

Channels/strides:

```text
15 → 16  (stride 1)
16 → 16  (stride 1)
16 → 32  (stride 2)
32 → 32  (stride 1)
32 → 64  (stride 2)
64 → 64  (stride 1)
64 → 128 (stride 2)
128 → 128 (stride 1)
```

There is no aggressive early pooling. Early downsampling would average away the
weak LSB signal.

**Head**

```text
AdaptiveAvgPool2d(1) → Flatten → Dropout(0.2) → Linear(128 → 1)
```

Output is a single logit; `sigmoid` converts it to `p(stego)`.

Trainable parameters: **688,497**.

### 12.2 Why this design

- High-pass residuals expose the high-frequency irregularities that LSB
  embedding introduces.
- Freezing that first layer prevents the network from simply learning the
  dataset's image content.
- No early pooling preserves spatial detail.
- Global average pooling produces a compact classifier head.
- The model is small enough to train on a 6 GB laptop GPU.

### 12.3 Data loading and augmentation

`StegoDataset`:

- For `in_channels == 3`, loads RGB and returns `(3, H, W)`.
- For `in_channels == 1`, loads grayscale and returns `(1, H, W)`.
- Scales to `[0, 1]`.
- Training: random crop (if source is larger than target) + horizontal flip.
- Validation/test: deterministic center crop or pad.

DataLoader:

- shuffle only for training
- `pin_memory` when CUDA is available
- persistent workers when `num_workers > 0`
- batch size / workers controlled by CLI/config

At the actual setting (`128×128` images, `--img-size 128`), the random crop does
nothing; the effective augmentation is horizontal flip.

### 12.4 Training protocol actually used

The final GPU training command was:

```bash
python3 -m src.train_cnn \
  --manifest data/manifest.csv \
  --out artifacts/cnn \
  --arch srnet_lite \
  --epochs 60 \
  --img-size 128 \
  --batch-size 64 \
  --lr 0.002 \
  --num-workers 8 \
  --patience 10 \
  --device cuda
```

Loss and optimization:

- `BCEWithLogitsLoss` with `pos_weight = neg / pos` from the training split
  (`800 / 1600 = 0.5`).
- AdamW, `weight_decay = 1e-4`.
- Cosine annealing over the total number of steps.
- AMP (mixed precision) enabled on CUDA:
  `torch.autocast(device_type="cuda", dtype=torch.float16)` plus
  `torch.amp.GradScaler`.
- Early stopping on **validation ROC-AUC** with patience 10.
- The best checkpoint is saved whenever validation AUC improves.

Checkpoint contents:

```python
{
    "arch": "srnet_lite",
    "state_dict": ...,
    "image_size": 128,
    "in_channels": 3,
    "epoch": 51,
    "val_auc": 0.99065,
    "val_loss": ...,
}
```

### 12.5 What happened during the run

The GPU run was interrupted near the end of the 60-epoch schedule, after the
epoch-51 checkpoint had already been written. That checkpoint had:

- validation AUC **0.99065**
- validation loss ≈ **0.0900**
- 51 epochs completed

Rather than throw away a strong checkpoint, the project retained `best.pt`,
regenerated validation/test metrics with `src/evaluate.py`, and corrected
`artifacts/cnn/meta.json` with:

```json
{
  "best_val_auc": 0.9907,
  "best_epoch": 51,
  "n_train": 2400,
  "dataset": "HF emirozturk/LSBStegoDataset 128x128_en (2 variants/cover)"
}
```

This is why there are no `history.json` / `history.csv` / `history.png` files for
the final CNN: the run was interrupted before those were written, and the stale
history from the earlier CPU attempt was removed. A fresh full retrain would
regenerate them.

### 12.6 Earlier CNN attempts

- An early CPU-only run at 128×128 produced a test AUC of only **0.549** and
  validation AUC ~0.5535; training loss was stuck around 0.46. That model was
  effectively random and was replaced.
- A 64×64 experiment was aborted and not used.
- The trained, retained model is the GPU run described above.

This is a useful caution: a small CNN can look completely broken if it is
undertrained or run at an inadequate budget. The same architecture reached
~0.984 test AUC with proper GPU training.

### 12.7 Transfer-learning alternatives

`src/models/transfer.py` supports ImageNet-pretrained:

- `resnet18`
- `efficientnet_v2_s`

They are selectable with `--arch`, and the factory handles them. They were not
the final chosen model. If used with grayscale input, `train_cnn.py`/inference
repeats the channel to 3; for RGB they use the natural 3-channel path.

---

## 13. Ensemble

Source: `src/predict.py`.

The ensemble is deliberately simple:

```python
prob = mean(probabilities of available models)
```

- If both models are present, `p = (p_classical + p_cnn) / 2`.
- If one model is missing or errors, the other is still used; the failure is
  recorded in `errors`.
- If neither can run, inference raises an error.
- The decision is `p >= threshold` (default 0.5).

Smoke test on one held-out pair:

| Image | Ensemble p(stego) | Verdict | Note |
|---|---:|---|---|
| clean test image | 0.2931 | CLEAN | CNN alone was 0.5547 on this image |
| corresponding stego | 0.9920 | STEGO | both models strongly agree |

That example illustrates the value of the ensemble: the CNN was borderline on one
clean image, and the classical model pulled the average below threshold.

The ensemble was not separately tuned or given a saved metrics file. It is a
convenience combination, not a third trained model.

---

## 14. Evaluation methodology

Source: `src/utils.py`, `src/evaluate.py`.

For each split the code computes:

- **Accuracy**: `(TP + TN) / N`.
- **Precision**: `TP / (TP + FP)` — how many flagged images are actually stego.
- **Recall / TPR**: `TP / (TP + FN)` — how many stego images are caught.
- **F1**: harmonic mean of precision and recall.
- **Confusion matrix**: rows actual `[clean, stego]`, columns predicted.
- **ROC-AUC**: threshold-independent ranking quality.
- **FPR at 95% TPR**: the false-positive rate when recall first reaches 95%.
  This is often more operationally meaningful than accuracy.
- **Threshold at target TPR**: the probability threshold corresponding to that
  operating point.

The FPR@95 calculation uses `sklearn.metrics.roc_curve` and
`np.searchsorted(tpr, 0.95)`: the first threshold where TPR is at least the
target.

Evaluation commands:

```bash
python3 -m src.evaluate --model classical --artifacts artifacts/classical --split test
python3 -m src.evaluate --model cnn --artifacts artifacts/cnn --split test --device cuda
```

`evaluate.py` loads the saved model and metadata, reconstructs the correct input
configuration, recomputes metrics, and overwrites the split's metrics/plots. It
does not change the model weights.

---

## 15. Web UI

Source: `app/streamlit_app.py`.

Run:

```bash
streamlit run app/streamlit_app.py
```

or the headless server form:

```bash
streamlit run app/streamlit_app.py \
  --server.headless true \
  --server.port 8501 \
  --browser.gatherUsageStats false
```

Health endpoint:

```text
http://localhost:8501/_stcore/health
```

### 15.1 Sidebar

- **Detector**: ensemble / classical / CNN.
- **Decision threshold**: slider from 0 to 1, default 0.5.
- **Device**: CUDA or CPU; only affects the CNN.
- **Model status**: checks `artifacts/classical/model.joblib` and
  `artifacts/cnn/best.pt`; shows ✅/❌.
- If no model exists, a warning tells the user to train one.
- A caption states the known-embedder limitation.
- If CUDA is requested but unavailable, an error is shown.

### 15.2 Detect tab

1. Upload an image.
2. The app saves the upload to a temporary file and loads it.
3. It shows the image and an upscaled LSB-plane visualization.
4. It runs the selected detector.
5. It reports:
   - verdict (`CLEAN` / `STEGO`),
   - `p(stego)`,
   - threshold,
   - a progress bar,
   - a red/green result banner.
6. It shows per-model probabilities.
7. It shows any skipped-model errors.
8. Expanders show:
   - **Statistical analysis**: global chi² p-value, block-mean chi² p-value,
     RS score, estimated rate, SPA close-pair ratio, SPA same-parity ratio.
   - **LSB-plane features**: the six LSB statistics.

For color images, the statistical/LSB panel values shown by `predict_image` are
computed on the **green channel** (`rgb[:, :, 1]`). The image display uses the
grayscale load; the model scoring uses RGB where the model was trained for RGB.

### 15.3 Batch tab

- Multi-file upload.
- “Run batch” button.
- Per-file rows: filename, verdict, `p_stego`, chi² p-value, RS score.
- Progress bar.
- CSV download of the results.

### 15.4 About tab

Explains the two detectors, the ensemble, how to interpret `p(stego)` and the
chi² p-value, and the limitations (known embedder, JPEG recompression, adaptive
schemes). It also renders the saved test metrics JSON for each available model.

### 15.5 Upload formats (current UI behavior)

The uploader whitelist is currently:

```text
png, bmp, tif, tiff, pgm, ppm, webp
```

The core loader also supports `jpg` / `jpeg` (`src/imageio.py`), but the Streamlit
file pickers do not expose them. This was an intentionally cautious, lossless-only
UI restriction: JPEG is lossy, and recompression can destroy the LSB payload.
It is not a limitation of the detector code itself; a JPEG can be scored through
the Python API/CLI, but results are less reliable if the file was recompressed
after embedding.

---

## 16. CLI and Python API

### 16.1 Prediction CLI

```bash
python3 -m src.predict photo.png
python3 -m src.predict photo.png --model classical
python3 -m src.predict photo.png --model cnn --device cuda
python3 -m src.predict a.png b.png --json
python3 -m src.predict photo.png --threshold 0.3
```

Output includes the label, ensemble probability, threshold, and per-model
probabilities.

### 16.2 Python API

```python
from src.predict import predict_image, predict_gray, predict_paths

res = predict_image("photo.png", model="ensemble", threshold=0.5)
print(res["label"])          # "clean" or "stego"
print(res["probability"])    # ensemble p(stego)
print(res["per_model"])      # {"classical": ..., "cnn": ...}
print(res["statistical"])    # chi-square / RS / SPA
print(res["lsb"])            # LSB-plane features
```

`predict_image()` loads both grayscale and RGB. `predict_gray()` accepts a uint8
array plus an optional `rgb` argument, which is useful for custom pipelines.
Artifacts are cached in-process, so repeated calls do not reload the models.

### 16.3 Training CLI

```bash
python3 -m src.train_classical
python3 -m src.train_classical --model svm
python3 -m src.train_classical --limit-per-class 3000
python3 -m src.train_cnn
python3 -m src.train_cnn --arch efficientnet_v2_s
python3 -m src.train_cnn --limit-per-class 3000 --epochs 15
```

### 16.4 Data / diagnostics CLI

```bash
python3 -m src.data.download_check
python3 -m src.data.split --manifest data/manifest.csv
python3 scripts/check_signal.py --split train --n 300
python3 scripts/make_synthetic_dataset.py
```

---

## 17. Configuration reference

`configs/default.yaml`:

| Key | Value | Meaning |
|---|---|---|
| `seed` | 1337 | global random seed |
| `paths.raw_dir` | `data/raw` | image root |
| `paths.features_dir` | `data/features` | cached feature `.npz` |
| `paths.artifacts_dir` | `artifacts` | model outputs |
| `paths.manifest` | `data/manifest.csv` | dataset index |
| `data.splits` | train/val/test | expected splits |
| `data.classes` | clean/stego | class folders |
| `data.grayscale` | true | legacy setting; actual feature config uses `features.color` |
| `data.image_size` | 256 | default image size for CNN/caches |
| `data.limit_per_class` | null | optional cap |
| `features.color` | true | **critical** per-channel RGB features |
| `features.spam.threshold` | 3 | SPAM clipping T |
| `features.srm.threshold` | 2 | SRM quantization T |
| `features.lsb.include_autocorr` | true | LSB autocorrelation features |
| `classical.model` | xgboost | chosen model |
| `classical.n_folds` | 5 | config field; current trainer uses the fixed split |
| `classical.n_jobs` | -1 | parallel feature extraction |
| `classical.xgboost.*` | 400 trees, depth 6, lr 0.08, hist, AUC | XGBoost params |
| `classical.svm.*` | C=5, gamma=scale | SVM alternative |
| `cnn.arch` | srnet_lite | default CNN |
| `cnn.epochs` | 40 | default; final run used 60 |
| `cnn.batch_size` | 32 | default; final run used 64 |
| `cnn.learning_rate` | 0.001 | default; final run used 0.002 |
| `cnn.weight_decay` | 0.0001 | AdamW weight decay |
| `cnn.num_workers` | 4 | default; final run used 8 |
| `cnn.amp` | true | mixed precision |
| `cnn.patience` | 8 | default; final run used 10 |
| `cnn.image_size` | 256 | default; final run used 128 |
| `cnn.in_channels` | 3 | RGB |
| `cnn.augment` | true | crop + flip |
| `eval.threshold` | 0.5 | decision threshold |
| `eval.target_tpr` | 0.95 | FPR@95 reporting |

The config loader supports deep-merging an optional user YAML and dotted
overrides such as `load_config(cnn__epochs=10)`.

---

## 18. Repository map

```text
configs/
  default.yaml                 tunables

data/
  manifest.csv                 generated index of 3,000 images
  raw/{train,val,test}/{clean,stego}/
  features/                    cached feature matrices

artifacts/
  classical/                   XGBoost model + metrics + plots
  cnn/                         SRNet-lite checkpoint + metrics + plots

src/
  config.py                    config loading, root resolution
  utils.py                     seed, metrics, JSON, plots
  imageio.py                   lossless loading, crop/pad
  data/
    download_check.py          scan, group ids, hashes, manifest
    split.py                   group-aware regroup
    dataset.py                 PyTorch dataset / dataloaders
  features/
    statistical.py             chi-square, RS, SPA
    spam.py                    SPAM features
    srm.py                     SRM-style residuals
    lsb.py                     LSB-plane statistics
    extract.py                 combine gray / per-channel RGB vectors
  models/
    classical.py               XGBoost / SVM train, save, load
    cnn.py                     SRNet-lite
    transfer.py                ResNet-18 / EfficientNet-V2-S
    factory.py                 model factory
  train_classical.py           feature extraction + XGBoost training
  train_cnn.py                 CNN training loop + evaluation
  evaluate.py                  standalone evaluation
  predict.py                   inference API + CLI

app/
  streamlit_app.py             web UI

scripts/
  prepare_hf_stego.py          pair HF covers/stego into data/raw
  check_signal.py              chi-square/RS/SPA separation report
  make_synthetic_dataset.py    no-download smoke-test dataset

tests/
  conftest.py                  synthetic image/dataset fixtures
  test_features.py             feature shapes, LSB/RS/chi² behavior
  test_data.py                 group ids, manifest counts, leakage
  test_classical_pipeline.py   XGBoost learns synthetic LSB
  test_cnn.py                  SRNet-lite shapes, frozen layer, RGB loader
  test_predict.py              saved-artifact inference
  test_app.py                  Streamlit AppTest

Makefile                       setup / data / train / eval / app / test
requirements.txt               Python dependencies
README.md                      original quick-start
PROJECT.md                     this document
```

`data/raw/`, `data/features/`, `data/manifest*.csv`, and `artifacts/` are
gitignored. The trained models and dataset are local, not committed.

---

## 19. Tests

Run:

```bash
python3 -m pytest -q
```

Current suite: **18 tests**, all passing.

Coverage by area:

| File | What it verifies |
|---|---|
| `test_features.py` | vector shapes/finiteness, LSB asymmetry drops after embedding, RS score rises, chi-square behavior, size invariance, RGB stacking |
| `test_data.py` | cover/stego group-id derivation, HF variant naming, manifest counts, no group leakage, `regroup` keeps groups together |
| `test_classical_pipeline.py` | XGBoost can learn synthetic LSB data (ROC-AUC > 0.8) |
| `test_cnn.py` | SRNet-lite forward shape, frozen high-pass layer, ResNet-18 factory, RGB input, RGB dataloader |
| `test_predict.py` | saved classical artifact inference returns valid probabilities and stego ≥ clean on synthetic data |
| `test_app.py` | Streamlit AppTest executes the app with no exceptions and renders the title |

The suite uses a synthetic dataset generated in `tests/conftest.py`:

- 128×128 synthetic covers with cover-like LSB correlation
- stego created by randomizing LSBs over 50% of pixels
- 40 train / 12 val / 12 test pairs per class
- `build_manifest(..., quick=True)` writes a temporary manifest

So the tests do not require the real HF dataset or a GPU. CNN tests are skipped if
PyTorch is absent.

---

## 20. Artifacts in detail

### 20.1 `artifacts/classical/`

- `model.joblib` — 452 KB XGBoost model.
- `meta.json` — inference contract: feature dim 2,976, color=true, SPAM T=3,
  SRM T=2, threshold 0.5.
- `metrics_{val,test}.json` — full metrics including ROC arrays and confusion
  matrix.
- `confusion_{val,test}.png`, `roc_{val,test}.png`.
- `feature_importance.json` — top 40 features.

### 20.2 `artifacts/cnn/`

- `best.pt` — 2.7 MB checkpoint with architecture, state dict, image size,
  input channels, best epoch, val AUC, val loss.
- `meta.json` — inference contract: `srnet_lite`, 128×128, 3 channels, 688,497
  parameters, threshold 0.5, best val AUC 0.9907, best epoch 51.
- `metrics_{val,test}.json` — regenerated from `best.pt` after the interrupted
  run.
- `confusion_{val,test}.png`, `roc_{val,test}.png`.
- No `history.*` files for the final run (see §12.5).

### 20.3 Loading contract

`src/predict.py` reads the metadata to decide:

- classical: whether to use RGB features and which SPAM/SRM thresholds;
- CNN: architecture, input size, and input channels from the checkpoint/meta.

This is why inference reproduces the training configuration without hardcoding
2,976 or 128 in the prediction path.

---

## 21. Reproduction runbook

### 21.1 Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The actual project used Python 3.14.7 and a user-site install. For PyTorch on
Python 3.14, install the CUDA wheel explicitly:

```bash
python3 -m pip install --user --force-reinstall \
  torch==2.14.1+cu130 torchvision==0.29.1+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
nvidia-smi
```

Inference works without CUDA. Training the CNN on CPU is possible but was far too
slow and produced a poor model in the earlier attempt.

### 21.2 Data

> The original downloads/extractions were in `/tmp`, which is tmpfs and was
> cleared on reboot. `data/raw/` and `data/manifest.csv` persist. To rebuild from
> scratch, re-download the tars.

```bash
cd /tmp
curl -L -O https://huggingface.co/datasets/emirozturk/LSBStegoDataset/resolve/main/128x128.tar
curl -L -O https://huggingface.co/datasets/emirozturk/LSBStegoDataset/resolve/main/128x128_en.tar

mkdir -p /tmp/stego_hf/covers /tmp/stego_hf/en
tar -xf 128x128.tar    -C /tmp/stego_hf/covers
tar -xf 128x128_en.tar -C /tmp/stego_hf/en
```

From the repo root:

```bash
python3 scripts/prepare_hf_stego.py \
  --covers-dir /tmp/stego_hf/covers/128x128 \
  --stego-dir  /tmp/stego_hf/en/128x128_en \
  --out data/raw \
  --variants 2 \
  --max-covers 1000 \
  --overwrite
```

Build and validate the manifest:

```bash
python3 -m src.data.download_check
```

Expected output: 3,000 rows, 1,000 groups, no groups across splits, no duplicate
content across splits.

Optional leakage repair if using a different dataset:

```bash
python3 -m src.data.split --manifest data/manifest.csv
```

### 21.3 Signal sanity check

```bash
python3 scripts/check_signal.py --split train --n 300
```

Look for feature AUCs well above 0.5. If chi-square/RS show no separation, the
labels or the embedding are wrong; do not train yet.

### 21.4 Train classical

```bash
python3 -m src.train_classical \
  --manifest data/manifest.csv \
  --out artifacts/classical
```

Feature extraction is cached under `data/features/`. To force recomputation:

```bash
python3 -m src.train_classical --no-cache
```

### 21.5 Train CNN

GPU form actually used:

```bash
python3 -m src.train_cnn \
  --manifest data/manifest.csv \
  --out artifacts/cnn \
  --arch srnet_lite \
  --epochs 60 \
  --img-size 128 \
  --batch-size 64 \
  --lr 0.002 \
  --num-workers 8 \
  --patience 10 \
  --device cuda
```

CPU fallback:

```bash
python3 -m src.train_cnn --device cpu --epochs 10 --batch-size 32
```

A full GPU run at this data size is on the order of a few minutes, not hours.
The earlier 51-epoch run reached ~5 s/epoch including validation.

### 21.6 Evaluate

```bash
python3 -m src.evaluate --model classical --artifacts artifacts/classical --split val
python3 -m src.evaluate --model classical --artifacts artifacts/classical --split test

python3 -m src.evaluate --model cnn --artifacts artifacts/cnn --split val --device cuda
python3 -m src.evaluate --model cnn --artifacts artifacts/cnn --split test --device cuda
```

### 21.7 Predict

```bash
python3 -m src.predict data/raw/test/clean/1.png --model ensemble
python3 -m src.predict data/raw/test/stego/1__v0.png --model ensemble --json
```

### 21.8 Run the webapp

```bash
streamlit run app/streamlit_app.py \
  --server.headless true \
  --server.port 8501 \
  --browser.gatherUsageStats false
```

Then open `http://localhost:8501`.

### 21.9 Makefile shortcuts

```bash
make help
make synthetic        # tiny no-download dataset
make check-data       # build/validate manifest
make check-signal     # chi-square/RS/SPA report
make train-classical
make train-cnn
make eval-classical
make eval-cnn
make app
make test
```

---

## 22. Design decisions and lessons learned

### 22.1 Color, not grayscale — the decisive choice

Grayscale conversion mixed the channel bit-planes and reduced RS separation from
~0.94 to ~0.51. All subsequent work used per-channel RGB features and a 3-channel
CNN. This is the single most important “why” in the project.

### 22.2 No resize, no lossy recompression

LSB information lives in the lowest bit. Any interpolation or JPEG-style
quantization can erase it. The pipeline:

- loads with `IMREAD_UNCHANGED`;
- never resizes in feature extraction;
- uses crops, not resizes, for CNN preprocessing;
- evaluates on the native 128×128 images.

### 22.3 Group-aware splitting

A cover and all of its stego variants share `group_id`. Splits are assigned by
group, so the same source image cannot appear in train and test. The manifest
reports `groups_across_splits = 0`. Without this, reported accuracy would be
inflated by memorization.

### 22.4 Frozen high-pass first layer

The SRNet-lite starts with fixed SRM-like residual filters. This biases the
network toward steganalysis-relevant high-frequency structure and prevents it
from wasting capacity learning arbitrary low-level filters.

### 22.5 No early pooling

The LSB signal is weak and high-frequency. Aggressive downsampling early in the
network would average it away. The architecture delays stride-2 blocks and uses
global average pooling only at the end.

### 22.6 Feature caching

SPAM/SRM extraction is the slow part of the classical pipeline. Caching per split
with a path-list check makes retraining and evaluation fast and avoids silently
using stale features after a dataset change.

### 22.7 Class weighting

The dataset is 1:2 clean:stego. Both models use negative/positive weighting so
that the majority class does not dominate:

- XGBoost: `scale_pos_weight = neg / pos = 0.5`.
- CNN: `BCEWithLogitsLoss(pos_weight = neg / pos = 0.5)`.

### 22.8 AUC and FPR@95 over raw accuracy

Accuracy at 0.5 can hide a bad operating point. FPR@95 answers the practical
question: “if I want to catch 95% of stego images, how many clean images do I
wrongly flag?” The project reports both.

### 22.9 Two model families, not one

The classical model and the CNN make different errors. The classical model had
better test precision; the CNN had better validation AUC/FPR@95. Averaging them
is cheap and corrected at least one borderline CNN false positive. A learned
stacking model was not attempted.

### 22.10 Bugs and gotchas encountered

- **High-pass initialization broadcast bug**: the first implementation tried to
  copy a `(n*C, 3, 3)` tensor into a `(n*C, 1, 3, 3)` convolution weight and hit
  a broadcast error. Fixed with `.unsqueeze(1)` so the filter bank has the
  correct per-channel shape.
- **Grayscale signal loss**: see §6.
- **CPU CNN looked random**: the first 128×128 CPU run had test AUC 0.549. The
  GPU run with a proper budget reached 0.984. Do not judge the architecture from
  an undertrained run.
- **Interrupted GPU run**: the run was aborted after epoch 51, but `best.pt` was
  already written. The checkpoint was retained and metrics were regenerated
  instead of retraining from scratch.
- **Buffered training output**: piping training output through `grep` made it
  look like nothing was happening. For long runs, use a log file or line-buffered
  output.
- **`/tmp` is tmpfs**: extracted datasets and downloaded tars disappeared on
  reboot. `data/raw` is the durable copy.
- **Driver/library mismatch**: the NVIDIA userspace library and kernel driver can
  drift apart after updates, making `nvidia-smi` and CUDA fail even though the
  device nodes exist. A reboot/driver realignment is the usual fix.
- **UI uploader whitelist**: the core supports JPEG, but the Streamlit uploader
  lists only lossless formats. This was a cautious guard, not a detector
  limitation.

### 22.11 Unused / optional paths

- SVM classical model: implemented, not final.
- ResNet-18 / EfficientNet-V2-S transfer models: implemented, not final.
- Kaggle dataset path: documented, not used.
- `n_folds` config: present, but the trainer uses a fixed split rather than
  k-fold cross-validation.
- Ensemble metrics file: not produced; the ensemble is a runtime average.

---

## 23. Limitations and honest operating envelope

1. **Known embedder only.** The models were trained on the variants in this
   dataset. Adaptive schemes (S-UNIWARD, HILL, MiPOD) and unseen embedders may
   evade them.
2. **JPEG recompression destroys the payload.** A JPEG that was re-encoded after
   embedding may be impossible to detect. Lossless formats are the reliable path.
3. **Small, single-resolution dataset.** 1,000 covers, 128×128, two variants per
   cover, one seed, one split. This is enough for a strong demo and a solid
   baseline, not a production forensic system.
4. **Class imbalance.** The dataset is 1:2 clean:stego. Metrics are reported on a
   test set with the same imbalance. Different priors will change precision.
5. **Threshold calibration.** `p(stego)` is not rigorously calibrated. The
   default 0.5 is a convenience. For a real deployment, choose a threshold from
   the ROC based on the acceptable false-positive rate.
6. **FPR@95 ≈ 9–10%.** At 95% recall, expect roughly 9–10 false alarms per 100
   clean images on this distribution. That is the practical cost of the current
   operating point.
7. **No payload extraction.** The system says clean/stego; it does not recover
   the message, estimate its length, or localize it.
8. **Green-channel statistics in the UI.** The statistical/LSB panel for color
   images is computed on the green channel for display. The models themselves use
   RGB where trained.
9. **JPEG upload support is absent in the UI.** The core loader and CLI support
   JPEG, but the Streamlit pickers restrict to lossless formats.
10. **No cross-validation or hyperparameter search.** The reported numbers are
    from a single group-aware split. A different split would move them somewhat.
11. **No adversarial robustness testing.** The detector is not evaluated against
    deliberate evasion, cover-source mismatch, or post-processing.
12. **GPU stack fragility.** The models are saved and portable, but the host's
    NVIDIA driver/library state can affect GPU inference/training until rebooted
    or repaired.

---

## 24. Current operational state

- `data/manifest.csv`: 3,000 rows, 1,000 groups, 0 leakage.
- `data/raw/`: 3,000 images (~101.5 MB).
- `artifacts/classical/`: trained XGBoost + metrics + plots present.
- `artifacts/cnn/`: trained `best.pt` (epoch 51, val AUC 0.9907) + regenerated
  metrics + plots present.
- Test suite: 18 tests pass.
- Streamlit app: verified earlier with `AppTest` (no exceptions) and an HTTP
  health check returning 200. At the time of writing, no Streamlit process was
  listening on port 8501; start it with the command in §15 or §21.8.
- GPU: training-time results above were produced on the RTX 3050. The current
  host state has a driver/library mismatch; see §4. Classical inference is
  unaffected; CNN can run on CPU.

Quick start if the artifacts are present:

```bash
streamlit run app/streamlit_app.py \
  --server.headless true \
  --server.port 8501
```

Then open `http://localhost:8501`.

---

## 25. Glossary

| Term | Meaning |
|---|---|
| **LSB** | Least significant bit; the lowest bit of each pixel value |
| **LSB replacement** | Replace the LSB with a payload bit; changes pixel values by at most ±1 |
| **Cover** | Original, unmodified image |
| **Stego** | Image containing an embedded payload |
| **Steganalysis** | Detecting hidden data |
| **SPAM** | Subtractive Pixel Adjacency Model; Markov features of pixel differences |
| **SRM** | Spatial Rich Model; high-pass residual + co-occurrence features |
| **RS analysis** | Regular/Singular group analysis for LSB detection |
| **SPA** | Sample-pair analysis |
| **Chi-square attack** | Even/odd histogram equalization test |
| **AUC** | Area under the ROC curve; 0.5 = random, 1.0 = perfect ranking |
| **FPR@95** | False-positive rate when true-positive rate is at least 95% |
| **AMP** | Automatic mixed precision (FP16 compute on CUDA) |
| **Group-aware split** | Split by cover identity so variants cannot leak across splits |
| **Known-embedder detector** | A detector trained against a specific embedding scheme, not a universal one |

---

## 26. One-paragraph summary

This project built a real LSB steganography detector from a real Hugging Face
dataset rather than synthetic data. The decisive insight was that grayscale
conversion destroys the LSB signal, so the entire pipeline was rebuilt around
per-channel RGB features. A 2,976-dimensional SPAM+SRM+LSB representation with
XGBoost reached **0.9845 test ROC-AUC** and **0.9467 accuracy**, while a compact
SRNet-lite CNN with a frozen SRM high-pass layer reached **0.9836 test ROC-AUC**
and **0.9133 accuracy**. Group-aware splitting prevented cover leakage, feature
caching made retraining fast, and both models plus a Streamlit UI were delivered.
The system is a strong known-embedder baseline; it is not a universal or
forensic-grade detector, and it should be used on lossless images without
resizing or JPEG recompression.
