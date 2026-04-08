# NeuroScan Architecture

This document describes the current state of the brain-tumor-classifier system as of an audit on 2026-04-08. It is a snapshot of *what is*, not *what should be*. See [Known Limitations](#known-limitations) for the gap list.

## System Overview

```
┌──────────────┐    HTTPS    ┌────────────────────┐    HTTPS    ┌──────────────────┐
│  Frontend    │  ────────▶  │  FastAPI Backend   │  ────────▶  │ Supabase (Postgres)│
│  (vanilla JS │             │  (Render, CPU)     │             │  predictions tbl  │
│   on Pages)  │  ◀────────  │                    │  ◀────────  │                  │
└──────────────┘             └─────────┬──────────┘             └──────────────────┘
                                       │
                                       ▼
                             ┌────────────────────┐
                             │  PyTorch models    │
                             │  Stage 1: 8-class  │
                             │  Stage 2: glioma   │
                             │           subtype  │
                             └────────────────────┘
```

The frontend is a static single-page app served from GitHub Pages. The backend is a FastAPI service running on Render's free CPU tier, holding two PyTorch models in memory. Predictions are persisted to Supabase Postgres.

## Model Architecture

### Two-stage cascade

Both stages are **EfficientNet-B0** (`torchvision.models.efficientnet_b0`) with a replaced classifier head:

```python
nn.Sequential(
    nn.Dropout(p=dropout),
    nn.Linear(1280, num_classes),
)
```

| Stage | Classes | Dropout | Weights file | Triggered when |
|-------|---------|---------|--------------|----------------|
| 1 | 8: `carcinoma, glioma, meningioma, neurocytoma, no_tumor, papilloma, pituitary, schwannoma` | 0.3 | `tumor_classifier.pth` | Always |
| 2 | 4: `astrocytoma, ependymoma, glioblastoma, oligodendroglioma` | 0.4 | `glioma_subtype_classifier.pth` | Stage 1 prediction == `glioma` |

Both are loaded once at FastAPI startup into module-level globals (`_model`, `_glioma_model`) in [backend/model.py](backend/model.py). Stage 2 is optional — if its weights file is missing, the backend logs a warning and disables subtype classification rather than failing to start.

Inference runs on CPU with `torch.no_grad()`.

### Class names

Stage-1 class names are hardcoded in [backend/model.py:16](backend/model.py#L16) in alphabetical order. They must match the index order produced by `torchvision.datasets.ImageFolder` at training time. There is no runtime assertion of this invariant.

The same list is duplicated in [frontend/app.js](frontend/app.js) (`CLASS_ORDER`, `TUMOR_INFO`, `formatClassName`) and must be kept in sync by hand.

## Data Flow

### Inference path (`POST /predict`)

1. **Receive upload** — [backend/main.py:50](backend/main.py#L50). Rejects non-image content types.
2. **Decode** — `PIL.Image.open(...).convert("RGB")`.
3. **MRI border crop** — [backend/model.py:63-86](backend/model.py#L63-L86). Converts to grayscale, masks rows/cols whose mean intensity exceeds 15, crops to the bounding box of the mask, pads 5px. Designed to strip black bezels around scanned MRI films.
4. **Resize** to 224×224 (bilinear).
5. **Normalize** with ImageNet mean/std.
6. **Forward pass** through stage 1 → softmax → argmax + full probability vector.
7. **(Conditional) Stage 2** — if predicted class is `glioma` and the subtype model is loaded, run a second forward pass on the same preprocessed tensor.
8. **Thumbnail** — re-decode the original bytes, downscale to 128×128, JPEG quality 70, base64.
9. **Persist** — insert into Supabase `predictions` table via [backend/database.py:16](backend/database.py#L16).
10. **Return** `PredictionResponse` (see [backend/schemas.py](backend/schemas.py)).

### Grad-CAM path (`POST /gradcam`)

Separate endpoint ([backend/main.py:92](backend/main.py#L92)). Re-decodes and re-preprocesses the image, then calls `generate_gradcam` ([backend/model.py:89](backend/model.py#L89)):

1. Register forward + full-backward hooks on `model.features[-1]` (the last conv block).
2. Forward pass with `requires_grad=True` on the input tensor.
3. Backprop the score of the predicted class.
4. Compute channel weights = mean of gradient over spatial dims.
5. Weighted sum of activations → ReLU → normalize to [0, 1].
6. Resize to 224×224, overlay on the denormalized input via `matplotlib.cm.jet` (0.55 input + 0.45 heatmap).
7. Render with matplotlib, save to a `BytesIO` PNG, base64-encode.

If the prediction is glioma, a second Grad-CAM is generated for the subtype model.

### Other endpoints

| Endpoint | Purpose | Source |
|---|---|---|
| `GET /health` | Liveness probe | [backend/main.py:45](backend/main.py#L45) |
| `GET /predictions?limit&offset` | Paginated history | [backend/main.py:107](backend/main.py#L107) |
| `GET /stats` | Aggregate stats (total, class distribution, avg confidence, avg latency) | [backend/main.py:115](backend/main.py#L115) |

`/stats` fetches every row from the `predictions` table and aggregates in Python. There is no `LIMIT`.

## Database Schema

Defined in [README.md:67-81](README.md#L67-L81):

```sql
CREATE TABLE predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz DEFAULT now(),
  predicted_class text NOT NULL,
  confidence float8 NOT NULL,
  all_confidences jsonb NOT NULL,
  thumbnail_base64 text,
  original_filename text,
  inference_time_ms float8
);
```

The backend additionally inserts `subtype`, `subtype_confidence`, and `subtype_confidences` when stage 2 fires ([backend/main.py:72-75](backend/main.py#L72-L75)) — **these columns are not in the README schema**, see Known Limitations.

Row-Level Security is enabled, but the policy is `USING (true) WITH CHECK (true)` — effectively unrestricted anonymous read/write.

## Training Configuration

Training was performed in Google Colab GPU notebooks. **Only the V1/V2 4-class notebook is committed to this repo as [training/train.ipynb](training/train.ipynb).** The 8-class stage-1 model and the glioma-subtype stage-2 model that are actually deployed were trained in additional Colab notebooks that are **not in the repo**. Their cells are recorded below for documentation, but the canonical source remains in Colab.

### Datasets used

| Key | Kaggle slug | Has split? | Purpose |
|---|---|---|---|
| `data1` | `masoudnickparvar/brain-tumor-mri-dataset` | yes (Training/Testing) | Original 4 classes (glioma, meningioma, notumor, pituitary) |
| `data2` | `briscdataset/brisc2025` (classification task) | yes (train/test) | Same 4 classes, second source |
| `data44` | `fernando2rad/brain-tumor-mri-images-44c` | no | 44 fine-grained tumor folders organized as `<TumorName> <Sequence>` (T1, T2, T1C+); used both to add rare classes and to build the glioma-subtype dataset |

A fourth Kaggle dataset, `sartajbhuvaji/brain-tumor-classification-mri`, was downloaded only to build a small "unseen" hand-test set (`unseen_test.zip`) and was not used for training.

### Stage-1 model: deployed `tumor_classifier.pth` (a.k.a. V5)

Built by combining all three datasets into an `expanded/` ImageFolder tree with 8 classes:

| Class | data1 | data2 (BRISC) | data44 source folders |
|---|---|---|---|
| `glioma` | ✓ | ✓ | Astrocitoma / Glioblastoma / Oligodendroglioma / Ependimoma / Ganglioglioma — each in T1, T2, T1C+ (15 folders) |
| `meningioma` | ✓ | ✓ | Meningioma T1 / T2 / T1C+ |
| `no_tumor` | ✓ | ✓ | _NORMAL T1 / T2 |
| `pituitary` | ✓ | — | — |
| `schwannoma` | — | — | Schwannoma T1 / T2 / T1C+ |
| `neurocytoma` | — | — | Neurocitoma T1 / T2 / T1C+ |
| `carcinoma` | — | — | Carcinoma T1 / T2 / T1C+ |
| `papilloma` | — | — | Papiloma T1 / T2 / T1C+ |

For data44 (which has no pre-defined split), each class's image list is shuffled with `random.seed(42)` and split 80/20 train/test at the **image** level.

Training:
- Architecture: EfficientNet-B0, pretrained ImageNet, **fully unfrozen**
- Head: `Dropout(0.3) → Linear(1280, 8)`
- Optimizer: `AdamW(lr=5e-5, weight_decay=0.01)`
- Schedule: `CosineAnnealingLR(T_max=20)`
- Loss: `CrossEntropyLoss` (no class weights)
- 20 epochs, batch size 32
- Augmentation: V2 pipeline (Resize 256 → RandomCrop 224 → H/V flip → ±20° rot → RandomAffine(translate 0.1, scale 0.9–1.1) → ColorJitter(0.3,0.3,0.2) → RandomGrayscale 0.1 → Normalize → RandomErasing 0.2)
- ImageFolder class index order (alphabetical): `[carcinoma, glioma, meningioma, neurocytoma, no_tumor, papilloma, pituitary, schwannoma]` — **must match** [backend/model.py:16](backend/model.py#L16)
- Saved to `tumor_classifier_v5.pth`, manually renamed to `tumor_classifier.pth` for deployment

### Stage-2 model: deployed `glioma_subtype_classifier.pth`

Built from data44 only. For each subtype, all three sequences (T1, T2, T1C+) are concatenated into a single class folder, then 80/20 train/test split with `random.seed(42)`:

```
glioma_subtypes/
├── train/
│   ├── astrocytoma/      ← Astrocitoma {T1,T2,T1C+}
│   ├── ependymoma/       ← Ependimoma {T1,T2,T1C+}
│   ├── glioblastoma/     ← Glioblastoma {T1,T2,T1C+}
│   └── oligodendroglioma/← Oligodendroglioma {T1,T2,T1C+}
└── test/  (same structure)
```

Training:
- Architecture: EfficientNet-B0, pretrained ImageNet, fully unfrozen
- Head: `Dropout(0.4) → Linear(1280, 4)`
- Optimizer: `AdamW(lr=5e-5, weight_decay=0.01)`
- Schedule: `CosineAnnealingLR(T_max=20)`
- 20 epochs, **batch size 16**
- Same V2 augmentation pipeline as stage 1
- ImageFolder class index order (alphabetical): `[astrocytoma, ependymoma, glioblastoma, oligodendroglioma]` — **must match** [backend/model.py:17](backend/model.py#L17)

### Earlier 4-class models (committed notebook)

The committed [training/train.ipynb](training/train.ipynb) trains a 4-class model on data1 only, in three sequential phases. None of these phases produced the deployed model — they pre-date the expansion to 8 classes — but the notebook is what the README still points users at.

| Phase | Epochs | Trainable | Optimizer | LR schedule | Final test acc |
|---|---|---|---|---|---|
| 1. Head only | 15 | 5,124 | Adam(1e-3) | ReduceLROnPlateau | 85.6% |
| 2. Last 2 blocks + head | 10 | partial | Adam(1e-4) | ReduceLROnPlateau | 92.6% |
| 3. Full fine-tune (V2) | 20 | all (~4M) | AdamW(5e-5, wd=0.01) | CosineAnnealingLR(T_max=20) | 95.4% best |

A V3 attempt added BRISC (data1 + data2) and trained 25 epochs full-unfreeze. A V4 added the four-class subset of data44 to the merged set. V5 (above) is the version actually deployed.

### Tracked metrics
- **Per epoch** (stdout only): train loss, train accuracy, test accuracy, learning rate
- **End of training (V1/V2 notebook only)**: `sklearn.metrics.classification_report`, confusion matrix heatmap
- **Per inference (production)**: predicted class, full confidence vector, inference time. Aggregated by `/stats` into total count, class distribution, average confidence, average latency

No per-class metrics across epochs, no AUC/ROC, no calibration metrics, no best-checkpoint saving (only final epoch), no `torch`/`numpy` random seed, no held-out test set distinct from validation.

## Frontend

Single-page vanilla JS app in [frontend/](frontend/) (and a duplicate copy in [docs/](docs/) for GitHub Pages). Three views: Classify, History, Stats.

- **Classify**: drag-and-drop or sample-button upload → `POST /predict` → render predicted class, confidence bars for all 8 classes, optional subtype section, "Show Grad-CAM" button that fires `POST /gradcam` lazily.
- **History**: paginated grid of past predictions from `GET /predictions`.
- **Stats**: counters and a class-distribution bar chart from `GET /stats`.

Backend URL is hardcoded in [frontend/app.js:5](frontend/app.js#L5).

## Deployment

| Component | Host | Notes |
|---|---|---|
| Backend | Render free tier | CPU-only PyTorch wheel, cold start ~30s |
| Frontend | GitHub Pages | Served from `frontend/` (or `docs/`) on `main` |
| Database | Supabase free tier | RLS enabled but unrestricted |
| Model weights | Committed to repo (`backend/*.pth`) | ~20 MB stage 1, additional file for stage 2 |

The frontend has a `checkServer()` retry loop ([frontend/app.js:108](frontend/app.js#L108)) that polls `/health` for up to 60s on page load to handle Render cold starts.

## Known Limitations

This list is the audit's findings. None of these have been fixed; this document records the current state.

### Methodological problems with the deployed models

These are the most credibility-damaging issues for an ML class project. They are all consequences of how the V5 stage-1 and the glioma-subtype datasets were assembled.

- **Patient-level data leakage.** Both data44 and (likely) BRISC contain multiple slices per patient. The 80/20 splits in V5 and the glioma-subtype model are random at the **image** level via `random.shuffle`, so slices from the same patient land in both train and test. Reported test accuracies are overstated by an unknown amount. A proper fix requires patient IDs and `GroupShuffleSplit`.
- **Source confound for the four data44-only classes.** `schwannoma`, `neurocytoma`, `carcinoma`, and `papilloma` exist **only in data44**, while their negatives (`glioma`, `meningioma`, `no_tumor`, `pituitary`) are sourced mostly from data1 and BRISC. The model can solve these classes by recognizing data44's scanner/contrast/file-format signature rather than the tumor itself. The high V5 test accuracy on these classes is therefore not trustworthy.
- **Severe class imbalance.** `pituitary` is sourced **only from data1** (~1,400 train images), while `glioma` is sourced from data1 + BRISC + 15 data44 folders and is several times larger. Training uses no class weights, no `WeightedRandomSampler`, no balanced loss. Loss and accuracy are dominated by majority classes.
- **Sample MRIs in [frontend/assets/sample_mris/](frontend/assets/sample_mris/) may have been in training.** The `v5_test.zip` and `new_samples.zip` collection cells grab `imgs[-3:]` / `imgs[0]` from each data44 folder, claiming these are "least likely in the 80% train split." This is wrong: the train/test split was performed by `random.shuffle` after `random.seed(42)`, so filesystem-order images have no relationship to the shuffled boundary. The samples shipped to users may be straight out of the training set.

### Reproducibility & training

- **Most of the training pipeline is not in the repo.** The committed [training/train.ipynb](training/train.ipynb) only trains the legacy 4-class model on data1. The deployed V5 stage-1 model and the glioma-subtype model live in separate Colab notebooks. The cells exist (recorded above and in chat history) but are not version-controlled.
- **Test set used as validation set** in every training script — the LR scheduler steps on test loss/accuracy, "best epoch" is reported on the same set. No held-out test exists for any model.
- **No best-checkpoint saving.** All training scripts save only the final epoch's weights, even when test accuracy peaks earlier.
- **No `torch` / `numpy` random seed** in any training script. Only the data44 file-list shuffle is seeded (`random.seed(42)`), so training runs are not reproducible.
- **Class index order is enforced by alphabetical convention only.** Both stages depend on `ImageFolder` producing classes in the same order as the hardcoded lists in [backend/model.py:16-17](backend/model.py#L16-L17). There is no runtime assertion or saved label map.
- **Naming drift.** The notebook saves `tumor_classifier_v2.pth` (later `_v5.pth`) and the backend expects `tumor_classifier.pth`. A manual rename is required for every redeploy and is undocumented.
- **Same data44 images used in both models.** Stage 1 (V5) and the glioma-subtype model both pull from data44 using the same `random.seed(42)`. The same image can appear in stage 1's train set as `glioma` and in stage 2's train set as e.g. `glioblastoma`. Not incorrect for a cascade, but means an end-to-end held-out evaluation would have to be carefully constructed.

### Security / secrets

- **Two Kaggle API tokens were previously leaked** across the training notebooks (values redacted). One had been committed to [training/train.ipynb](training/train.ipynb); the other lived only in Colab cells that were never committed. Both have since been rotated; the committed token was scrubbed from git history via `git filter-repo` and force-pushed. Going forward, all training notebooks read the token from the Colab Secrets store via `google.colab.userdata.get("KAGGLE_API_TOKEN")` rather than hardcoding it.

### Schema and data integrity
- **Subtype columns missing from the README's `CREATE TABLE`.** [backend/main.py:72-75](backend/main.py#L72-L75) inserts `subtype`, `subtype_confidence`, `subtype_confidences` when stage 2 fires, but the README's schema does not declare them. A fresh deploy will throw on the first glioma prediction.
- **`original_filename` is never inserted** by [backend/main.py:64-70](backend/main.py#L64-L70), even though it exists in the schema. History rows always show `null`.

### Correctness risks
- **No assertion that `CLASS_NAMES` order matches the training `ImageFolder.classes` order.** Silent label-permutation bug if class folders are renamed at retrain time.
- **No `state_dict` shape validation** when loading weights. Mismatched `num_classes` will fail loudly at best, silently mispredict at worst.
- **Grad-CAM hooks use module-level closure-captured lists** ([backend/model.py:93-100](backend/model.py#L93-L100)). Concurrent `/gradcam` requests will race on the shared activations/gradients buffers and on `model.zero_grad()`. There is no concurrency control.

### Architecture and extensibility
- **Module-level model globals.** No `Classifier` abstraction; adding a stage means another global and another `if` branch in `predict()`.
- **`predict()` and `predict_with_gradcam()` duplicate** decode + preprocess + forward.
- **No configuration file.** Image size, normalization stats, class lists, dropout rates, model paths, thumbnail size, JPEG quality, second-opinion thresholds, CORS origins, and API URL are all hardcoded across multiple files.
- **Class names live in three places**: `backend/model.py`, `frontend/app.js`, and the training notebook. They must be kept in sync by hand.
- **Frontend duplicated** in `frontend/` and `docs/`. They will drift.

### Performance
- **`get_stats()` fetches every row** from `predictions` with no `LIMIT` and aggregates in Python. O(n) per request.
- **`make_thumbnail_base64` re-decodes** the original bytes that `predict()` already decoded.
- **Matplotlib is loaded just for the Grad-CAM overlay.** Heavy dependency for one image composite that PIL could handle.

### Security
- **Hardcoded Kaggle API token in [training/train.ipynb](training/train.ipynb) cell-3.** Must be rotated and the notebook scrubbed.
- **No auth, no rate limiting** on any backend endpoint.
- **Supabase RLS policy is `USING (true) WITH CHECK (true)`** — anonymous unrestricted read/write to the predictions table.
- **CORS `allow_origins` includes the backend's own Render URL** ([backend/main.py:25](backend/main.py#L25)), which is never a browser-request origin.

### Testing & tooling
- No tests, no CI configuration, no linter/formatter config in the repo.

### Dead code & inconsistencies
- **`PredictionResponse.gradcam` and `subtype_gradcam` fields** ([backend/schemas.py:14-15](backend/schemas.py#L14-L15)) are declared but never populated by `/predict`.
- **`predict_with_gradcam` returns no class info**, forcing the frontend to call `/predict` first and pay for two decode/preprocess/forward cycles per image.
- **Frontend ships 8 sample MRIs** but the notebook only generates 4. The other 4 are sourced from somewhere undocumented.
- **Frontend progress bar is theatrical** ([frontend/app.js:266](frontend/app.js#L266)) — fake stages advance on a 2-second timer with no relation to actual server progress.
