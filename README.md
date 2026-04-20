# NeuroScan — Brain Tumor MRI Classifier

Full-stack web app for classifying brain MRI scans using a two-stage deep learning pipeline. The primary model classifies scans into eight tumor categories; a secondary model further subtypes gliomas into four subtypes.

**[Live Demo](https://kiranshay.github.io/brain-tumor-classifier)** &nbsp;|&nbsp; [Demo Video](NeuroScan%20Demo.mov)

## Architecture

```
┌──────────────┐    HTTPS    ┌────────────────────┐    HTTPS    ┌───────────────────┐
│  Frontend    │  ────────▶  │  FastAPI Backend   │  ────────▶  │ Supabase (Postgres)│
│  (GitHub     │             │  (Render, CPU)     │             │  predictions tbl  │
│   Pages)     │  ◀────────  │                    │  ◀────────  │                   │
└──────────────┘             └─────────┬──────────┘             └───────────────────┘
                                       │
                                       ▼
                              ┌───────────────────┐
                              │  EfficientNet-B0  │
                              │  Stage 1: 8-class │
                              │  Stage 2: glioma  │
                              │         subtype   │
                              └───────────────────┘
```

## Model

### Stage 1 — Tumor Classification (8-class)

EfficientNet-B0 (transfer learning from ImageNet), fully fine-tuned with AdamW and cosine annealing LR.

**Classes:** carcinoma, glioma, meningioma, neurocytoma, no tumor, papilloma, pituitary, schwannoma

| Metric | Value |
|--------|-------|
| Test accuracy | 94.8% |
| Weighted F1 | 0.948 |
| Macro F1 | 0.909 |

### Stage 2 — Glioma Subtyping (4-class)

A second EfficientNet-B0 fires conditionally when Stage 1 predicts glioma, classifying the subtype.

**Subtypes:** astrocytoma, ependymoma, glioblastoma, oligodendroglioma

### Trustworthy 4-class baseline

A separate patient-level split model trained on the original 4-class task (glioma, meningioma, no tumor, pituitary) achieves **96.4% test accuracy** with proper train/val/test separation.

### Interpretability

Grad-CAM heatmaps are generated on demand, highlighting the regions the model attends to for each prediction.

## Features

- Upload an MRI scan or use built-in samples
- Real-time classification with confidence scores for all classes
- Conditional glioma subtype classification
- Grad-CAM saliency maps for model interpretability
- Prediction history and aggregate statistics persisted in Supabase

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | PyTorch, EfficientNet-B0, transfer learning |
| Backend | FastAPI, Uvicorn |
| Database | Supabase (PostgreSQL) |
| Frontend | HTML, CSS, vanilla JavaScript |
| Training | Google Colab (T4 GPU) |
| Deployment | Render (backend), GitHub Pages (frontend) |

## Datasets

| Source | Kaggle Slug | Classes |
|--------|------------|---------|
| Brain Tumor MRI | `masoudnickparvar/brain-tumor-mri-dataset` | 4-class (glioma, meningioma, no tumor, pituitary) |
| BRISC 2025 | `briscdataset/brisc2025` | 4-class |
| Brain Tumor 44-class | `fernando2rad/brain-tumor-mri-images-44c` | 44 fine-grained tumor types (T1, T2, T1C+ sequences) |

The 8-class stage-1 model is trained on all three datasets combined. The glioma subtype model uses the 44-class dataset only.

## Training

All models use:
- **Architecture:** EfficientNet-B0 with custom classifier head (`Dropout → Linear`)
- **Optimizer:** AdamW (lr=5e-5, weight_decay=0.01)
- **Schedule:** CosineAnnealingLR (T_max=20)
- **Loss:** CrossEntropyLoss
- **Augmentation:** Resize 256 → RandomCrop 224 → H/V flip → rotation ±20° → affine → color jitter → random grayscale → random erasing

Training notebooks and scripts are in [`training/`](training/):

| File | Purpose |
|------|---------|
| `train_8class_v5.ipynb` | Stage 1 (deployed 8-class model) |
| `train_glioma_subtype.ipynb` | Stage 2 (glioma subtype model) |
| `train.ipynb` | Legacy 4-class model (progressive unfreezing) |
| `train_clean.py` | Reproducible CLI trainer with patient-level splits |
| `build_clean_dataset.py` | Dataset builder with patient-level deduplication |
| `build_leaky_dataset.py` | Controlled leakage experiment |
| `wilson_intervals.py` | Wilson 95% CI computation for per-class metrics |

## Evaluation Results

Tracked in [`results/`](results/):

| Run | Description | Test Accuracy |
|-----|-------------|---------------|
| `trustworthy_4class` | Patient-level split, 4-class, data1 only | 96.4% |
| `extended_8class` | 8-class, all datasets combined | 94.8% |
| `leaky_4class` | Controlled leakage experiment (image-level split) | — |

Each run includes: confusion matrix, training curves, test predictions CSV, classification report, Wilson confidence intervals, and a run summary JSON.

## Project Structure

```
brain-tumor-classifier/
├── backend/
│   ├── main.py                         # FastAPI endpoints
│   ├── model.py                        # Model loading, inference, Grad-CAM
│   ├── database.py                     # Supabase client
│   ├── schemas.py                      # Pydantic response models
│   ├── requirements.txt
│   ├── tumor_classifier.pth            # Stage 1 weights
│   └── glioma_subtype_classifier.pth   # Stage 2 weights
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── assets/sample_mris/
├── training/
│   ├── train_8class_v5.ipynb
│   ├── train_glioma_subtype.ipynb
│   ├── train.ipynb
│   ├── train_clean.py
│   ├── build_clean_dataset.py
│   ├── build_leaky_dataset.py
│   └── wilson_intervals.py
├── results/
│   ├── trustworthy_4class/
│   ├── extended_8class/
│   └── leaky_4class/
├── docs/                               # GitHub Pages deployment copy
└── ARCHITECTURE.md
```

## Setup

### 1. Train the model

Open the training notebooks in Google Colab with a GPU runtime. See [`training/`](training/) for details on each training configuration.

### 2. Set up Supabase

Create a project at [supabase.com](https://supabase.com) and run:

```sql
CREATE TABLE predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz DEFAULT now(),
  predicted_class text NOT NULL,
  confidence float8 NOT NULL,
  all_confidences jsonb NOT NULL,
  thumbnail_base64 text,
  original_filename text,
  inference_time_ms float8,
  subtype text,
  subtype_confidence float8,
  subtype_confidences jsonb
);

ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow anonymous access" ON predictions
  FOR ALL USING (true) WITH CHECK (true);
```

### 3. Run the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SUPABASE_URL="your-project-url"
export SUPABASE_KEY="your-anon-key"

uvicorn main:app --reload --port 8000
```

### 4. Run the frontend

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500`.

## Deployment

| Component | Host | Notes |
|-----------|------|-------|
| Backend | Render (free tier) | CPU-only PyTorch, ~30s cold start |
| Frontend | GitHub Pages | Served from `docs/` on `main` |
| Database | Supabase (free tier) | Kept alive via scheduled GitHub Action ping |

## Author

[Kiran Shay](https://kiranshay.github.io) — Johns Hopkins University, Computer Science & Neuroscience
