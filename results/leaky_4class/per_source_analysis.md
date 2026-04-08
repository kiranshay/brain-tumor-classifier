# Leaky vs trustworthy: per-source breakdown

Generated 2026-04-08 from `runs/leaky_4class/test_predictions.csv` and
`results/trustworthy_4class/test_predictions.csv`.

The leaky build was created by pooling every image from
`trustworthy_4class/{train,val,test}/<class>/` and re-splitting at the
image level with a random shuffle (`build_leaky_dataset.py`,
`seed=42`). It was trained with **identical** hyperparameters,
architecture, and seed as the trustworthy run via `train_clean.py
--mode leaky`. The only variable that differs between the two runs is
the split strategy. Train/val/test sizes are constant within 3 images
(7769/1373/2301 vs 7772/1371/2300).

## Headline

| | trustworthy | leaky | delta |
|---|---|---|---|
| overall accuracy | 96.43% | 98.96% | **+2.52 pp** |

## Per-source overall

| source | trustworthy | leaky | delta |
|---|---|---|---|
| d1 (masoudnickparvar) | 95.31% | 98.49% | +3.18 pp |
| d2 (BRISC) | 99.00% | 99.76% | +0.76 pp |

## Per-source per-class recall

| (source, class) | trustworthy | leaky | delta |
|---|---|---|---|
| **d1 glioma** | **0.863** | **0.979** | **+0.116** |
| d2 glioma | 0.992 | 1.000 | +0.008 |
| d1 meningioma | 0.985 | 0.997 | +0.012 |
| d2 meningioma | 0.984 | 0.997 | +0.013 |
| d1 no_tumor | 1.000 | 0.994 | -0.006 |
| d2 no_tumor | 1.000 | 0.996 | -0.004 |
| d1 pituitary | 0.965 | 0.970 | +0.005 |

(d2 has no pituitary; pituitary is data1-only.)

## Interpretation

The +2.52 pp aggregate inflation from random image-level splits is
concentrated almost entirely in d1 gliomas: the +11.6 pp boost on
that single (class, source) cell accounts for the bulk of the
overall delta.

The non-glioma classes move by less than ±1.3 pp between the two
split strategies. Meningioma, no_tumor, and pituitary did not have
a distributional weakness in the trustworthy evaluation, so leakage
has nothing to "fix" in those classes — and the inflation
correspondingly does not appear.

This is the central finding: **random image-level split evaluation
does not inflate test accuracy uniformly. It selectively inflates
the classes that have real distributional problems**, masking
exactly the failure modes that an honest evaluation would surface.
A practitioner using the dominant Kaggle notebook split pattern on
this benchmark family would see d1 glioma recall at ~98% and have
no reason to investigate further. The published-split evaluation
shows it at 86.3% — a finding with clinical relevance about
distributional weaknesses across imaging sources.

## Why this is a controlled experiment

| variable | trustworthy | leaky |
|---|---|---|
| architecture | EfficientNet-B0 + Dropout(0.3) + Linear(1280, 4) | same |
| pretrained weights | ImageNet IMAGENET1K_V1 | same |
| optimizer | AdamW(lr=5e-5, wd=0.01) | same |
| LR schedule | CosineAnnealingLR(T_max=20) | same |
| epochs | 20 | same |
| batch size | 32 | same |
| augmentation | V2 pipeline (Resize 256 → RandomCrop 224 → flips → rot → affine → jitter → grayscale → normalize → erasing) | same |
| RNG seed | 42 | same |
| seeding scope | random/numpy/torch/cuda/cudnn/DataLoader generator + worker_init_fn | same |
| best checkpoint | val_loss | same |
| data sources | data1 + data2 | same |
| train images | 7,772 | 7,769 |
| val images | 1,371 | 1,373 |
| test images | 2,300 | 2,301 |
| **split strategy** | **published Training/test/ folders, 15% file-level val carve-out from train** | **all images pooled, random 67.9/12.0/20.1 image-level shuffle** |

The +2.52 pp delta is therefore attributable entirely to the split
strategy. This is the cleanest causal estimate of leakage inflation
that can be obtained on this benchmark family without institutional
patient-level data.
