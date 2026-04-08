# Wilson 95% confidence intervals — `trustworthy_4class`

Wilson score intervals (Wilson 1927) on per-class precision and recall, computed from `test_predictions.csv`. The Wilson interval is asymmetric, has correct coverage at small n, and does not collapse at p=0 or p=1. For classes with n<70 the interval is the more honest reporting unit than the point estimate.

**Test set size:** 2300 images.

## Overall accuracy

| metric | estimate | 95% CI | half-width |
|---|---|---|---|
| accuracy | 0.9643 | [0.9560, 0.9712] | ±0.76 pp |

## Per-class precision and recall

| class | n (true) | recall | recall 95% CI | recall ±pp | n (pred) | precision | precision 95% CI |
|---|---|---|---|---|---|---|---|
| glioma | 654 | 0.913 | [0.889, 0.932] | ±2.2 | 602 | 0.992 | [0.981, 0.996] |
| meningioma | 706 | 0.984 | [0.972, 0.991] | ±0.9 | 737 | 0.943 | [0.924, 0.958] |
| no_tumor | 540 | 1.000 | [0.993, 1.000] | ±0.4 | 572 | 0.944 | [0.922, 0.960] |
| pituitary | 400 | 0.965 | [0.942, 0.979] | ±1.8 | 389 | 0.992 | [0.978, 0.997] |

## Per-source overall accuracy

| source | n | accuracy | 95% CI | half-width |
|---|---|---|---|---|
| d1 | 1600 | 0.9531 | [0.9416, 0.9624] | ±1.04 pp |
| d2 | 700 | 0.9900 | [0.9795, 0.9951] | ±0.78 pp |

## Per-source per-class recall

Recall computed within each (source, class) cell. NaN cells indicate the class is not represented in that source.

| class | d1 | d2 |
|---|---|---|
| glioma | 0.863 [0.825, 0.893] (n=400) | 0.992 [0.972, 0.998] (n=254) |
| meningioma | 0.985 [0.968, 0.993] (n=400) | 0.984 [0.962, 0.993] (n=306) |
| no_tumor | 1.000 [0.990, 1.000] (n=400) | 1.000 [0.973, 1.000] (n=140) |
| pituitary | 0.965 [0.942, 0.979] (n=400) | — |

## Notes for the paper
