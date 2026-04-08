# Wilson 95% confidence intervals — `leaky_4class`

Wilson score intervals (Wilson 1927) on per-class precision and recall, computed from `test_predictions.csv`. The Wilson interval is asymmetric, has correct coverage at small n, and does not collapse at p=0 or p=1. For classes with n<70 the interval is the more honest reporting unit than the point estimate.

**Test set size:** 2301 images.

## Overall accuracy

| metric | estimate | 95% CI | half-width |
|---|---|---|---|
| accuracy | 0.9896 | [0.9845, 0.9930] | ±0.42 pp |

## Per-class precision and recall

| class | n (true) | recall | recall 95% CI | recall ±pp | n (pred) | precision | precision 95% CI |
|---|---|---|---|---|---|---|---|
| glioma | 644 | 0.988 | [0.976, 0.994] | ±0.9 | 640 | 0.994 | [0.984, 0.998] |
| meningioma | 691 | 0.997 | [0.990, 0.999] | ±0.5 | 699 | 0.986 | [0.974, 0.992] |
| no_tumor | 604 | 0.995 | [0.985, 0.998] | ±0.6 | 609 | 0.987 | [0.974, 0.993] |
| pituitary | 362 | 0.970 | [0.946, 0.983] | ±1.8 | 353 | 0.994 | [0.980, 0.998] |

## Per-source overall accuracy

| source | n | accuracy | 95% CI | half-width |
|---|---|---|---|---|
| d1 | 1457 | 0.9849 | [0.9772, 0.9900] | ±0.64 pp |
| d2 | 844 | 0.9976 | [0.9914, 0.9993] | ±0.40 pp |

## Per-source per-class recall

Recall computed within each (source, class) cell. NaN cells indicate the class is not represented in that source.

| class | d1 | d2 |
|---|---|---|
| glioma | 0.979 [0.960, 0.989] (n=385) | 1.000 [0.985, 1.000] (n=259) |
| meningioma | 0.997 [0.984, 1.000] (n=360) | 0.997 [0.983, 0.999] (n=331) |
| no_tumor | 0.994 [0.979, 0.998] (n=350) | 0.996 [0.978, 0.999] (n=254) |
| pituitary | 0.970 [0.946, 0.983] (n=362) | — |

## Notes for the paper
