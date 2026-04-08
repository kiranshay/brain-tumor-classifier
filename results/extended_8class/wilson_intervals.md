# Wilson 95% confidence intervals — `extended_8class`

Wilson score intervals (Wilson 1927) on per-class precision and recall, computed from `test_predictions.csv`. The Wilson interval is asymmetric, has correct coverage at small n, and does not collapse at p=0 or p=1. For classes with n<70 the interval is the more honest reporting unit than the point estimate.

**Test set size:** 2901 images.

## Overall accuracy

| metric | estimate | 95% CI | half-width |
|---|---|---|---|
| accuracy | 0.9476 | [0.9389, 0.9551] | ±0.81 pp |

## Per-class precision and recall

| class | n (true) | recall | recall 95% CI | recall ±pp | n (pred) | precision | precision 95% CI |
|---|---|---|---|---|---|---|---|
| carcinoma | 37 | 0.919 | [0.787, 0.972] | ±9.3 | 34 | 1.000 | [0.898, 1.000] |
| glioma | 836 | 0.925 | [0.905, 0.941] | ±1.8 | 810 | 0.954 | [0.938, 0.967] |
| meningioma | 837 | 0.951 | [0.934, 0.964] | ±1.5 | 848 | 0.939 | [0.920, 0.953] |
| neurocytoma | 68 | 0.941 | [0.858, 0.977] | ±5.9 | 67 | 0.955 | [0.876, 0.985] |
| no_tumor | 619 | 0.989 | [0.977, 0.995] | ±0.9 | 646 | 0.947 | [0.927, 0.962] |
| papilloma | 35 | 0.657 | [0.492, 0.792] | ±15.0 | 31 | 0.742 | [0.568, 0.863] |
| pituitary | 400 | 0.965 | [0.942, 0.979] | ±1.8 | 388 | 0.995 | [0.981, 0.999] |
| schwannoma | 69 | 0.884 | [0.788, 0.940] | ±7.6 | 77 | 0.792 | [0.689, 0.868] |

## Per-source overall accuracy

| source | n | accuracy | 95% CI | half-width |
|---|---|---|---|---|
| d1 | 1600 | 0.9506 | [0.9389, 0.9602] | ±1.07 pp |
| d2 | 700 | 0.9771 | [0.9632, 0.9859] | ±1.13 pp |
| d3 | 601 | 0.9052 | [0.8791, 0.9261] | ±2.35 pp |

## Per-source per-class recall

Recall computed within each (source, class) cell. NaN cells indicate the class is not represented in that source.

| class | d1 | d2 | d3 |
|---|---|---|---|
| carcinoma | — | — | 0.919 [0.787, 0.972] (n=37) |
| glioma | 0.873 [0.836, 0.902] (n=400) | 0.992 [0.972, 0.998] (n=254) | 0.945 [0.902, 0.970] (n=182) |
| meningioma | 0.968 [0.945, 0.981] (n=400) | 0.961 [0.933, 0.977] (n=306) | 0.878 [0.811, 0.923] (n=131) |
| neurocytoma | — | — | 0.941 [0.858, 0.977] (n=68) |
| no_tumor | 0.998 [0.986, 1.000] (n=400) | 0.986 [0.949, 0.996] (n=140) | 0.949 [0.877, 0.980] (n=79) |
| papilloma | — | — | 0.657 [0.492, 0.792] (n=35) |
| pituitary | 0.965 [0.942, 0.979] (n=400) | — | — |
| schwannoma | — | — | 0.884 [0.788, 0.940] (n=69) |

## Notes for the paper

**Rare-class CIs (n<100):**

- `papilloma` recall = 0.657 (n=35) → 95% CI [0.492, 0.792], half-width ±15.0 pp
- `carcinoma` recall = 0.919 (n=37) → 95% CI [0.787, 0.972], half-width ±9.3 pp
- `neurocytoma` recall = 0.941 (n=68) → 95% CI [0.858, 0.977], half-width ±5.9 pp
- `schwannoma` recall = 0.884 (n=69) → 95% CI [0.788, 0.940], half-width ±7.6 pp

These intervals are the quantitative version of Finding 4. The point estimate alone is not the right reporting unit at this sample size.
