"""Compute Wilson 95% score intervals for per-class precision and recall.

Takes the test_predictions.csv produced by train_clean.py and computes a
Wilson score interval (Wilson 1927) for each class's precision and recall.
The Wilson interval is the right tool here because it does not assume
normality, it does not collapse at p=0 or p=1, and it has correct coverage
at small n. Per-class metrics on the data44-only classes (papilloma n=35,
carcinoma n=37, neurocytoma n=68, schwannoma n=69) are exactly the regime
where the normal-approximation Wald interval is wrong.

Usage
-----
    python training/wilson_intervals.py \\
        --predictions results/trustworthy_4class/test_predictions.csv \\
        --output      results/trustworthy_4class/wilson_intervals.md \\
        --label       trustworthy_4class

Or run on all three runs at once:

    python training/wilson_intervals.py --all

What you get
------------
- A markdown table with precision, recall, F1, and 95% Wilson CIs for
  every class, plus per-source breakdowns where the source_prefix
  column has more than one value (d1, d2, d3).
- A CSV with the same content for downstream use in the paper.

The "central finding" version of this analysis is the rare-class CIs
in the extended 8-class build, where papilloma's recall is reported as
0.657 (n=35). The Wilson 95% interval on that estimate is roughly
[0.49, 0.79] -- a 30-point range, which is the entire reason Finding 4
of the paper exists. Quantifying it makes that finding citable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd


# Z value for a two-sided 95% interval. Using the standard normal
# percentile here even though we're computing the Wilson score
# interval -- the z just parameterizes which interval we want.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (point_estimate, lower_bound, upper_bound). Handles n==0 by
    returning (0.0, 0.0, 0.0) -- the only sensible thing for a class
    with zero support, since you can't compute a recall from no positives.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))) / denom
    return p_hat, max(0.0, center - half), min(1.0, center + half)


def per_class_metrics(df: pd.DataFrame) -> dict[str, dict[str, tuple[float, float, float, int]]]:
    """For each class, compute Wilson intervals on recall and precision.

    Recall denominator = number of true-class images (support).
    Precision denominator = number of times the model predicted that class.
    """
    classes = sorted(set(df["true_class"]) | set(df["predicted_class"]))
    out: dict[str, dict[str, tuple[float, float, float, int]]] = {}
    for cls in classes:
        # Recall: of the true positives for this class, how many did we get?
        true_mask = df["true_class"] == cls
        n_true = int(true_mask.sum())
        n_recalled = int((true_mask & (df["predicted_class"] == cls)).sum())
        rec_p, rec_lo, rec_hi = wilson_interval(n_recalled, n_true)

        # Precision: of the times we predicted this class, how many were right?
        pred_mask = df["predicted_class"] == cls
        n_pred = int(pred_mask.sum())
        n_correct = int((pred_mask & (df["true_class"] == cls)).sum())
        prec_p, prec_lo, prec_hi = wilson_interval(n_correct, n_pred)

        out[cls] = {
            "recall": (rec_p, rec_lo, rec_hi, n_true),
            "precision": (prec_p, prec_lo, prec_hi, n_pred),
        }
    return out


def overall_accuracy(df: pd.DataFrame) -> tuple[float, float, float, int]:
    n = len(df)
    correct = int((df["true_class"] == df["predicted_class"]).sum())
    return (*wilson_interval(correct, n), n)


def fmt_interval(p: float, lo: float, hi: float) -> str:
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_pct_interval(p: float, lo: float, hi: float) -> str:
    return f"{p*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]"


def half_width_pp(lo: float, hi: float) -> float:
    """Half-width of the interval in percentage points."""
    return (hi - lo) / 2.0 * 100.0


def render_markdown(label: str, df: pd.DataFrame, out_path: Path) -> None:
    """Write a markdown report with overall, per-class, and per-source tables."""
    overall_p, overall_lo, overall_hi, n_total = overall_accuracy(df)
    per_class = per_class_metrics(df)

    sources = sorted(df["source_prefix"].dropna().unique())
    has_sources = len(sources) > 1

    lines = []
    lines.append(f"# Wilson 95% confidence intervals — `{label}`")
    lines.append("")
    lines.append(
        "Wilson score intervals (Wilson 1927) on per-class precision and "
        "recall, computed from `test_predictions.csv`. The Wilson interval "
        "is asymmetric, has correct coverage at small n, and does not "
        "collapse at p=0 or p=1. For classes with n<70 the interval is "
        "the more honest reporting unit than the point estimate."
    )
    lines.append("")
    lines.append(f"**Test set size:** {n_total} images.")
    lines.append("")

    # ---- Overall ----------------------------------------------------------
    lines.append("## Overall accuracy")
    lines.append("")
    lines.append("| metric | estimate | 95% CI | half-width |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| accuracy | {overall_p:.4f} | [{overall_lo:.4f}, {overall_hi:.4f}] | "
        f"±{half_width_pp(overall_lo, overall_hi):.2f} pp |"
    )
    lines.append("")

    # ---- Per-class --------------------------------------------------------
    lines.append("## Per-class precision and recall")
    lines.append("")
    lines.append(
        "| class | n (true) | recall | recall 95% CI | recall ±pp | "
        "n (pred) | precision | precision 95% CI |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for cls in sorted(per_class.keys()):
        rec_p, rec_lo, rec_hi, n_true = per_class[cls]["recall"]
        prec_p, prec_lo, prec_hi, n_pred = per_class[cls]["precision"]
        lines.append(
            f"| {cls} | {n_true} | {rec_p:.3f} | "
            f"[{rec_lo:.3f}, {rec_hi:.3f}] | "
            f"±{half_width_pp(rec_lo, rec_hi):.1f} | "
            f"{n_pred} | {prec_p:.3f} | "
            f"[{prec_lo:.3f}, {prec_hi:.3f}] |"
        )
    lines.append("")

    # ---- Per-source -------------------------------------------------------
    if has_sources:
        lines.append("## Per-source overall accuracy")
        lines.append("")
        lines.append("| source | n | accuracy | 95% CI | half-width |")
        lines.append("|---|---|---|---|---|")
        for src in sources:
            sub = df[df["source_prefix"] == src]
            p, lo, hi, n = overall_accuracy(sub)
            lines.append(
                f"| {src} | {n} | {p:.4f} | [{lo:.4f}, {hi:.4f}] | "
                f"±{half_width_pp(lo, hi):.2f} pp |"
            )
        lines.append("")

        lines.append("## Per-source per-class recall")
        lines.append("")
        lines.append(
            "Recall computed within each (source, class) cell. NaN cells "
            "indicate the class is not represented in that source."
        )
        lines.append("")
        # Build a wide table: rows = class, columns = source.
        all_classes = sorted(set(df["true_class"]))
        header = "| class | " + " | ".join(sources) + " |"
        sep = "|---|" + "---|" * len(sources)
        lines.append(header)
        lines.append(sep)
        for cls in all_classes:
            row = [cls]
            for src in sources:
                sub = df[(df["source_prefix"] == src) & (df["true_class"] == cls)]
                if len(sub) == 0:
                    row.append("—")
                    continue
                n_true = len(sub)
                n_recalled = int((sub["predicted_class"] == cls).sum())
                p, lo, hi = wilson_interval(n_recalled, n_true)
                row.append(
                    f"{p:.3f} [{lo:.3f}, {hi:.3f}] (n={n_true})"
                )
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ---- Notes for the paper ---------------------------------------------
    lines.append("## Notes for the paper")
    lines.append("")
    rare = [(c, m["recall"]) for c, m in per_class.items() if m["recall"][3] < 100]
    if rare:
        lines.append("**Rare-class CIs (n<100):**")
        lines.append("")
        for cls, (p, lo, hi, n) in sorted(rare, key=lambda kv: kv[1][3]):
            lines.append(
                f"- `{cls}` recall = {p:.3f} (n={n}) → 95% CI [{lo:.3f}, {hi:.3f}], "
                f"half-width ±{half_width_pp(lo, hi):.1f} pp"
            )
        lines.append("")
        lines.append(
            "These intervals are the quantitative version of Finding 4. "
            "The point estimate alone is not the right reporting unit at "
            "this sample size."
        )
        lines.append("")

    out_path.write_text("\n".join(lines))


def render_csv(label: str, df: pd.DataFrame, out_path: Path) -> None:
    """Companion CSV with one row per class for downstream use."""
    per_class = per_class_metrics(df)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "class", "n_true", "recall", "recall_lo", "recall_hi",
            "n_pred", "precision", "precision_lo", "precision_hi",
        ])
        for cls in sorted(per_class.keys()):
            rec_p, rec_lo, rec_hi, n_true = per_class[cls]["recall"]
            prec_p, prec_lo, prec_hi, n_pred = per_class[cls]["precision"]
            w.writerow([
                label, cls, n_true,
                f"{rec_p:.6f}", f"{rec_lo:.6f}", f"{rec_hi:.6f}",
                n_pred,
                f"{prec_p:.6f}", f"{prec_lo:.6f}", f"{prec_hi:.6f}",
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path,
                   help="Path to a test_predictions.csv from train_clean.py")
    p.add_argument("--output", type=Path,
                   help="Where to write the wilson_intervals.md report. "
                        "A sibling .csv is also written.")
    p.add_argument("--label", type=str,
                   help="Human label for the report header (e.g. trustworthy_4class).")
    p.add_argument("--all", action="store_true",
                   help="Run on all three results/{trustworthy,extended,leaky}_*/ "
                        "directories. Overrides --predictions/--output/--label.")
    return p.parse_args()


def run_one(predictions: Path, output: Path, label: str) -> None:
    df = pd.read_csv(predictions)
    if "source_prefix" not in df.columns:
        df["source_prefix"] = ""
    output.parent.mkdir(parents=True, exist_ok=True)
    render_markdown(label, df, output)
    csv_path = output.with_suffix(".csv")
    render_csv(label, df, csv_path)
    print(f"[ok] {label}: wrote {output} and {csv_path}")


def main() -> int:
    args = parse_args()

    if args.all:
        repo_root = Path(__file__).resolve().parent.parent
        results = repo_root / "results"
        runs = [
            ("trustworthy_4class", results / "trustworthy_4class"),
            ("extended_8class",    results / "extended_8class"),
            ("leaky_4class",       results / "leaky_4class"),
        ]
        for label, run_dir in runs:
            preds = run_dir / "test_predictions.csv"
            if not preds.exists():
                print(f"[skip] {label}: {preds} not found")
                continue
            out = run_dir / "wilson_intervals.md"
            run_one(preds, out, label)
        return 0

    if not args.predictions or not args.output or not args.label:
        raise SystemExit(
            "ERROR: pass --predictions, --output, and --label, or use --all."
        )
    run_one(args.predictions, args.output, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
