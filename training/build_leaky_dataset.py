"""Build a deliberately leaky baseline dataset for the controlled leakage experiment.

This script is the methodological control for the trustworthy_4class build.
The trustworthy build preserves the published train/test splits from data1
(masoudnickparvar) and data2 (BRISC) — so the test set is curated by the
dataset authors and contains no images from the training pool. This script
does the opposite: it pools every image from
trustworthy_4class/{train,val,test}/<class>/ into a single bucket per class,
then re-splits at the IMAGE level using a random shuffle. That is what the
dominant published Kaggle notebooks on these datasets actually do.

The leaky build is then trained with the SAME hyperparameters, the SAME
seed, the SAME architecture, and the SAME training protocol as the
trustworthy build via training/train_clean.py. The only variable that
differs between the two runs is the split strategy. The delta in test
accuracy is therefore a controlled estimate of how much file-level random
splitting inflates reported accuracy on these benchmarks.

Why pool from trustworthy_4class instead of from the raw Kaggle sources:
trustworthy_4class already contains exactly the union of data1 + data2
images for the four overlapping classes, with d1_/d2_ source prefixes
preserved on the filenames. Pooling from this directory guarantees the
leaky experiment uses literally the same image set as the trustworthy
experiment — no risk of accidentally including or excluding different
files. The d1_/d2_ prefixes also survive into the leaky run's
test_predictions.csv, so per-source post-hoc analysis still works.

What the leaky test set looks like vs. the trustworthy test set: the leaky
test set will contain a random ~20% of the same physical images that the
trustworthy build assigned across train/val/test. The model trained on the
leaky train split will have seen ~80% of those test images during training.
That is the leak. The resulting test accuracy is what a naive Kaggle
pipeline reports. The trustworthy accuracy (96.43%) is what an honest
pipeline reports. The delta is the inflation introduced by pretending a
random shuffle is a held-out test set.

Split ratios are chosen to match the trustworthy build's actual ratios
(7772 train / 1371 val / 2300 test = 67.9% / 12.0% / 20.1%) so that train
size, val size, and test size are all held constant between the two runs.
The only thing that changes is which images land in which bucket.

Usage
-----
    python training/build_leaky_dataset.py \\
        --source-dir /content/clean_dataset/trustworthy_4class \\
        --output-dir /content/clean_dataset/leaky_4class \\
        --seed 42

then train it with the SAME command you used for the trustworthy run,
swapping --mode and --data-dir:

    python training/train_clean.py \\
        --mode leaky \\
        --data-dir /content/clean_dataset/leaky_4class \\
        --output-dir runs/leaky_4class \\
        --epochs 20 --batch-size 32 --lr 5e-5 \\
        --weight-decay 0.01 --dropout 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# These match the trustworthy build's actual split ratios so train size, val
# size, and test size are constant between trustworthy and leaky. The only
# thing that changes is which images land in which split. trustworthy was
# 7772/1371/2300 = 67.9 / 12.0 / 20.1.
TRAIN_FRAC = 0.679
VAL_FRAC = 0.120
# TEST_FRAC = 0.201 implied


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True,
                   help="Path to an existing trustworthy_4class build "
                        "(must contain train/, val/, test/ subdirs).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the leaky train/val/test/<class>/ tree.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    src_train = args.source_dir / "train"
    if not src_train.is_dir():
        raise SystemExit(
            f"ERROR: {src_train} does not exist. Pass --source-dir pointing "
            f"at an existing trustworthy_4class build."
        )
    classes = sorted(d.name for d in src_train.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"ERROR: no class subdirectories found under {src_train}")
    print(f"[classes] {classes}")
    print(f"[seed] {args.seed}")

    rng = random.Random(args.seed)

    for split in ("train", "val", "test"):
        for cls in classes:
            (args.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {
        cls: {"train": 0, "val": 0, "test": 0} for cls in classes
    }

    for cls in classes:
        # Pool every image for this class across all three source splits.
        all_files: list[Path] = []
        for split in ("train", "val", "test"):
            all_files.extend(list_images(args.source_dir / split / cls))
        if not all_files:
            print(f"WARNING: no images found for class '{cls}'")
            continue

        # Random shuffle is the leak. We deliberately do NOT respect the
        # published train/test split, scan series, or any other structure —
        # the whole point is to mimic what train_test_split() on the pooled
        # image list does in the dominant published Kaggle notebooks.
        rng.shuffle(all_files)

        n = len(all_files)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]

        # Copy with the original filename so the d1_/d2_ source prefix
        # survives into test_predictions.csv — per-source post-hoc analysis
        # still works on the leaky run.
        for split, files in (("train", train_files),
                             ("val", val_files),
                             ("test", test_files)):
            dst = args.output_dir / split / cls
            for src in files:
                shutil.copy2(src, dst / src.name)
            counts[cls][split] = len(files)

    # ---- Summary printout --------------------------------------------------
    print()
    print("=" * 60)
    print(f"Leaky 4-class build summary  [{args.output_dir}]")
    print("=" * 60)
    splits = ("train", "val", "test")
    print(f"{'class':>14s} | " + " | ".join(f"{s:>7s}" for s in splits))
    print("-" * 50)
    totals = {s: 0 for s in splits}
    for cls in classes:
        row = f"{cls:>14s} | " + " | ".join(
            f"{counts[cls][s]:>7d}" for s in splits
        )
        print(row)
        for s in splits:
            totals[s] += counts[cls][s]
    print("-" * 50)
    print(f"{'TOTAL':>14s} | " + " | ".join(f"{totals[s]:>7d}" for s in splits))

    # ---- Dataset card -----------------------------------------------------
    card = {
        "label": "leaky_4class",
        "purpose": (
            "Methodological control for the trustworthy_4class build. "
            "Created by pooling every image from trustworthy_4class/"
            "{train,val,test}/<class>/ and re-splitting at the IMAGE level "
            "with a random shuffle, deliberately ignoring the published "
            "train/test splits authored by the dataset creators. The "
            "resulting test set shares file content with the trustworthy "
            "test set, but the trained model will have seen ~80% of those "
            "images during training. This is what a naive Kaggle pipeline "
            "produces. Compare its test accuracy to trustworthy_4class's "
            "test accuracy (96.43%); the delta is a controlled estimate of "
            "the inflation introduced by file-level random splits on these "
            "datasets."
        ),
        "classes": classes,
        "split_fractions": {
            "train": TRAIN_FRAC,
            "val": VAL_FRAC,
            "test": round(1.0 - TRAIN_FRAC - VAL_FRAC, 3),
        },
        "seed": args.seed,
        "counts_per_class_per_split": counts,
        "warnings": [
            "DO NOT report metrics from this build as a real result. It "
            "exists only as a methodological control to estimate leakage "
            "inflation against the trustworthy_4class build."
        ],
        "compare_against": "results/trustworthy_4class/run_summary.json",
    }
    (args.output_dir / "dataset_card.json").write_text(
        json.dumps(card, indent=2)
    )
    print(f"\nWrote dataset card: {args.output_dir / 'dataset_card.json'}")
    print()
    print("Next: train with IDENTICAL hyperparameters as the trustworthy run:")
    print()
    print(f"  python training/train_clean.py \\")
    print(f"      --mode leaky \\")
    print(f"      --data-dir {args.output_dir} \\")
    print(f"      --output-dir runs/leaky_4class \\")
    print(f"      --epochs 20 --batch-size 32 --lr 5e-5 \\")
    print(f"      --weight-decay 0.01 --dropout 0.3 --seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
