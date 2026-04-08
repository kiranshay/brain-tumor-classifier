"""Build a clean train/val/test dataset for the NeuroScan 8-class classifier.

Why patient-level splitting matters
-----------------------------------
Brain MRI datasets typically contain many 2-D slices (and often multiple
sequences -- T1, T2, T1C+) per individual patient. If you split such a
dataset randomly at the *image* level, slices from the same patient land
in both the training and the test set. The model can then memorise
patient-specific anatomy (skull shape, ventricle geometry, scanner
artefacts) instead of learning the *tumour* features that should
generalise to new patients. Reported test accuracy is inflated by an
amount that cannot be recovered after the fact.

The deployed NeuroScan models (V5 stage 1 and the glioma-subtype stage 2)
were both built with image-level ``random.shuffle`` 80/20 splits on the
data44 source -- see ``ARCHITECTURE.md`` "Known Methodological
Limitations". This script rebuilds the dataset directory with three
fixes:

1. data44 is split at the **patient** level when patient IDs can be
   inferred from filenames, so that all slices and all sequences from
   one patient stay together in a single split.
2. A held-out **validation** set is carved out of the training data
   (15% of the patient pool for data44; 15% of the train set for data1
   and data2). The validation set is what the LR scheduler / early
   stopping should look at, and the test set is never touched until
   final evaluation.
3. The four data44-only classes (schwannoma, neurocytoma, carcinoma,
   papilloma) carry an unfixable **source confound** -- they only exist
   in one source, so any model trained on this data may be learning
   scanner signatures rather than tumour morphology. This script cannot
   fix that, but it emits a warning in the dataset card so the
   limitation is documented next to the per-class counts.

This script does NOT train any model. It only assembles
``output_dir/{train,val,test}/<class>/`` and writes a
``dataset_card.json`` describing exactly what was built.

Modes
-----
``--mode trustworthy``
    4-class build (glioma, meningioma, no_tumor, pituitary) using
    only data1 and data2. Both sources ship with their own
    train/test splits authored by the dataset creators; we preserve
    those and carve a 15% file-level val set out of each train pool.
    No data44 is used. This is the **headline / defensible** result
    for any reporting.

``--mode extended``
    8-class build that adds the 4 data44-only rare classes
    (schwannoma, neurocytoma, carcinoma, papilloma) and pulls
    additional images for the 4 overlapping classes from data44
    sister folders. Patient-level grouping is attempted from
    filenames; if it fails (e.g. data44's content-hash filenames),
    falls back to image-level splits with a loud warning. Treat
    metrics on the data44-only classes as **exploratory** because of
    (a) source confound (only one source) and (b) possible
    patient-level leakage when the fallback triggers.

``--mode both`` (default)
    Builds both into ``<output-dir>/trustworthy_4class/`` and
    ``<output-dir>/extended_8class/``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Class -> source-folder mapping. Mirrors EXPANDED_CLASSES in
# training/train_8class_v5.ipynb (cell-6) so the resulting class set is
# identical to what the deployed model expects.
# ---------------------------------------------------------------------------

EXPANDED_CLASSES: dict[str, dict] = {
    "glioma": {
        "data1": {"Training": ["glioma"], "Testing": ["glioma"]},
        "data2": {"train": ["glioma"], "test": ["glioma"]},
        "data44": [
            "Astrocitoma T1", "Astrocitoma T2", "Astrocitoma T1C+",
            "Glioblastoma T1", "Glioblastoma T2", "Glioblastoma T1C+",
            "Oligodendroglioma T1", "Oligodendroglioma T2", "Oligodendroglioma T1C+",
            "Ependimoma T1", "Ependimoma T2", "Ependimoma T1C+",
            "Ganglioglioma T1", "Ganglioglioma T2", "Ganglioglioma T1C+",
        ],
    },
    "meningioma": {
        "data1": {"Training": ["meningioma"], "Testing": ["meningioma"]},
        "data2": {"train": ["meningioma"], "test": ["meningioma"]},
        "data44": ["Meningioma T1", "Meningioma T2", "Meningioma T1C+"],
    },
    "no_tumor": {
        "data1": {"Training": ["notumor"], "Testing": ["notumor"]},
        "data2": {"train": ["no_tumor"], "test": ["no_tumor"]},
        "data44": ["_NORMAL T1", "_NORMAL T2"],
    },
    "pituitary": {
        "data1": {"Training": ["pituitary"], "Testing": ["pituitary"]},
    },
    "schwannoma": {
        "data44": ["Schwannoma T1", "Schwannoma T2", "Schwannoma T1C+"],
    },
    "neurocytoma": {
        "data44": ["Neurocitoma T1", "Neurocitoma T2", "Neurocitoma T1C+"],
    },
    "carcinoma": {
        "data44": ["Carcinoma T1", "Carcinoma T2", "Carcinoma T1C+"],
    },
    "papilloma": {
        "data44": ["Papiloma T1", "Papiloma T2", "Papiloma T1C+"],
    },
}

DATA44_ONLY_CLASSES = {"schwannoma", "neurocytoma", "carcinoma", "papilloma"}

# The "trustworthy" 4-class subset: classes for which both data1 and data2
# provide images, so the model can be trained and evaluated entirely on
# sources with published train/test splits and (in BRISC's case) some
# patient-aware curation. Used by --mode trustworthy.
TRUSTWORTHY_CLASSES: tuple[str, ...] = (
    "glioma", "meningioma", "no_tumor", "pituitary",
)
EXTENDED_CLASSES: tuple[str, ...] = tuple(EXPANDED_CLASSES.keys())

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

VAL_FRACTION_OF_TRAIN = 0.15  # for data1 / data2 carve-out
DATA44_TRAIN_FRAC = 0.70
DATA44_VAL_FRAC = 0.15
DATA44_TEST_FRAC = 0.15  # implied; the three sum to 1.0


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ensure_split_dirs(output_dir: Path, classes: Iterable[str]) -> None:
    for split in ("train", "val", "test"):
        for cls in classes:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def copy_with_prefix(src: Path, dst_dir: Path, prefix: str, idx: int) -> None:
    """Copy ``src`` into ``dst_dir`` renaming with ``<prefix>_<idx>_<basename>``.

    The numeric ``idx`` guarantees uniqueness across sources even when two
    sources happen to use the same basename.
    """
    dst_name = f"{prefix}_{idx}_{src.name}"
    shutil.copy2(src, dst_dir / dst_name)


# ---------------------------------------------------------------------------
# Patient-ID inference for data44
# ---------------------------------------------------------------------------
#
# The fernando2rad/brain-tumor-mri-images-44c dataset is organised as
#     data44/<TumorName Sequence>/<file>.<ext>
# where <Sequence> is one of T1, T2, T1C+. Multiple slices per patient
# are common, and the same patient often has scans in more than one
# sequence folder.
#
# The filename pattern across data44 is not 100% uniform across all 44
# folders, so we try a couple of heuristics in order and pick the one
# that produces the most plausible grouping. If none of them group
# anything (i.e. every file ends up as its own "patient"), we warn and
# fall back to a file-level split for that class.
#
# Heuristics tried, in order:
#   1. A run of >=2 digits anywhere in the basename. The first such run
#      is taken as the patient ID. This catches patterns like
#      "p0012_axial_03.jpg", "patient12-slice4.png", "0012 (1).jpg".
#      It also (incorrectly) groups files where the digits are a
#      *slice* index rather than a patient index -- but in that case
#      the slice numbers usually still correlate with patients within
#      a sister-sequence folder, and the worst case is that we
#      over-group, which is conservative (it never causes leakage,
#      only reduces effective sample size).
#   2. The leading non-digit token of the basename (everything up to
#      the first digit or separator). Catches "PatientA_*.jpg".
#
# If the chosen heuristic produces a number of unique IDs equal to the
# number of files (i.e. no grouping at all), we mark the inference as
# UNRELIABLE for that class and fall back to a deterministic
# image-level split, with a warning recorded in the dataset card.
#
# NOTE: This is a best-effort inference. Without ground-truth patient
# manifests from the dataset authors, perfect grouping cannot be
# guaranteed. Any over-grouping is safe (no leakage); any
# under-grouping leaves residual leakage which the warning surfaces.

_DIGIT_RUN_RE = re.compile(r"\d{2,}")
_LEADING_TOKEN_RE = re.compile(r"^[A-Za-z]+")


def _infer_id_digits(stem: str) -> str | None:
    m = _DIGIT_RUN_RE.search(stem)
    return m.group(0) if m else None


def _infer_id_leading(stem: str) -> str | None:
    m = _LEADING_TOKEN_RE.match(stem)
    return m.group(0) if m else None


def infer_patient_ids(files: list[Path]) -> tuple[dict[Path, str], str, bool]:
    """Return (file -> patient_id, strategy_name, reliable).

    ``reliable`` is False when the chosen strategy produced one ID per
    file (i.e. no grouping at all). Callers should fall back to a
    file-level split in that case.
    """
    if not files:
        return {}, "none", True

    strategies = (
        ("digit_run", _infer_id_digits),
        ("leading_token", _infer_id_leading),
    )

    best_mapping: dict[Path, str] = {}
    best_unique = len(files) + 1  # lower is better (more grouping)
    best_name = "image_level"
    for name, fn in strategies:
        mapping: dict[Path, str] = {}
        for f in files:
            pid = fn(f.stem)
            if pid is None:
                # If the strategy can't extract anything, fall back to
                # the unique filename so this file becomes its own group.
                pid = f.stem
            mapping[f] = pid
        unique = len(set(mapping.values()))
        if unique < best_unique:
            best_unique = unique
            best_mapping = mapping
            best_name = name

    # Reliability check. The earlier version of this function only
    # required best_unique < len(files) -- i.e. "did ANY two files end
    # up in the same group". That is way too lenient on content-hash
    # filenames (data44 is full of SHA-256-prefixed names like
    # ``01809e58...jpeg``), where leading-digit collisions happen by
    # accident even though the filenames carry no patient information.
    # The result was bogus "patient counts" in the dataset card and one
    # data44 class (papilloma) where the random partition put more
    # images in val than in train because one bogus group absorbed half
    # the class.
    #
    # Real brain MRI patients have many slices each (typically 10+
    # axial slices per sequence, often across multiple sequences). If
    # the median group size is 1 -- i.e. more than half of all files
    # are alone in their group -- the inference is essentially noise
    # and we should fall back to image-level. We also require a mean
    # of at least ~3 files per group, since a real per-patient pile of
    # MRI slices comfortably clears that bar but spurious hash-prefix
    # collisions do not.
    from statistics import median
    group_sizes = list(_group_size_counts(best_mapping).values())
    med = median(group_sizes) if group_sizes else 0
    mean = (sum(group_sizes) / len(group_sizes)) if group_sizes else 0
    reliable = (
        best_unique < len(files)
        and med >= 2
        and mean >= 3.0
    )
    if not reliable:
        # Either no strategy grouped anything, or the grouping was so
        # sparse it can't reflect real patient identity. Treat each
        # file as its own "patient" so the caller's fallback path
        # produces a deterministic image-level split with a warning.
        best_mapping = {f: f.stem for f in files}
        best_name = "image_level_fallback"
    return best_mapping, best_name, reliable


def _group_size_counts(mapping: dict[Path, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pid in mapping.values():
        counts[pid] = counts.get(pid, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Patient-level splitting
# ---------------------------------------------------------------------------

def split_patients(
    patient_to_files: dict[str, list[Path]],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path], dict[str, int]]:
    """Split a {patient_id: [files]} mapping into (train, val, test) file lists.

    Patients (not files) are shuffled and partitioned, then the files
    inside each patient are emitted whole into the chosen split.
    """
    patients = sorted(patient_to_files.keys())
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Guard against degenerate tiny classes: ensure at least 1 patient
    # in val and test if there are >=3 patients overall.
    if n >= 3:
        if n_val == 0:
            n_val = 1
        if n_train + n_val >= n:
            n_train = max(1, n - n_val - 1)
    train_p = patients[:n_train]
    val_p = patients[n_train:n_train + n_val]
    test_p = patients[n_train + n_val:]

    train_files = [f for p in train_p for f in patient_to_files[p]]
    val_files = [f for p in val_p for f in patient_to_files[p]]
    test_files = [f for p in test_p for f in patient_to_files[p]]
    counts = {
        "train_patients": len(train_p),
        "val_patients": len(val_p),
        "test_patients": len(test_p),
    }
    return train_files, val_files, test_files, counts


def split_files(
    files: list[Path],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return (
        files[:n_train],
        files[n_train:n_train + n_val],
        files[n_train + n_val:],
    )


# ---------------------------------------------------------------------------
# Per-source builders
# ---------------------------------------------------------------------------

class BuildState:
    """Mutable state shared across per-source builders."""

    def __init__(
        self,
        output_dir: Path,
        classes: tuple[str, ...],
        label: str,
    ) -> None:
        self.output_dir = output_dir
        self.classes = classes
        self.label = label  # human name e.g. "trustworthy_4class"
        self.idx = 0  # global running counter for filename uniqueness
        self.warnings: list[str] = []
        # per-class per-split file counts (for the dataset card / printout)
        self.counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"train": 0, "val": 0, "test": 0}
        )
        # per-class data44 patient counts (only filled when reliable)
        self.data44_patient_counts: dict[str, dict[str, int]] = {}
        # per-class chosen split methodology
        self.methodology: dict[str, dict] = {}
        # sources actually used
        self.sources_used: set[str] = set()

    def emit(self, files: Iterable[Path], cls: str, split: str, prefix: str) -> None:
        dst_dir = self.output_dir / split / cls
        for src in files:
            copy_with_prefix(src, dst_dir, prefix, self.idx)
            self.idx += 1
            self.counts[cls][split] += 1


def build_data1(
    cls: str,
    sources: dict,
    data1_dir: Path | None,
    state: BuildState,
    rng: random.Random,
) -> None:
    if "data1" not in sources or data1_dir is None:
        return
    state.sources_used.add("data1")

    train_pool: list[Path] = []
    test_files: list[Path] = []
    for split_src, folders in sources["data1"].items():
        for folder in folders:
            files = list_images(data1_dir / split_src / folder)
            if split_src == "Training":
                train_pool.extend(files)
            else:
                test_files.extend(files)

    # Carve 15% val out of the train pool. data1 filenames (e.g.
    # "Tr-gl_0010.jpg") are slice-indexed without patient IDs, so this
    # is a file-level split. Documented as a known limitation in the
    # dataset card.
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * VAL_FRACTION_OF_TRAIN))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    state.emit(train_files, cls, "train", prefix="d1")
    state.emit(val_files, cls, "val", prefix="d1")
    state.emit(test_files, cls, "test", prefix="d1")

    state.methodology.setdefault(cls, {})["data1"] = {
        "split": "preserved-test, file-level val carve-out",
        "reason": (
            "data1 ships with its own Training/Testing split. Patient IDs "
            "are not encoded in the filenames, so the val carve-out is "
            "file-level."
        ),
    }


def build_data2(
    cls: str,
    sources: dict,
    data2_dir: Path | None,
    state: BuildState,
    rng: random.Random,
) -> None:
    if "data2" not in sources or data2_dir is None:
        return
    state.sources_used.add("data2")

    # BRISC layout: data2/brisc2025/classification_task/{train,test}/<class>/
    # Allow either "<data2-dir>" pointing at the unzipped root or at the
    # classification_task subdir.
    candidates = [
        data2_dir / "brisc2025" / "classification_task",
        data2_dir / "classification_task",
        data2_dir,
    ]
    base = next((c for c in candidates if (c / "train").is_dir()), None)
    if base is None:
        state.warnings.append(
            f"data2: could not locate train/ subdir under {data2_dir} for class {cls}"
        )
        return

    train_pool: list[Path] = []
    test_files: list[Path] = []
    for split_src, folders in sources["data2"].items():
        for folder in folders:
            files = list_images(base / split_src / folder)
            if split_src == "train":
                train_pool.extend(files)
            else:
                test_files.extend(files)

    # File-level val carve-out -- BRISC filenames also lack a stable
    # patient ID convention; documented in the dataset card.
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * VAL_FRACTION_OF_TRAIN))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    state.emit(train_files, cls, "train", prefix="d2")
    state.emit(val_files, cls, "val", prefix="d2")
    state.emit(test_files, cls, "test", prefix="d2")

    state.methodology.setdefault(cls, {})["data2"] = {
        "split": "preserved-test, file-level val carve-out",
        "reason": (
            "BRISC ships with its own train/test split. Patient IDs are "
            "not encoded in the filenames in a stable way across folders, "
            "so the val carve-out is file-level."
        ),
    }


def build_data44(
    cls: str,
    sources: dict,
    data44_dir: Path | None,
    state: BuildState,
    rng: random.Random,
) -> None:
    if "data44" not in sources or data44_dir is None:
        return
    state.sources_used.add("data44")

    # Collect all images for this class across its (tumor, sequence) folders.
    all_files: list[Path] = []
    for folder in sources["data44"]:
        all_files.extend(list_images(data44_dir / folder))

    if not all_files:
        state.warnings.append(
            f"data44: no images found for class '{cls}' under {data44_dir}"
        )
        return

    file_to_pid, strategy, reliable = infer_patient_ids(all_files)

    if reliable:
        patient_to_files: dict[str, list[Path]] = defaultdict(list)
        for f, pid in file_to_pid.items():
            patient_to_files[pid].append(f)
        train_files, val_files, test_files, pcounts = split_patients(
            patient_to_files,
            train_frac=DATA44_TRAIN_FRAC,
            val_frac=DATA44_VAL_FRAC,
            rng=rng,
        )
        state.data44_patient_counts[cls] = pcounts
        state.methodology.setdefault(cls, {})["data44"] = {
            "split": f"patient-level 70/15/15 via '{strategy}' filename heuristic",
            "reason": (
                "Inferred patient IDs from filename digit runs / leading "
                "tokens, then split unique patients (not files) so all "
                "slices and all sequences from one patient stay in one "
                "split. Heuristic; not guaranteed perfect -- see script "
                "docstring."
            ),
        }
    else:
        train_files, val_files, test_files = split_files(
            all_files,
            train_frac=DATA44_TRAIN_FRAC,
            val_frac=DATA44_VAL_FRAC,
            rng=rng,
        )
        msg = (
            f"data44/{cls}: could not infer patient IDs from filenames "
            f"(every file is its own group). Falling back to image-level "
            f"70/15/15 split. Patient leakage may remain for this class."
        )
        state.warnings.append(msg)
        state.methodology.setdefault(cls, {})["data44"] = {
            "split": "image-level 70/15/15 (FALLBACK)",
            "reason": (
                "No filename pattern produced any grouping; treated as "
                "image-level split. Possible patient-level leakage."
            ),
        }

    state.emit(train_files, cls, "train", prefix="d3")
    state.emit(val_files, cls, "val", prefix="d3")
    state.emit(test_files, cls, "test", prefix="d3")


# ---------------------------------------------------------------------------
# Dataset card / printout
# ---------------------------------------------------------------------------

def print_dataset_card(state: BuildState) -> None:
    print()
    print("=" * 64)
    print(f"NeuroScan clean dataset — build summary [{state.label}]")
    print(f"output: {state.output_dir}")
    print("=" * 64)

    classes = sorted(state.classes)
    splits = ("train", "val", "test")

    header = f"{'class':>14s} | " + " | ".join(f"{s:>7s}" for s in splits)
    print(header)
    print("-" * len(header))
    totals = {s: 0 for s in splits}
    for cls in classes:
        row = f"{cls:>14s} | " + " | ".join(
            f"{state.counts[cls][s]:>7d}" for s in splits
        )
        print(row)
        for s in splits:
            totals[s] += state.counts[cls][s]
    print("-" * len(header))
    print(f"{'TOTAL':>14s} | " + " | ".join(f"{totals[s]:>7d}" for s in splits))

    if state.data44_patient_counts:
        print()
        print("data44 unique patient counts (where IDs were inferable):")
        for cls in sorted(state.data44_patient_counts):
            pc = state.data44_patient_counts[cls]
            print(
                f"  {cls:>14s}  train={pc['train_patients']:>4d}  "
                f"val={pc['val_patients']:>4d}  test={pc['test_patients']:>4d}"
            )

    # Sparse-test-class warning
    print()
    sparse = [cls for cls in classes if state.counts[cls]["test"] < 50]
    if sparse:
        print("WARNING: the following classes have FEWER THAN 50 test images;")
        print("         test metrics for these classes will be very noisy:")
        for cls in sparse:
            print(f"           - {cls} (test n={state.counts[cls]['test']})")
        for cls in sparse:
            state.warnings.append(
                f"class '{cls}' has only {state.counts[cls]['test']} test "
                f"images (<50); per-class test metrics will be unreliable."
            )

    # Source-confound warning for data44-only classes
    confounded = [c for c in classes if c in DATA44_ONLY_CLASSES
                  and state.counts[c]["train"] > 0]
    if confounded:
        print()
        print("WARNING: the following classes are sourced ONLY from data44.")
        print("         The model may learn data44 scanner/source artefacts")
        print("         instead of tumour morphology -- accuracy on these")
        print("         classes is NOT trustworthy as a generalisation signal:")
        for cls in confounded:
            print(f"           - {cls}")
        state.warnings.append(
            "source confound: classes "
            + ", ".join(sorted(confounded))
            + " come only from data44; risk of learning source artefacts."
        )

    if state.warnings:
        print()
        print("All warnings:")
        for w in state.warnings:
            print(f"  - {w}")

    print("=" * 64)


def write_dataset_card_json(
    state: BuildState,
    seed: int,
    allowed_sources: tuple[str, ...],
) -> None:
    card = {
        "label": state.label,
        "classes": list(state.classes),
        "allowed_sources": list(allowed_sources),
        "sources_used": sorted(state.sources_used),
        "seed": seed,
        "split_fractions": {
            "data44": {
                "train": DATA44_TRAIN_FRAC,
                "val": DATA44_VAL_FRAC,
                "test": DATA44_TEST_FRAC,
            },
            "data1_data2_val_carveout_from_train": VAL_FRACTION_OF_TRAIN,
        },
        "methodology_per_class": state.methodology,
        "counts_per_class_per_split": {
            cls: dict(state.counts[cls]) for cls in sorted(state.classes)
        },
        "data44_patient_counts": state.data44_patient_counts,
        "warnings": state.warnings,
        "notes": [
            "Filenames are prefixed with d1_/d2_/d3_ to identify the source "
            "dataset (data1/data2/data44). For the trustworthy 4-class build, "
            "this lets a training script run cross-source evaluation "
            "(train on d1_*, test on d2_*, and vice versa) by filtering on "
            "the prefix.",
        ],
    }
    out_path = state.output_dir / "dataset_card.json"
    out_path.write_text(json.dumps(card, indent=2, sort_keys=True))
    print(f"\nWrote dataset card: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a clean train/val/test dataset for NeuroScan.",
    )
    p.add_argument("--data44-dir", type=Path, default=None,
                   help="Path to the unzipped data44 root (folders like "
                        "'Astrocitoma T1', 'Schwannoma T1C+', ...).")
    p.add_argument("--data1-dir", type=Path, default=None,
                   help="Path to the unzipped data1 root (contains "
                        "Training/ and Testing/).")
    p.add_argument("--data2-dir", type=Path, default=None,
                   help="Path to the unzipped BRISC data2 root.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the {train,val,test}/<class>/ tree. "
                        "When --mode=both, two subdirs "
                        "(trustworthy_4class/, extended_8class/) are "
                        "created underneath this path.")
    p.add_argument("--mode", choices=("trustworthy", "extended", "both"),
                   default="both",
                   help="trustworthy: 4 classes from data1+data2 only "
                        "(defensible methodology, what to report as the "
                        "headline result). "
                        "extended: 8 classes including data44 (matches the "
                        "deployed backend; treat data44-only class metrics "
                        "as exploratory due to source confound and possible "
                        "patient leakage). "
                        "both (default): build both into subdirs.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for patient/file shuffling (default: 42).")
    return p.parse_args(argv)


def build_one_dataset(
    output_dir: Path,
    classes: tuple[str, ...],
    allowed_sources: tuple[str, ...],
    label: str,
    args: argparse.Namespace,
) -> BuildState:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_split_dirs(output_dir, classes)

    state = BuildState(output_dir, classes=classes, label=label)

    for cls in classes:
        sources = EXPANDED_CLASSES[cls]
        # Per-class RNG so the order in which sources are processed does
        # not change the data44 patient assignment.
        cls_rng = random.Random(args.seed + hash(cls) % (2**31))
        if "data1" in allowed_sources:
            build_data1(cls, sources, args.data1_dir, state, cls_rng)
        if "data2" in allowed_sources:
            build_data2(cls, sources, args.data2_dir, state, cls_rng)
        if "data44" in allowed_sources:
            build_data44(cls, sources, args.data44_dir, state, cls_rng)

    print_dataset_card(state)
    write_dataset_card_json(state, seed=args.seed,
                            allowed_sources=allowed_sources)
    return state


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.data44_dir is None and args.data1_dir is None and args.data2_dir is None:
        print("ERROR: at least one of --data44-dir / --data1-dir / --data2-dir "
              "must be provided.", file=sys.stderr)
        return 2

    base_output: Path = args.output_dir

    builds: list[tuple[Path, tuple[str, ...], tuple[str, ...], str]] = []

    if args.mode in ("trustworthy", "both"):
        out = (base_output / "trustworthy_4class"
               if args.mode == "both" else base_output)
        builds.append((
            out,
            TRUSTWORTHY_CLASSES,
            ("data1", "data2"),
            "trustworthy_4class",
        ))

    if args.mode in ("extended", "both"):
        out = (base_output / "extended_8class"
               if args.mode == "both" else base_output)
        builds.append((
            out,
            EXTENDED_CLASSES,
            ("data1", "data2", "data44"),
            "extended_8class",
        ))

    for out, classes, allowed_sources, label in builds:
        build_one_dataset(out, classes, allowed_sources, label, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
