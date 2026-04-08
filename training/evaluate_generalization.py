"""Cross-dataset generalization experiment for the NeuroScan stage-1 classifier.

This script measures how well the deployed 8-class brain-tumor classifier
generalizes to an MRI dataset it was never trained on. Distribution shift is
the central failure mode of medical imaging models in deployment: scanner
vendor, acquisition protocol, slice thickness, contrast agent dose, and
patient demographics all change between hospitals, and a model whose accuracy
is only ever reported on a single in-distribution test split tells you very
little about how it will behave on a new site. ARCHITECTURE.md additionally
flags patient-level data leakage in the V5 training split, which means even
the in-distribution number is optimistic. A second, fully held-out dataset
gives us a much harder lower bound on real-world accuracy.

The script runs two evaluations:

  1. Zero-shot: load the stage-1 weights and evaluate on dataset-b without
     any adaptation. This is the closest analogue to "drop the model into a
     new hospital and see what happens."

  2. Few-shot fine-tune: take a small fraction of dataset-b, freeze the
     EfficientNet backbone, and fine-tune only the Dropout+Linear head for
     10 epochs with Adam(lr=1e-3). Evaluate on the held-out remainder. This
     simulates the realistic deployment workflow where a new site can collect
     a small labeled set for adaptation but cannot afford a full retrain.

For both evaluations we record overall accuracy, sklearn classification_report,
the confusion matrix, and the mean MC-Dropout predictive entropy from
neuroscan.uncertainty.predict_with_uncertainty. The uncertainty number is the
most informative single signal of distribution shift: a well-behaved model
should become measurably *less confident* on out-of-distribution data, even
when its accuracy holds up.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Make the backend/ package importable so we can use neuroscan.* from here.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from neuroscan.config import load_config, resolve_weights_path  # noqa: E402
from neuroscan.data import build_inference_transform  # noqa: E402
from neuroscan.logging_setup import get_logger, setup_logging  # noqa: E402
from neuroscan.models import load_classifier  # noqa: E402
from neuroscan.uncertainty import predict_with_uncertainty  # noqa: E402


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) to the same value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_label_remap(
    dataset_classes: list[str], stage1_classes: list[str]
) -> dict[int, int]:
    """Map dataset-b ImageFolder indices to stage-1 class indices.

    Every dataset-b class folder must be a stage-1 class name. Missing
    stage-1 classes in dataset-b are fine (the model just isn't tested on
    them); extra dataset-b classes that aren't in stage-1 are a hard error
    because the head has no output for them.
    """
    remap: dict[int, int] = {}
    for idx, name in enumerate(dataset_classes):
        if name not in stage1_classes:
            raise ValueError(
                f"dataset-b class folder {name!r} is not in stage1.class_names "
                f"{stage1_classes}. The stage-1 head has no output for it."
            )
        remap[idx] = stage1_classes.index(name)
    return remap


def split_indices(
    n: int, finetune_split: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic shuffle-and-split of [0, n) into (finetune, test) lists."""
    if not 0.0 <= finetune_split < 1.0:
        raise ValueError(
            f"finetune_split must be in [0, 1), got {finetune_split}"
        )
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_finetune = int(round(n * finetune_split))
    return indices[:n_finetune], indices[n_finetune:]


def evaluate(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    indices: list[int],
    label_remap: dict[int, int],
    stage1_classes: list[str],
    device: torch.device,
    n_mc_samples: int,
) -> dict[str, Any]:
    """Run per-image MC-Dropout prediction over `indices` and compute metrics.

    Returns a dict with: accuracy, mean_uncertainty, classification_report
    string, confusion matrix, and the label index/name lists used for those.
    """
    model.to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    uncertainties: list[float] = []

    for sample_idx in indices:
        image, raw_label = dataset[sample_idx]
        true_idx = label_remap[raw_label]
        tensor = image.unsqueeze(0).to(device)

        result = predict_with_uncertainty(
            model, tensor, stage1_classes, n_samples=n_mc_samples
        )
        y_true.append(true_idx)
        y_pred.append(result["predicted_index"])
        uncertainties.append(result["uncertainty"])

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)

    accuracy = (
        float(accuracy_score(y_true_arr, y_pred_arr)) if y_true_arr.size else 0.0
    )
    mean_unc = float(np.mean(uncertainties)) if uncertainties else 0.0

    # Restrict the report/matrix to classes that actually appear (in either
    # ground truth or predictions) so the report stays readable, but anchor
    # the labels in the canonical stage-1 name space.
    present_indices = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    target_names = [stage1_classes[i] for i in present_indices]
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=present_indices,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=present_indices)

    return {
        "accuracy": accuracy,
        "mean_uncertainty": mean_unc,
        "report": report,
        "confusion_matrix": cm,
        "label_indices": present_indices,
        "label_names": target_names,
    }


def save_confusion_heatmap(
    cm: np.ndarray, label_names: list[str], output_path: Path, title: str
) -> None:
    """Render a confusion-matrix heatmap with cell counts annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(label_names), 1)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    cm_max = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > cm_max / 2 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def freeze_backbone_train_head(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    finetune_indices: list[int],
    label_remap: dict[int, int],
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> None:
    """Freeze every parameter except the classifier head, then train it.

    Mutates `model` in place. ImageFolder labels are remapped to stage-1
    indices on the fly so the head's output space stays consistent with the
    deployed model.
    """
    log = get_logger("evaluate_generalization")

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    subset = Subset(dataset, finetune_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)

    max_raw = max(label_remap.keys())
    remap_tensor = torch.full((max_raw + 1,), -1, dtype=torch.long)
    for k, v in label_remap.items():
        remap_tensor[k] = v
    remap_tensor = remap_tensor.to(device)

    model.to(device)
    model.train()  # Re-enables Dropout, which is correct for fine-tuning.
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        seen = 0
        for inputs, raw_labels in loader:
            inputs = inputs.to(device)
            labels = remap_tensor[raw_labels.to(device)]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * inputs.size(0)
            seen += inputs.size(0)
        log.info(
            "fine-tune epoch %d/%d  loss=%.4f  n=%d",
            epoch, epochs, running_loss / max(seen, 1), seen,
        )
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-dataset generalization eval for the stage-1 NeuroScan classifier."
        ),
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to backend/config.yaml.",
    )
    parser.add_argument(
        "--dataset-b", type=str, required=True,
        help="Path to a held-out ImageFolder-compatible MRI dataset directory.",
    )
    parser.add_argument(
        "--finetune-split", type=float, default=0.2,
        help="Fraction of dataset-b reserved for fine-tuning (default 0.2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits and torch/numpy/random (default 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/generalization",
        help="Where to write reports, plots, and the comparison JSON.",
    )
    args = parser.parse_args()

    set_all_seeds(args.seed)

    config = load_config(args.config)
    setup_logging(config)
    log = get_logger("evaluate_generalization")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s1 = config["model"]["stage1"]
    stage1_classes: list[str] = list(s1["class_names"])
    weights_path = resolve_weights_path(config, "stage1")
    log.info("Loading stage 1 classifier from %s", weights_path)

    transform = build_inference_transform(config)
    dataset_b = datasets.ImageFolder(args.dataset_b, transform=transform)
    log.info(
        "Loaded dataset-b: %d images, classes=%s",
        len(dataset_b), dataset_b.classes,
    )

    label_remap = build_label_remap(dataset_b.classes, stage1_classes)

    finetune_indices, test_indices = split_indices(
        len(dataset_b), args.finetune_split, args.seed,
    )
    log.info(
        "dataset-b split: %d fine-tune / %d test (finetune_split=%.3f, seed=%d)",
        len(finetune_indices), len(test_indices), args.finetune_split, args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_mc_samples = int(config.get("uncertainty", {}).get("n_samples", 50))

    # ------------------------------------------------------------------
    # Zero-shot evaluation
    # ------------------------------------------------------------------
    log.info("Zero-shot evaluation on %d held-out images...", len(test_indices))
    model_zs = load_classifier(
        weights_path=weights_path,
        num_classes=s1["num_classes"],
        dropout=s1["dropout"],
        class_names=stage1_classes,
    )
    zs = evaluate(
        model_zs, dataset_b, test_indices, label_remap,
        stage1_classes, device, n_mc_samples,
    )
    (output_dir / "zero_shot_report.txt").write_text(zs["report"])
    save_confusion_heatmap(
        zs["confusion_matrix"], zs["label_names"],
        output_dir / "zero_shot_confusion.png",
        title="Zero-shot confusion matrix (dataset-b)",
    )

    # ------------------------------------------------------------------
    # Few-shot fine-tuned evaluation
    # ------------------------------------------------------------------
    log.info(
        "Fine-tuning classifier head on %d images (10 epochs, Adam lr=1e-3)...",
        len(finetune_indices),
    )
    # Reload weights so the fine-tuned model is independent of the zero-shot one.
    model_ft = load_classifier(
        weights_path=weights_path,
        num_classes=s1["num_classes"],
        dropout=s1["dropout"],
        class_names=stage1_classes,
    )
    freeze_backbone_train_head(
        model_ft, dataset_b, finetune_indices, label_remap, device,
        epochs=10, lr=1e-3, batch_size=16,
    )
    log.info("Evaluating fine-tuned model on %d held-out images...", len(test_indices))
    ft = evaluate(
        model_ft, dataset_b, test_indices, label_remap,
        stage1_classes, device, n_mc_samples,
    )
    (output_dir / "finetuned_report.txt").write_text(ft["report"])
    save_confusion_heatmap(
        ft["confusion_matrix"], ft["label_names"],
        output_dir / "finetuned_confusion.png",
        title="Fine-tuned confusion matrix (dataset-b)",
    )

    # ------------------------------------------------------------------
    # Comparison artifact + stdout summary
    # ------------------------------------------------------------------
    comparison = {
        "zero_shot": {
            "accuracy": zs["accuracy"],
            "mean_uncertainty": zs["mean_uncertainty"],
        },
        "finetuned": {
            "accuracy": ft["accuracy"],
            "mean_uncertainty": ft["mean_uncertainty"],
        },
        "delta_accuracy": ft["accuracy"] - zs["accuracy"],
    }
    with open(output_dir / "comparison_table.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("=" * 64)
    print("Cross-dataset generalization summary (stage-1, dataset-b)")
    print("=" * 64)
    print(f"  dataset-b path:           {args.dataset_b}")
    print(f"  fine-tune / test split:   {len(finetune_indices)} / {len(test_indices)}")
    print(f"  MC samples per image:     {n_mc_samples}")
    print(f"  seed:                     {args.seed}")
    print("-" * 64)
    print(f"  {'':<14}{'accuracy':>14}{'mean H':>14}")
    print(f"  {'zero-shot':<14}{zs['accuracy']:>14.4f}{zs['mean_uncertainty']:>14.4f}")
    print(f"  {'fine-tuned':<14}{ft['accuracy']:>14.4f}{ft['mean_uncertainty']:>14.4f}")
    print(f"  {'delta':<14}{comparison['delta_accuracy']:>14.4f}")
    print("-" * 64)
    print(f"  artifacts written to:     {output_dir}/")


if __name__ == "__main__":
    main()
