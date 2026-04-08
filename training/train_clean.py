"""Train an EfficientNet-B0 brain-tumor classifier on a clean dataset.

Trains either the trustworthy 4-class model (data1+data2 only) or the
extended 8-class model (adds data44-only rare classes), depending on
``--mode``. Both runs share the exact same training protocol; only the
dataset path, the inferred ``num_classes``, and the output directory
differ.

This script fixes the gaps recorded in ARCHITECTURE.md "Known
Methodological Limitations":
  - sets every random seed (random / numpy / torch / cudnn / DataLoader)
  - uses train/, val/, test/ as three distinct sets and never touches
    test/ until the very end
  - saves the best-on-val checkpoint and runs final test eval against it
  - records per-class precision/recall/F1 every epoch to metrics.jsonl
  - persists the exact ImageFolder.classes order to label_map.json
  - emits a single run_summary.json suitable for pasting into a writeup

It does NOT modify backend/ -- deployment of the new weights is a
separate concern (see ARCHITECTURE.md and dataset_card.json next to the
clean dataset).

Usage
-----
    python training/train_clean.py \
        --mode trustworthy \
        --data-dir clean_dataset/trustworthy_4class \
        --output-dir runs/trustworthy_4class \
        --epochs 20 --batch-size 32 --lr 5e-5 \
        --weight-decay 0.01 --dropout 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("trustworthy", "extended", "leaky"),
                   required=True,
                   help="Build label, recorded in run_summary.json. "
                        "'leaky' is the methodological control for "
                        "'trustworthy' — see training/build_leaky_dataset.py.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing train/ val/ test/ subdirs.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints, metrics, plots.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> torch.Generator:
    """Set every RNG we touch and return a seeded generator for the train loader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    # Each DataLoader worker derives its own RNG from the base seed so
    # augmentations are reproducible across runs.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder,
           datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    # Sanity-check class consistency across splits -- ImageFolder uses
    # alphabetical order so this should always pass, but if a class is
    # missing from val/ or test/ the silent index drift would be a
    # nightmare to debug.
    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise RuntimeError(
            f"Class list mismatch across splits.\n"
            f"  train: {train_ds.classes}\n"
            f"  val:   {val_ds.classes}\n"
            f"  test:  {test_ds.classes}"
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, dropout: float) -> nn.Module:
    """EfficientNet-B0 with Dropout+Linear head, fully unfrozen.

    Head shape matches backend/neuroscan/models.py so checkpoints could
    in principle be loaded by the deployed inference path (with a
    matching CLASS_NAMES update).
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int], list[float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        total_loss += loss.item() * x.size(0)
        total_correct += (pred == y).sum().item()
        total_n += x.size(0)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())
    return (total_loss / total_n, total_correct / total_n,
            all_true, all_pred, all_conf)


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def parse_source_prefix(filename: str) -> str:
    """Extract the d1/d2/d3 source prefix produced by build_clean_dataset.py.

    Returns "unknown" if the filename does not start with the expected
    prefix (shouldn't happen for cleanly-built datasets, but we don't
    want the test eval to crash on a stray file).
    """
    base = os.path.basename(filename)
    head = base.split("_", 1)[0]
    if head in ("d1", "d2", "d3"):
        return head
    return "unknown"


def save_confusion_matrix(
    cm: np.ndarray, classes: list[str], path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(classes)),
                                    max(5, len(classes) * 0.9)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    # Annotate each cell with the count.
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curves(metrics_rows: list[dict], path: Path) -> None:
    epochs = [r["epoch"] for r in metrics_rows]
    train_loss = [r["train_loss"] for r in metrics_rows]
    val_loss = [r["val_loss"] for r in metrics_rows]
    val_acc = [r["val_acc"] for r in metrics_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_acc, color="tab:green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confused_pairs(cm: np.ndarray, classes: list[str], k: int = 3
                       ) -> list[dict]:
    """Off-diagonal cells with the highest counts."""
    pairs: list[tuple[int, int, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    out = []
    for count, i, j in pairs[:k]:
        out.append({
            "true": classes[i],
            "predicted": classes[j],
            "count": count,
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed] using seed={args.seed}")
    generator = set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} ({gpu_name})")

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, generator,
    )
    class_names: list[str] = list(train_ds.classes)
    num_classes = len(class_names)
    print(f"[data] num_classes={num_classes} classes={class_names}")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Persist label map immediately so we have it even if training crashes.
    (args.output_dir / "label_map.json").write_text(json.dumps({
        "classes": class_names,
        "num_classes": num_classes,
        "source": "ImageFolder.classes from train/ at training time",
    }, indent=2))

    model = build_model(num_classes=num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    criterion = nn.CrossEntropyLoss()

    metrics_path = args.output_dir / "metrics.jsonl"
    # Truncate any prior run's metrics file before we start appending.
    metrics_path.write_text("")

    best_val_loss = float("inf")
    best_epoch = -1
    metrics_rows: list[dict] = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_loss, val_acc, val_true, val_pred, _ = evaluate(
            model, val_loader, criterion, device,
        )
        # Step the scheduler on the epoch boundary -- cosine, no input
        # from val (val is only used for checkpoint selection).
        scheduler.step()

        per_class = classification_report(
            val_true, val_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        epoch_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_time,
            "val_per_class": per_class,
        }
        metrics_rows.append(row)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "best.pth")

        marker = " *" if improved else ""
        print(
            f"epoch {epoch:>3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={row['lr']:.2e}  ({epoch_time:.1f}s){marker}"
        )

    torch.save(model.state_dict(), args.output_dir / "last.pth")
    total_train_time = time.time() - train_start

    # ---- Final test evaluation against best checkpoint ----------------
    print(f"\n[test] loading best.pth from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    model.load_state_dict(torch.load(args.output_dir / "best.pth", map_location=device))
    test_loss, test_acc, test_true, test_pred, test_conf = evaluate(
        model, test_loader, criterion, device,
    )
    print(f"[test] test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    test_per_class = classification_report(
        test_true, test_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_true, test_pred,
                          labels=list(range(num_classes)))

    test_report = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": test_per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    (args.output_dir / "test_report.json").write_text(
        json.dumps(test_report, indent=2)
    )

    # Per-image predictions CSV. test_ds.samples preserves the order
    # ImageFolder used to enumerate files, which is the same order our
    # (shuffle=False) test loader walks them in.
    csv_path = args.output_dir / "test_predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_class", "predicted_class",
                         "confidence", "source_prefix"])
        for (path, _label), t, p, c in zip(
            test_ds.samples, test_true, test_pred, test_conf
        ):
            writer.writerow([
                os.path.basename(path),
                class_names[t],
                class_names[p],
                f"{c:.6f}",
                parse_source_prefix(path),
            ])

    save_confusion_matrix(cm, class_names,
                          args.output_dir / "confusion_matrix.png")
    save_training_curves(metrics_rows,
                         args.output_dir / "training_curves.png")

    # ---- Run summary --------------------------------------------------
    best_row = next((r for r in metrics_rows if r["epoch"] == best_epoch), None)
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "num_classes": num_classes,
        "class_names": class_names,
        "num_train_images": len(train_ds),
        "num_val_images": len(val_ds),
        "num_test_images": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_row["val_acc"] if best_row else None,
        "final_test_accuracy": test_acc,
        "final_test_loss": test_loss,
        "top_confused_pairs": top_confused_pairs(cm, class_names, k=3),
        "total_wall_clock_seconds": total_train_time,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "dataset_card_path": str(args.data_dir / "dataset_card.json"),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"\n[done] artifacts written to {args.output_dir}")
    print(f"  best epoch: {best_epoch}  best val_loss: {best_val_loss:.4f}")
    print(f"  final test accuracy: {test_acc:.4f}")
    print(f"  total wall-clock: {total_train_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
