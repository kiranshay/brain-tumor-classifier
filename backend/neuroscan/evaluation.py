"""Evaluation utilities: validation loop and per-class metrics.

Used by the training script to log per-epoch validation accuracy and
per-class metrics. Pure PyTorch — no sklearn dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .logging_setup import get_logger

_log = get_logger("evaluation")


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    per_class_accuracy: dict[str, float]
    per_class_count: dict[str, int]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> EvalResult:
    model.eval()
    n_classes = len(class_names)
    correct_per_class = [0] * n_classes
    total_per_class = [0] * n_classes
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += float(loss.item()) * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += int(predicted.eq(labels).sum().item())

        for label, pred in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
            total_per_class[label] += 1
            if label == pred:
                correct_per_class[label] += 1

    per_class_acc = {}
    per_class_count = {}
    for i, cls in enumerate(class_names):
        per_class_count[cls] = total_per_class[i]
        per_class_acc[cls] = (
            100.0 * correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0
        )

    return EvalResult(
        loss=total_loss / max(total, 1),
        accuracy=100.0 * correct / max(total, 1),
        per_class_accuracy=per_class_acc,
        per_class_count=per_class_count,
    )


def log_per_class_metrics(result: EvalResult, prefix: str = "val") -> None:
    for cls, acc in result.per_class_accuracy.items():
        n = result.per_class_count[cls]
        _log.info("%s per-class | %-18s acc=%5.1f%%  (n=%d)", prefix, cls, acc, n)
