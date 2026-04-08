"""Training loop driven by config.

Reproduces the logic in training/train_8class_v5.ipynb and
training/train_glioma_subtype.ipynb. The architecture and training math are
deliberately unchanged from those notebooks — see ARCHITECTURE.md and
config.yaml for the parameter set.

Use the `stage` field under `training:` in config.yaml to select which
stage's class_names / num_classes / dropout / weights_path to use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from .config import resolve_stage_config, DEFAULT_CONFIG_PATH
from .data import build_dataloaders
from .evaluation import evaluate, log_per_class_metrics
from .logging_setup import get_logger
from .models import build_efficientnet

_log = get_logger("training")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(config: dict[str, Any]) -> Path:
    """Train a stage according to `config`. Returns the path to the saved weights."""
    train_cfg = config["training"]
    stage_key = train_cfg["stage"]
    stage_cfg = resolve_stage_config(config, stage_key)

    num_classes = stage_cfg["num_classes"]
    dropout = stage_cfg["dropout"]
    expected_classes = list(stage_cfg["class_names"])
    weights_path = DEFAULT_CONFIG_PATH.parent / stage_cfg["weights_path"]

    device = _resolve_device(train_cfg["device"])
    _log.info("Training stage=%s on device=%s", stage_key, device)

    # ---- Data ---------------------------------------------------------------
    train_loader, test_loader, dataset_classes = build_dataloaders(config)

    if dataset_classes != expected_classes:
        raise ValueError(
            f"ImageFolder produced classes {dataset_classes} but config "
            f"declares {expected_classes}. Fix the data layout or update "
            f"config.yaml — refusing to silently train on a permuted label space."
        )

    # ---- Model --------------------------------------------------------------
    model = build_efficientnet(num_classes=num_classes, dropout=dropout, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["scheduler"]["t_max"])
    criterion = nn.CrossEntropyLoss()

    epochs = train_cfg["epochs"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        eval_result = evaluate(model, test_loader, criterion, device, expected_classes)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        _log.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% | lr=%.6f",
            epoch + 1, epochs, train_loss, train_acc,
            eval_result.loss, eval_result.accuracy, lr,
        )
        log_per_class_metrics(eval_result, prefix=f"epoch{epoch+1}")

    # ---- Save ---------------------------------------------------------------
    model.cpu()
    torch.save(model.state_dict(), weights_path)
    _log.info("Saved trained weights to %s", weights_path)
    return weights_path
