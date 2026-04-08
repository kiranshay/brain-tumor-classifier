"""Model construction and checkpoint loading.

The deployed architecture (EfficientNet-B0 with a Dropout+Linear head) is
unchanged from the legacy backend. What's new here:

- A single `build_efficientnet` builder driven by config.
- `load_classifier` verifies the checkpoint's output dimension matches the
  configured num_classes BEFORE swapping it into the live model. This is the
  fix for the CLASS_NAMES ordering hazard called out in ARCHITECTURE.md
  ("No state_dict shape validation when loading weights").
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from .logging_setup import get_logger

_log = get_logger("models")


def build_efficientnet(num_classes: int, dropout: float, pretrained: bool = False) -> nn.Module:
    """Build an EfficientNet-B0 with a Dropout+Linear classification head."""
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes),
    )
    return model


def _checkpoint_num_classes(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Pull the output dimension out of the head's Linear layer in a state_dict.

    Returns None if the expected key isn't present (unknown architecture).
    """
    # Our head is `Sequential(Dropout, Linear)`, so the Linear weight is at index 1.
    key = "classifier.1.weight"
    if key in state_dict:
        return state_dict[key].shape[0]
    return None


def load_classifier(
    weights_path: str | Path,
    num_classes: int,
    dropout: float,
    class_names: list[str],
    map_location: str = "cpu",
) -> nn.Module:
    """Load an EfficientNet-B0 classifier from a checkpoint with shape assertions.

    Asserts:
        len(class_names) == num_classes
        checkpoint head out_features == num_classes

    Raises ValueError on mismatch — this is the runtime guard for the
    CLASS_NAMES ordering hazard.
    """
    if len(class_names) != num_classes:
        raise ValueError(
            f"Config inconsistency: num_classes={num_classes} but "
            f"len(class_names)={len(class_names)} ({class_names})"
        )

    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)

    ckpt_num_classes = _checkpoint_num_classes(state_dict)
    if ckpt_num_classes is None:
        raise ValueError(
            f"Could not determine output dimension from checkpoint at {weights_path}: "
            f"expected key 'classifier.1.weight' (Sequential[Dropout, Linear] head)."
        )
    if ckpt_num_classes != num_classes:
        raise ValueError(
            f"Checkpoint at {weights_path} has {ckpt_num_classes} output classes, "
            f"but config declares num_classes={num_classes} "
            f"(class_names={class_names}). Refusing to load — this is the "
            f"CLASS_NAMES ordering / wrong-checkpoint guard."
        )

    model = build_efficientnet(num_classes=num_classes, dropout=dropout, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    _log.info(
        "Loaded classifier from %s (num_classes=%d, classes=%s)",
        weights_path, num_classes, class_names,
    )
    return model
