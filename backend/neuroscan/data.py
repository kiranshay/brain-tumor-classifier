"""Data module: MRI preprocessing, transforms, and dataloaders.

All values are sourced from the loaded config dict — nothing is hardcoded.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .logging_setup import get_logger

_log = get_logger("data")


# ---------------------------------------------------------------------------
# Inference-time MRI preprocessing
# ---------------------------------------------------------------------------

def crop_mri_borders(image: Image.Image, threshold: int, pad: int) -> Image.Image:
    """Crop black borders from a scanned MRI.

    Converts to grayscale, masks rows/cols whose mean intensity exceeds
    `threshold`, then crops to the bounding box of the mask plus `pad` px.
    Returns the original image unchanged if no foreground is detected.
    """
    gray = image.convert("L")
    gray_np = np.array(gray)

    row_mask = gray_np.mean(axis=1) > threshold
    col_mask = gray_np.mean(axis=0) > threshold

    if not (row_mask.any() and col_mask.any()):
        return image

    rows = np.where(row_mask)[0]
    cols = np.where(col_mask)[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(image.height, bottom + pad)
    right = min(image.width, right + pad)

    return image.crop((left, top, right, bottom))


def build_inference_transform(config: dict[str, Any]) -> transforms.Compose:
    """Eval-time transform: resize → tensor → normalize."""
    model_cfg = config["model"]
    size = model_cfg["image_size"]
    mean = model_cfg["normalization"]["mean"]
    std = model_cfg["normalization"]["std"]

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Training-time transforms and dataloaders
# ---------------------------------------------------------------------------

def build_train_transform(config: dict[str, Any]) -> transforms.Compose:
    model_cfg = config["model"]
    aug = config["training"]["augmentation"]
    mean = model_cfg["normalization"]["mean"]
    std = model_cfg["normalization"]["std"]
    cj = aug["color_jitter"]

    pipeline: list = [
        transforms.Resize((aug["resize"], aug["resize"])),
        transforms.RandomCrop(aug["crop"]),
    ]
    if aug.get("horizontal_flip", True):
        pipeline.append(transforms.RandomHorizontalFlip())
    if aug.get("vertical_flip", True):
        pipeline.append(transforms.RandomVerticalFlip())

    pipeline += [
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.RandomAffine(
            degrees=0,
            translate=tuple(aug["affine_translate"]),
            scale=tuple(aug["affine_scale"]),
        ),
        transforms.ColorJitter(
            brightness=cj["brightness"],
            contrast=cj["contrast"],
            saturation=cj["saturation"],
        ),
        transforms.RandomGrayscale(p=aug["grayscale_p"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=aug["erasing_p"]),
    ]
    return transforms.Compose(pipeline)


def build_eval_transform(config: dict[str, Any]) -> transforms.Compose:
    """Training-time eval transform (no border crop — that's an inference-only step)."""
    return build_inference_transform(config)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train/test dataloaders from config.

    Returns (train_loader, test_loader, class_names_from_dataset).
    The caller is responsible for asserting class_names matches the configured
    stage's class_names (see neuroscan.training.train).
    """
    train_cfg = config["training"]
    train_dir = train_cfg["data"]["train_dir"]
    test_dir = train_cfg["data"]["test_dir"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]

    train_ds = datasets.ImageFolder(train_dir, transform=build_train_transform(config))
    test_ds = datasets.ImageFolder(test_dir, transform=build_eval_transform(config))

    _log.info("Loaded train dataset: %d images, classes=%s", len(train_ds), train_ds.classes)
    _log.info("Loaded test dataset:  %d images", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, list(train_ds.classes)
