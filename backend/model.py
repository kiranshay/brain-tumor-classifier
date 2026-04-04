import io
import time

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image, ImageFilter

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# TTA augmentations — each produces a slightly different view
tta_transforms = [
    # Original
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # Vertical flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # Slight rotation via crop
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # Tighter crop
    transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]

# Global model reference, loaded once at startup
_model = None


def load_model(weights_path: str = "tumor_classifier.pth"):
    global _model
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 4),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model


def preprocess_mri(image: Image.Image) -> Image.Image:
    """Clean up an MRI image before classification.

    Crops out black borders and removes non-brain regions that can
    confuse the model, especially with internet-sourced images.
    """
    # Convert to grayscale for border detection
    gray = image.convert("L")
    gray_np = np.array(gray)

    # Find rows and columns that aren't mostly black
    threshold = 15
    row_mask = gray_np.mean(axis=1) > threshold
    col_mask = gray_np.mean(axis=0) > threshold

    if row_mask.any() and col_mask.any():
        rows = np.where(row_mask)[0]
        cols = np.where(col_mask)[0]
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]

        # Add small padding
        pad = 5
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(image.height, bottom + pad)
        right = min(image.width, right + pad)

        image = image.crop((left, top, right, bottom))

    return image


def predict(image_bytes: bytes) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess: crop black borders
    image = preprocess_mri(image)

    start = time.perf_counter()

    # Test-time augmentation: run multiple views and average predictions
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(image).unsqueeze(0)
            outputs = _model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs)

    # Average probabilities across all augmented views
    avg_probs = torch.stack(all_probs).mean(dim=0)

    inference_time_ms = (time.perf_counter() - start) * 1000

    predicted_idx = avg_probs.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = avg_probs[predicted_idx].item()

    all_confidences = {
        CLASS_NAMES[i]: round(avg_probs[i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_confidences": all_confidences,
        "inference_time_ms": round(inference_time_ms, 2),
    }
