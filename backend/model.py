import io
import time

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

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
    """Crop black borders from MRI images.

    Internet-sourced MRIs often have large black margins that the
    training data didn't have, which can confuse the model.
    """
    gray = image.convert("L")
    gray_np = np.array(gray)

    threshold = 15
    row_mask = gray_np.mean(axis=1) > threshold
    col_mask = gray_np.mean(axis=0) > threshold

    if row_mask.any() and col_mask.any():
        rows = np.where(row_mask)[0]
        cols = np.where(col_mask)[0]
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]

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
    image = preprocess_mri(image)
    tensor = inference_transforms(image).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    inference_time_ms = (time.perf_counter() - start) * 1000

    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    all_confidences = {
        CLASS_NAMES[i]: round(probabilities[i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_confidences": all_confidences,
        "inference_time_ms": round(inference_time_ms, 2),
    }
