import io
import time

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
GLIOMA_SUBTYPES = ["astrocytoma", "ependymoma", "glioblastoma", "oligodendroglioma"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Global model references, loaded once at startup
_model = None
_glioma_model = None


def _build_efficientnet(num_classes: int, dropout: float = 0.3):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes),
    )
    return model


def load_model(
    weights_path: str = "tumor_classifier.pth",
    glioma_weights_path: str = "glioma_subtype_classifier.pth",
):
    global _model, _glioma_model

    # Stage 1: broad classifier (4 classes)
    _model = _build_efficientnet(4, dropout=0.3)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    _model.load_state_dict(state_dict)
    _model.eval()

    # Stage 2: glioma subtype classifier (4 subtypes)
    try:
        _glioma_model = _build_efficientnet(4, dropout=0.4)
        state_dict = torch.load(glioma_weights_path, map_location="cpu", weights_only=True)
        _glioma_model.load_state_dict(state_dict)
        _glioma_model.eval()
        print("Glioma subtype model loaded.")
    except FileNotFoundError:
        _glioma_model = None
        print("Glioma subtype model not found, skipping stage 2.")


def preprocess_mri(image: Image.Image) -> Image.Image:
    """Crop black borders from MRI images."""
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

    # Stage 1: broad classification
    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    all_confidences = {
        CLASS_NAMES[i]: round(probabilities[i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }

    # Stage 2: glioma subtype classification
    subtype = None
    subtype_confidence = None
    subtype_confidences = None

    if predicted_class == "glioma" and _glioma_model is not None:
        with torch.no_grad():
            sub_outputs = _glioma_model(tensor)
            sub_probs = torch.softmax(sub_outputs, dim=1)[0]

        sub_idx = sub_probs.argmax().item()
        subtype = GLIOMA_SUBTYPES[sub_idx]
        subtype_confidence = round(sub_probs[sub_idx].item(), 4)
        subtype_confidences = {
            GLIOMA_SUBTYPES[i]: round(sub_probs[i].item(), 4)
            for i in range(len(GLIOMA_SUBTYPES))
        }

    inference_time_ms = (time.perf_counter() - start) * 1000

    result = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_confidences": all_confidences,
        "inference_time_ms": round(inference_time_ms, 2),
    }

    if subtype is not None:
        result["subtype"] = subtype
        result["subtype_confidence"] = subtype_confidence
        result["subtype_confidences"] = subtype_confidences

    return result
