"""Inference module: Classifier abstraction + cascade.

Replaces the legacy module-level globals (`_model`, `_glioma_model`) and the
duplicated decode/preprocess code in `predict()` / `predict_with_gradcam()`.

Behavior is unchanged: stage 1 always runs, stage 2 only fires when stage 1
predicts the configured trigger class (e.g. `glioma`).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from .data import build_inference_transform, crop_mri_borders
from .gradcam import generate_gradcam
from .logging_setup import get_logger
from .models import load_classifier
from .uncertainty import predict_with_uncertainty as _predict_with_uncertainty

_log = get_logger("inference")


@dataclass
class Classifier:
    """A single classification stage: model + class names."""
    name: str
    model: nn.Module
    class_names: list[str]
    dropout: float

    def forward_softmax(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(tensor)
            return torch.softmax(outputs, dim=1)[0]


class CascadeInference:
    """Two-stage cascade inference engine.

    Holds a primary `Classifier` and an optional secondary classifier that
    fires only when the primary's predicted class matches the configured
    `trigger_class_for_stage2`. Encapsulates preprocessing so the FastAPI
    endpoints don't have to know about it.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.image_size = config["model"]["image_size"]
        self.mean = config["model"]["normalization"]["mean"]
        self.std = config["model"]["normalization"]["std"]
        self.border_threshold = config["preprocessing"]["border_crop_threshold"]
        self.border_pad = config["preprocessing"]["border_crop_pad"]

        self.transform = build_inference_transform(config)

        # ---- Stage 1 (always loaded) ---------------------------------------
        s1 = config["model"]["stage1"]
        from .config import DEFAULT_CONFIG_PATH  # local import to avoid cycle
        backend_dir = DEFAULT_CONFIG_PATH.parent
        s1_weights = backend_dir / s1["weights_path"]

        self.stage1 = Classifier(
            name=s1["name"],
            model=load_classifier(
                weights_path=s1_weights,
                num_classes=s1["num_classes"],
                dropout=s1["dropout"],
                class_names=s1["class_names"],
            ),
            class_names=list(s1["class_names"]),
            dropout=s1["dropout"],
        )
        self.trigger_class = s1.get("trigger_class_for_stage2")

        # ---- Stage 2 (optional) --------------------------------------------
        self.stage2: Classifier | None = None
        s2 = config["model"].get("stage2")
        if s2 is not None:
            s2_weights = backend_dir / s2["weights_path"]
            try:
                self.stage2 = Classifier(
                    name=s2["name"],
                    model=load_classifier(
                        weights_path=s2_weights,
                        num_classes=s2["num_classes"],
                        dropout=s2["dropout"],
                        class_names=s2["class_names"],
                    ),
                    class_names=list(s2["class_names"]),
                    dropout=s2["dropout"],
                )
                _log.info("Stage 2 (%s) loaded.", s2["name"])
            except FileNotFoundError:
                _log.warning(
                    "Stage 2 weights not found at %s — disabling cascade stage 2.",
                    s2_weights,
                )

    # ---- internal helpers --------------------------------------------------

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = crop_mri_borders(image, threshold=self.border_threshold, pad=self.border_pad)
        return self.transform(image).unsqueeze(0)

    @staticmethod
    def _round_dict(probs: torch.Tensor, names: list[str]) -> dict[str, float]:
        return {names[i]: round(float(probs[i].item()), 4) for i in range(len(names))}

    # ---- public API --------------------------------------------------------

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run the cascade on raw image bytes. Mirrors the legacy contract."""
        tensor = self._preprocess(image_bytes)

        start = time.perf_counter()
        probs = self.stage1.forward_softmax(tensor)
        predicted_idx = int(probs.argmax().item())
        predicted_class = self.stage1.class_names[predicted_idx]
        confidence = float(probs[predicted_idx].item())
        all_confidences = self._round_dict(probs, self.stage1.class_names)

        result: dict[str, Any] = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_confidences": all_confidences,
        }

        if (
            self.stage2 is not None
            and self.trigger_class is not None
            and predicted_class == self.trigger_class
        ):
            sub_probs = self.stage2.forward_softmax(tensor)
            sub_idx = int(sub_probs.argmax().item())
            result["subtype"] = self.stage2.class_names[sub_idx]
            result["subtype_confidence"] = round(float(sub_probs[sub_idx].item()), 4)
            result["subtype_confidences"] = self._round_dict(sub_probs, self.stage2.class_names)

        result["inference_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    def predict_with_uncertainty(
        self, image_bytes: bytes, n_samples: int = 50
    ) -> dict[str, Any]:
        """Run stage-1 MC-Dropout inference on raw image bytes.

        Mirrors the standard `predict()` contract (predicted_class, confidence,
        all_confidences, inference_time_ms) and additionally returns:
            uncertainty: predictive entropy of the mean MC-Dropout distribution.
            variance_per_class: per-class variance across the n_samples passes.

        Only stage 1 is run — the cascade's stage 2 is skipped because it would
        require its own MC-Dropout pass and complicate the uncertainty contract.

        Why predictive entropy: it is a single bounded scalar that captures
        total predictive uncertainty (data + model), which is what a downstream
        "decline-to-predict" rule actually needs to threshold on.
        """
        tensor = self._preprocess(image_bytes)

        start = time.perf_counter()
        unc = _predict_with_uncertainty(
            self.stage1.model, tensor, self.stage1.class_names, n_samples=n_samples,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "predicted_class": unc["predicted_class"],
            "confidence": round(unc["confidence"], 4),
            "all_confidences": unc["all_confidences"],
            "uncertainty": round(unc["uncertainty"], 4),
            "variance_per_class": {
                k: round(v, 6) for k, v in unc["variance_per_class"].items()
            },
            "inference_time_ms": elapsed_ms,
            "n_samples": n_samples,
        }

    def predict_with_gradcam(self, image_bytes: bytes) -> dict[str, Any]:
        """Run inference and return Grad-CAM heatmaps for stage 1 (and stage 2 if it fires)."""
        tensor = self._preprocess(image_bytes)

        probs = self.stage1.forward_softmax(tensor)
        predicted_idx = int(probs.argmax().item())
        predicted_class = self.stage1.class_names[predicted_idx]

        gradcam_b64 = generate_gradcam(
            self.stage1.model, tensor, predicted_idx,
            image_size=self.image_size, mean=self.mean, std=self.std,
        )
        self.stage1.model.eval()

        result: dict[str, Any] = {"gradcam": gradcam_b64}

        if (
            self.stage2 is not None
            and self.trigger_class is not None
            and predicted_class == self.trigger_class
        ):
            sub_probs = self.stage2.forward_softmax(tensor)
            sub_idx = int(sub_probs.argmax().item())
            result["subtype_gradcam"] = generate_gradcam(
                self.stage2.model, tensor, sub_idx,
                image_size=self.image_size, mean=self.mean, std=self.std,
            )
            self.stage2.model.eval()

        return result
