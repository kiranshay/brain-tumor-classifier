"""Grad-CAM heatmap generation for EfficientNet-B0.

Logic is unchanged from the legacy backend/model.py:generate_gradcam — only
the normalization stats are now read from config rather than re-imported.
"""
from __future__ import annotations

import base64
import io
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    predicted_idx: int,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> str:
    """Generate a Grad-CAM heatmap and return it as a base64-encoded PNG string.

    Hooks the last conv block (`model.features[-1]`), backprops the score of
    `predicted_idx`, and overlays the resulting heatmap on the denormalized
    input image.
    """
    target_layer = model.features[-1]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor_grad = tensor.clone().requires_grad_(True)
        output = model(tensor_grad)
        score = output[0, predicted_idx]

        model.zero_grad()
        score.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1, 2))

    gradcam = torch.zeros(act.shape[1:], dtype=act.dtype)
    for i, w in enumerate(weights):
        gradcam += w * act[i]

    gradcam = F.relu(gradcam)
    if gradcam.max() > 0:
        gradcam = gradcam / gradcam.max()

    gradcam_np = gradcam.numpy()
    gradcam_resized = np.array(
        Image.fromarray((gradcam_np * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        )
    ) / 255.0

    img_np = tensor[0].permute(1, 2, 0).numpy()
    img_np = img_np * np.array(std) + np.array(mean)
    img_np = np.clip(img_np, 0, 1)

    heatmap = cm.jet(gradcam_resized)[:, :, :3]
    overlay = 0.55 * img_np + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
    ax.imshow(overlay)
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode("utf-8")
