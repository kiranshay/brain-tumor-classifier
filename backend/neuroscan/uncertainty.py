"""Monte Carlo Dropout uncertainty quantification for NeuroScan.

This module adds epistemic-uncertainty estimation on top of the existing
EfficientNet-B0 classifiers without retraining. The trick is simple:

    At inference time, leave Dropout layers in *train* mode (so they stochastically
    drop units) but keep BatchNorm and the rest of the network in *eval* mode (so
    running statistics are unchanged). Run the same input through the model
    several times and look at the distribution of softmax outputs. The spread of
    that distribution is a Monte Carlo estimate of the model's uncertainty
    about the prediction.

We use **predictive entropy** of the *mean* softmax distribution as the scalar
uncertainty score:

    H(p_mean) = - sum_c p_mean[c] * log(p_mean[c] + eps)

Why predictive entropy specifically:
    - It captures *both* aleatoric (data) uncertainty and epistemic (model)
      uncertainty in one number, which is what users of a clinical classifier
      actually care about ("how confused is the model overall?").
    - It is bounded in [0, log(num_classes)], so it is comparable across inputs
      for a fixed model and easy to threshold for a "decline-to-predict" rule.
    - It is the standard MC-Dropout summary statistic in the literature
      (Gal & Ghahramani, 2016), so reported numbers are interpretable to anyone
      familiar with that work.

Variance per class is also returned for callers that want to inspect *which*
classes the stochastic forward passes disagree about — this is more diagnostic
than predictive entropy but harder to threshold on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .logging_setup import get_logger

_log = get_logger("uncertainty")

_EPS = 1e-9


def enable_mc_dropout(model: nn.Module) -> None:
    """Put every Dropout layer into train mode while leaving the rest in eval mode.

    This is the core MC-Dropout trick: BatchNorm running stats stay frozen
    (eval mode), but Dropout keeps stochastically zeroing units, so repeated
    forward passes on the same input produce different softmax outputs that
    sample from an approximate posterior over the model's predictions.

    Mutates `model` in place. Call `model.eval()` afterwards to restore
    deterministic inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predict_with_uncertainty(
    model: nn.Module,
    tensor: torch.Tensor,
    class_names: list[str],
    n_samples: int = 50,
) -> dict[str, Any]:
    """Run MC-Dropout inference on a single preprocessed image tensor.

    Performs `n_samples` stochastic forward passes through `model` with Dropout
    active, then summarizes the resulting softmax distribution.

    Args:
        model: A classifier whose head contains nn.Dropout layers (e.g. our
            EfficientNet-B0 with `Sequential(Dropout, Linear)` head).
        tensor: A preprocessed input tensor of shape (1, C, H, W).
        class_names: Ordered list of class labels matching the model's output dim.
        n_samples: Number of stochastic forward passes. Higher = tighter MC
            estimate but linearly more compute. 50 is a common default.

    Returns:
        A dict with:
            mean_probs: List[float] of length num_classes — sample-mean of softmax.
            predicted_class: str — argmax of mean_probs.
            predicted_index: int.
            confidence: float — mean_probs[predicted_index].
            uncertainty: float — predictive entropy of mean_probs (see module docstring).
            variance_per_class: Dict[str, float] — per-class variance across samples.
            all_confidences: Dict[str, float] — class_name -> mean_probs[i] (rounded).

    Why predictive entropy as the uncertainty score:
        It is a bounded scalar in [0, log(num_classes)] that captures total
        predictive uncertainty (both data noise and model disagreement) in
        one threshold-friendly number, which matches how a downstream
        "decline-to-predict" rule needs to consume it.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise ValueError(
            f"predict_with_uncertainty expects a single-image batch tensor "
            f"of shape (1, C, H, W), got {tuple(tensor.shape)}"
        )

    enable_mc_dropout(model)

    sampled_probs = torch.zeros(n_samples, len(class_names), dtype=torch.float32)
    with torch.no_grad():
        for i in range(n_samples):
            logits = model(tensor)
            sampled_probs[i] = torch.softmax(logits, dim=1)[0]

    # Restore deterministic eval mode for any subsequent caller.
    model.eval()

    mean_probs = sampled_probs.mean(dim=0)
    var_probs = sampled_probs.var(dim=0, unbiased=False)

    predicted_idx = int(mean_probs.argmax().item())
    predicted_class = class_names[predicted_idx]
    confidence = float(mean_probs[predicted_idx].item())

    # Predictive entropy of the *mean* distribution.
    entropy = float(-(mean_probs * torch.log(mean_probs + _EPS)).sum().item())

    return {
        "mean_probs": [float(p) for p in mean_probs.tolist()],
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "confidence": confidence,
        "uncertainty": entropy,
        "variance_per_class": {
            class_names[i]: float(var_probs[i].item()) for i in range(len(class_names))
        },
        "all_confidences": {
            class_names[i]: round(float(mean_probs[i].item()), 4)
            for i in range(len(class_names))
        },
    }


def _save_calibration_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Render a reliability diagram (mean confidence vs. empirical accuracy per bin).

    A perfectly calibrated classifier lies on the y=x diagonal: when it says
    "70% confident", it is right 70% of the time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            bin_conf[b] = confidences[mask].mean()
            bin_acc[b] = correct[mask].mean()
            bin_count[b] = mask.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    nonempty = bin_count > 0
    ax.plot(bin_conf[nonempty], bin_acc[nonempty], marker="o", label="model")
    ax.set_xlabel("Mean predicted confidence (per bin)")
    ax.set_ylabel("Empirical accuracy (per bin)")
    ax.set_title("Calibration curve (MC-Dropout)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_uncertainty_histogram(
    uncertainties: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
) -> None:
    """Histogram of predictive entropy split by correct vs. incorrect predictions.

    A useful uncertainty measure should put more mass at higher entropy for
    *incorrect* predictions than for correct ones — i.e. the model should be
    less sure when it is wrong.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    correct_u = uncertainties[correct.astype(bool)]
    wrong_u = uncertainties[~correct.astype(bool)]
    if uncertainties.size > 0:
        max_u = float(uncertainties.max())
    else:
        max_u = 1.0
    bins = np.linspace(0.0, max(max_u, _EPS), 30)
    if correct_u.size > 0:
        ax.hist(correct_u, bins=bins, alpha=0.6, label=f"correct (n={correct_u.size})")
    if wrong_u.size > 0:
        ax.hist(wrong_u, bins=bins, alpha=0.6, label=f"incorrect (n={wrong_u.size})")
    ax.set_xlabel("Predictive entropy")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty distribution by correctness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_uncertainty_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    n_samples: int = 50,
    output_dir: str | Path = "outputs/uncertainty",
) -> list[dict[str, Any]]:
    """Run MC-Dropout uncertainty over an entire dataloader and dump diagnostics.

    For every image in `dataloader` (batches are unrolled to single examples so
    that the per-image MC sampling is straightforward), this function calls
    `predict_with_uncertainty`, then writes:

        - `<output_dir>/calibration_curve.png` — reliability diagram.
        - `<output_dir>/uncertainty_histogram.png` — entropy histogram split
          by correct/incorrect predictions.
        - `<output_dir>/uncertainty_summary.json` — summary statistics
          (mean uncertainty conditional on correctness, per-class accuracy,
          dataset size, n_samples).

    Returns:
        A list of per-image result dicts with keys:
            true_label, predicted_label, correct, confidence, uncertainty.

    Why predictive entropy is the uncertainty measure:
        See module docstring — bounded, captures total predictive uncertainty,
        and matches the standard MC-Dropout summary in the literature so the
        diagnostic plots here are directly comparable to published results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    per_class_correct: dict[str, int] = {c: 0 for c in class_names}
    per_class_total: dict[str, int] = {c: 0 for c in class_names}

    n_seen = 0
    for batch_inputs, batch_labels in dataloader:
        for i in range(batch_inputs.shape[0]):
            tensor = batch_inputs[i : i + 1]
            true_idx = int(batch_labels[i].item())
            if true_idx < 0 or true_idx >= len(class_names):
                _log.warning("Skipping sample with out-of-range label %d", true_idx)
                continue
            true_label = class_names[true_idx]

            pred = predict_with_uncertainty(
                model, tensor, class_names, n_samples=n_samples
            )
            predicted_label = pred["predicted_class"]
            is_correct = bool(predicted_label == true_label)

            per_class_total[true_label] += 1
            if is_correct:
                per_class_correct[true_label] += 1

            results.append({
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": is_correct,
                "confidence": pred["confidence"],
                "uncertainty": pred["uncertainty"],
            })
            n_seen += 1
            if n_seen % 50 == 0:
                _log.info("Processed %d images", n_seen)

    if not results:
        _log.warning("run_uncertainty_evaluation: dataloader produced no samples.")
        return results

    confidences = np.array([r["confidence"] for r in results], dtype=np.float64)
    uncertainties = np.array([r["uncertainty"] for r in results], dtype=np.float64)
    correct = np.array([1 if r["correct"] else 0 for r in results], dtype=np.int64)

    _save_calibration_curve(confidences, correct, output_dir / "calibration_curve.png")
    _save_uncertainty_histogram(uncertainties, correct, output_dir / "uncertainty_histogram.png")

    correct_mask = correct.astype(bool)
    mean_unc_correct = float(uncertainties[correct_mask].mean()) if correct_mask.any() else None
    mean_unc_incorrect = (
        float(uncertainties[~correct_mask].mean()) if (~correct_mask).any() else None
    )

    per_class_accuracy = {
        c: (per_class_correct[c] / per_class_total[c]) if per_class_total[c] > 0 else None
        for c in class_names
    }

    summary = {
        "n_samples_per_image": n_samples,
        "n_images": len(results),
        "overall_accuracy": float(correct.mean()),
        "mean_uncertainty_correct": mean_unc_correct,
        "mean_uncertainty_incorrect": mean_unc_incorrect,
        "per_class_accuracy": per_class_accuracy,
        "per_class_total": per_class_total,
    }
    with open(output_dir / "uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log.info(
        "Uncertainty eval done: n=%d acc=%.4f mean_H_correct=%s mean_H_wrong=%s",
        len(results), summary["overall_accuracy"],
        mean_unc_correct, mean_unc_incorrect,
    )
    return results
