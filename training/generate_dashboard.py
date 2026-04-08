"""NeuroScan inspection dashboard generator.

Renders a single self-contained HTML report that lets a human visually
inspect the stage-1 classifier on a random sample of validation images.
For each image the dashboard shows the original MRI (after the same
border-crop preprocessing the FastAPI backend uses), the Grad-CAM
heatmap overlay, the predicted vs. true label, the confidence (mean of
the MC-Dropout softmax distribution), the predictive entropy (the
uncertainty score), and the full per-class probability distribution.

How to read this dashboard
--------------------------
Cards are sorted by predictive entropy in descending order so the
images the model is *least sure about* float to the top. That ordering
is intentional: those are the cases worth eyeballing.

What "uncertainty" means here
-----------------------------
The uncertainty score is the predictive entropy of the *mean*
softmax distribution from `n_samples` Monte Carlo Dropout forward
passes (see backend/neuroscan/uncertainty.py for the rationale).
It is bounded in [0, log(num_classes)] — for the 8-class stage-1
model that ceiling is ~2.08 nats.

Heuristics when interpreting a card:

- Low entropy + correct: the model is confidently right. Boring,
  but useful as a sanity check that the Grad-CAM lands on the lesion.
- Low entropy + incorrect: the model is confidently wrong. These are
  the most dangerous cases — calibration is failing here. Look at
  Grad-CAM: is the model fixating on a scanner artifact, the skull
  bezel, or text overlays rather than the lesion itself?
- High entropy + incorrect: the model knows it doesn't know. These
  are the right kind of mistakes — a downstream "decline-to-predict"
  rule thresholded on entropy would catch them.
- High entropy + correct: the model got it right despite hedging.
  Often these are images from minority classes or images that look
  like edge cases.

Pair this view with the calibration curve and entropy histogram
written by `training/evaluate_uncertainty.py` for the aggregate story.

Why this script exists
----------------------
The FastAPI backend gives one-off predictions; the uncertainty CLI
gives aggregate plots. Neither makes it easy to *look at the actual
images* the model is failing on. This dashboard fills that gap with
no Supabase, no FastAPI server, and no external CSS/JS dependencies —
the output is a single HTML file you can open locally or attach to
a writeup.

Usage:
    python training/generate_dashboard.py --config backend/config.yaml
    python training/generate_dashboard.py --config backend/config.yaml \
        --n-samples 100 --seed 7 --output outputs/dashboard/report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import random
import sys
from pathlib import Path

# Make the backend/ package importable so we can use neuroscan.* from here.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from PIL import Image  # noqa: E402
from torchvision import datasets  # noqa: E402

from neuroscan.config import load_config, resolve_weights_path  # noqa: E402
from neuroscan.data import build_inference_transform, crop_mri_borders  # noqa: E402
from neuroscan.gradcam import generate_gradcam  # noqa: E402
from neuroscan.logging_setup import get_logger, setup_logging  # noqa: E402
from neuroscan.models import load_classifier  # noqa: E402
from neuroscan.uncertainty import predict_with_uncertainty  # noqa: E402


def _pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _collect_sample(
    image_path: str,
    true_label: str,
    model,
    class_names: list[str],
    transform,
    border_threshold: int,
    border_pad: int,
    image_size: int,
    norm_mean,
    norm_std,
    n_mc_samples: int,
) -> dict:
    """Run preprocess + MC-Dropout + Grad-CAM for a single image path."""
    raw = Image.open(image_path).convert("RGB")
    cropped = crop_mri_borders(raw, threshold=border_threshold, pad=border_pad)

    display = cropped.resize((image_size, image_size), Image.BILINEAR)
    original_b64 = _pil_to_base64_png(display)

    tensor = transform(cropped).unsqueeze(0)

    pred = predict_with_uncertainty(
        model=model,
        tensor=tensor,
        class_names=class_names,
        n_samples=n_mc_samples,
    )

    gradcam_b64 = generate_gradcam(
        model=model,
        tensor=tensor,
        predicted_idx=pred["predicted_index"],
        image_size=image_size,
        mean=norm_mean,
        std=norm_std,
    )

    return {
        "original_b64": original_b64,
        "gradcam_b64": gradcam_b64,
        "true_label": true_label,
        "predicted_label": pred["predicted_class"],
        "correct": bool(pred["predicted_class"] == true_label),
        "confidence": float(pred["confidence"]),
        "uncertainty": float(pred["uncertainty"]),
        "mean_probs": list(pred["mean_probs"]),
    }


def _render_html(
    samples: list[dict],
    class_names: list[str],
    n_requested: int,
    seed: int,
    n_mc_samples: int,
) -> str:
    """Build the self-contained HTML document as a string."""
    total = len(samples)
    n_correct = sum(1 for s in samples if s["correct"])
    pct_correct = (100.0 * n_correct / total) if total else 0.0

    correct_unc = [s["uncertainty"] for s in samples if s["correct"]]
    wrong_unc = [s["uncertainty"] for s in samples if not s["correct"]]
    mean_unc_correct = sum(correct_unc) / len(correct_unc) if correct_unc else None
    mean_unc_wrong = sum(wrong_unc) / len(wrong_unc) if wrong_unc else None

    def _fmt_unc(v):
        return f"{v:.2f}" if v is not None else "n/a"

    summary_line = (
        f"{total} images sampled (requested {n_requested}, seed {seed}, "
        f"MC samples per image {n_mc_samples}) &middot; "
        f"{n_correct}/{total} correct ({pct_correct:.1f}%) &middot; "
        f"mean entropy correct {_fmt_unc(mean_unc_correct)} &middot; "
        f"mean entropy incorrect {_fmt_unc(mean_unc_wrong)}"
    )

    cards_html_parts: list[str] = []
    for s in samples:
        true_label_h = html.escape(s["true_label"])
        pred_label_h = html.escape(s["predicted_label"])
        correct_class = "correct" if s["correct"] else "incorrect"
        correct_attr = "true" if s["correct"] else "false"
        confidence_pct = s["confidence"] * 100.0
        uncertainty = s["uncertainty"]

        bar_rows: list[str] = []
        for cls_name, prob in zip(class_names, s["mean_probs"]):
            cls_h = html.escape(cls_name)
            pct = prob * 100.0
            highlight = " bar-row-pred" if cls_name == s["predicted_label"] else ""
            bar_rows.append(
                f'<div class="bar-row{highlight}">'
                f'<div class="bar-label">{cls_h}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>'
                f'<div class="bar-value">{pct:.1f}%</div>'
                f"</div>"
            )
        bars_html = "".join(bar_rows)

        card = (
            f'<div class="card" data-correct="{correct_attr}">'
            f'  <div class="card-images">'
            f'    <figure><figcaption>Original MRI</figcaption>'
            f'      <img alt="Original MRI" src="data:image/png;base64,{s["original_b64"]}"/></figure>'
            f'    <figure><figcaption>Grad-CAM</figcaption>'
            f'      <img alt="Grad-CAM overlay" src="data:image/png;base64,{s["gradcam_b64"]}"/></figure>'
            f'  </div>'
            f'  <div class="card-meta">'
            f'    <div class="labels">'
            f'      <div class="label-row"><span class="label-key">True</span>'
            f'        <span class="label-val">{true_label_h}</span></div>'
            f'      <div class="label-row {correct_class}"><span class="label-key">Predicted</span>'
            f'        <span class="label-val">{pred_label_h}</span></div>'
            f'    </div>'
            f'    <div class="scores">'
            f'      <div><span class="score-key">Confidence</span>'
            f'        <span class="score-val">{confidence_pct:.2f}%</span></div>'
            f'      <div><span class="score-key">Uncertainty (H)</span>'
            f'        <span class="score-val">{uncertainty:.2f}</span></div>'
            f'    </div>'
            f'    <div class="bars">{bars_html}</div>'
            f'  </div>'
            f'</div>'
        )
        cards_html_parts.append(card)

    cards_html = "\n".join(cards_html_parts)

    # Inline CSS — minimal, white background, sans-serif, subtle borders.
    css = """
    * { box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        background: #ffffff;
        color: #1a1a1a;
        margin: 0;
        padding: 24px;
        line-height: 1.4;
    }
    header { max-width: 1100px; margin: 0 auto 16px auto; }
    h1 { font-size: 20px; margin: 0 0 6px 0; font-weight: 600; }
    .summary { font-size: 13px; color: #444; margin-bottom: 12px; }
    .filters { display: flex; gap: 8px; margin-bottom: 18px; }
    .filters button {
        font: inherit;
        padding: 6px 14px;
        border: 1px solid #d0d0d0;
        background: #f8f8f8;
        color: #1a1a1a;
        border-radius: 6px;
        cursor: pointer;
    }
    .filters button.active {
        background: #1a1a1a;
        color: #ffffff;
        border-color: #1a1a1a;
    }
    main {
        max-width: 1100px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
    }
    .card {
        border: 1px solid #e3e3e3;
        border-radius: 8px;
        padding: 14px;
        display: grid;
        grid-template-columns: minmax(280px, 360px) 1fr;
        gap: 18px;
        background: #ffffff;
    }
    .card.hidden { display: none; }
    .card-images { display: flex; gap: 10px; }
    .card-images figure { margin: 0; flex: 1; text-align: center; }
    .card-images figcaption {
        font-size: 11px; color: #666; margin-bottom: 4px; text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .card-images img { width: 100%; height: auto; border: 1px solid #ececec; border-radius: 4px; display: block; }
    .card-meta { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
    .labels { display: flex; flex-direction: column; gap: 4px; }
    .label-row { font-size: 13px; }
    .label-key { display: inline-block; width: 80px; color: #666; }
    .label-val { font-weight: 600; }
    .label-row.correct .label-val { color: #137333; }
    .label-row.incorrect .label-val { color: #c5221f; }
    .scores { display: flex; gap: 24px; font-size: 13px; }
    .score-key { color: #666; margin-right: 6px; }
    .score-val { font-weight: 600; }
    .bars { display: flex; flex-direction: column; gap: 3px; margin-top: 4px; }
    .bar-row { display: grid; grid-template-columns: 110px 1fr 50px; align-items: center; gap: 8px; font-size: 11px; }
    .bar-label { color: #444; text-align: right; }
    .bar-track { background: #f0f0f0; height: 10px; border-radius: 2px; overflow: hidden; }
    .bar-fill { background: #6b7cff; height: 100%; }
    .bar-value { color: #444; font-variant-numeric: tabular-nums; }
    .bar-row-pred .bar-fill { background: #1a1a1a; }
    .bar-row-pred .bar-label { color: #1a1a1a; font-weight: 600; }
    """

    js = """
    (function() {
        var buttons = document.querySelectorAll('.filters button');
        var cards = document.querySelectorAll('.card');
        function applyFilter(mode) {
            for (var i = 0; i < cards.length; i++) {
                var c = cards[i];
                var isCorrect = c.getAttribute('data-correct') === 'true';
                var show = (mode === 'all') ||
                           (mode === 'correct' && isCorrect) ||
                           (mode === 'incorrect' && !isCorrect);
                if (show) { c.classList.remove('hidden'); }
                else { c.classList.add('hidden'); }
            }
        }
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].addEventListener('click', function(ev) {
                var mode = ev.currentTarget.getAttribute('data-filter');
                for (var j = 0; j < buttons.length; j++) { buttons[j].classList.remove('active'); }
                ev.currentTarget.classList.add('active');
                applyFilter(mode);
            });
        }
    })();
    """

    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\"/>\n"
        "<title>NeuroScan inspection dashboard</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "  <h1>NeuroScan inspection dashboard</h1>\n"
        f"  <div class=\"summary\">{summary_line}</div>\n"
        "  <div class=\"filters\">\n"
        "    <button data-filter=\"all\" class=\"active\">All</button>\n"
        "    <button data-filter=\"correct\">Correct Only</button>\n"
        "    <button data-filter=\"incorrect\">Incorrect Only</button>\n"
        "  </div>\n"
        "</header>\n"
        f"<main>\n{cards_html}\n</main>\n"
        f"<script>{js}</script>\n"
        "</body>\n"
        "</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML inspection dashboard for the stage-1 classifier."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to backend/config.yaml.")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of validation images to inspect (default: 50).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42).")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dashboard/inspection_report.html",
        help="Path to save the HTML file (default: outputs/dashboard/inspection_report.html).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    log = get_logger("generate_dashboard")

    s1 = config["model"]["stage1"]
    weights_path = resolve_weights_path(config, "stage1")
    log.info("Loading stage 1 classifier from %s", weights_path)
    model = load_classifier(
        weights_path=weights_path,
        num_classes=s1["num_classes"],
        dropout=s1["dropout"],
        class_names=list(s1["class_names"]),
    )

    image_size = int(config["model"]["image_size"])
    norm_mean = config["model"]["normalization"]["mean"]
    norm_std = config["model"]["normalization"]["std"]
    border_threshold = int(config["preprocessing"]["border_crop_threshold"])
    border_pad = int(config["preprocessing"]["border_crop_pad"])
    n_mc_samples = int(config.get("uncertainty", {}).get("n_samples", 50))

    test_dir = config["training"]["data"]["test_dir"]
    log.info("Indexing validation images from %s", test_dir)
    test_ds = datasets.ImageFolder(test_dir)
    dataset_classes = list(test_ds.classes)
    if dataset_classes != list(s1["class_names"]):
        log.warning(
            "Validation dataset class order %s does not match stage1.class_names %s. "
            "Using configured stage1 class_names for label mapping.",
            dataset_classes, list(s1["class_names"]),
        )
    class_names = list(s1["class_names"])

    all_samples = list(test_ds.samples)  # list of (path, class_index)
    if not all_samples:
        raise RuntimeError(f"No validation images found under {test_dir}")

    rng = random.Random(args.seed)
    n_take = min(args.n_samples, len(all_samples))
    chosen = rng.sample(all_samples, n_take)

    transform = build_inference_transform(config)

    log.info("Collecting %d samples (MC samples per image: %d)", n_take, n_mc_samples)
    collected: list[dict] = []
    for i, (path, class_idx) in enumerate(chosen, start=1):
        true_label = dataset_classes[class_idx]
        result = _collect_sample(
            image_path=path,
            true_label=true_label,
            model=model,
            class_names=class_names,
            transform=transform,
            border_threshold=border_threshold,
            border_pad=border_pad,
            image_size=image_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
            n_mc_samples=n_mc_samples,
        )
        collected.append(result)
        if i % 10 == 0:
            log.info("  processed %d/%d", i, n_take)

    collected.sort(key=lambda r: r["uncertainty"], reverse=True)

    html_doc = _render_html(
        samples=collected,
        class_names=class_names,
        n_requested=args.n_samples,
        seed=args.seed,
        n_mc_samples=n_mc_samples,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")

    n_correct = sum(1 for r in collected if r["correct"])
    log.info(
        "Wrote dashboard with %d cards (%d correct) to %s",
        len(collected), n_correct, output_path,
    )
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
