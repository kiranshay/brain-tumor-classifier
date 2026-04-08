"""NeuroScan MC-Dropout uncertainty evaluation CLI.

Usage:
    python training/evaluate_uncertainty.py --config backend/config.yaml

Loads the stage-1 classifier configured in the YAML, builds the test
dataloader from the same config, and runs `run_uncertainty_evaluation` over
it. Diagnostic plots and a summary JSON are written to the configured
`uncertainty.output_dir`. A short summary is also printed to stdout.

Why predictive entropy is the uncertainty score: see
backend/neuroscan/uncertainty.py — bounded, captures total predictive
uncertainty, and matches the standard MC-Dropout summary in the literature.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the backend/ package importable so we can use neuroscan.* from here.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from neuroscan.config import load_config, resolve_weights_path  # noqa: E402
from neuroscan.data import build_dataloaders  # noqa: E402
from neuroscan.logging_setup import setup_logging, get_logger  # noqa: E402
from neuroscan.models import load_classifier  # noqa: E402
from neuroscan.uncertainty import run_uncertainty_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MC-Dropout uncertainty evaluation on the stage-1 classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "backend" / "config.yaml"),
        help="Path to a YAML config file (default: backend/config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    log = get_logger("evaluate_uncertainty")

    s1 = config["model"]["stage1"]
    weights_path = resolve_weights_path(config, "stage1")
    log.info("Loading stage 1 classifier from %s", weights_path)
    model = load_classifier(
        weights_path=weights_path,
        num_classes=s1["num_classes"],
        dropout=s1["dropout"],
        class_names=s1["class_names"],
    )

    log.info("Building dataloaders from config...")
    _, test_loader, dataset_classes = build_dataloaders(config)

    if dataset_classes != list(s1["class_names"]):
        log.warning(
            "Test dataset class order %s does not match stage1.class_names %s. "
            "Using configured stage1 class_names for label mapping.",
            dataset_classes, list(s1["class_names"]),
        )

    unc_cfg = config.get("uncertainty", {})
    n_samples = int(unc_cfg.get("n_samples", 50))
    output_dir = unc_cfg.get("output_dir", "outputs/uncertainty")

    log.info("Running uncertainty eval: n_samples=%d output_dir=%s", n_samples, output_dir)
    results = run_uncertainty_evaluation(
        model=model,
        dataloader=test_loader,
        class_names=list(s1["class_names"]),
        n_samples=n_samples,
        output_dir=output_dir,
    )

    summary_path = Path(output_dir) / "uncertainty_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("=" * 60)
        print("MC-Dropout uncertainty evaluation summary")
        print("=" * 60)
        print(f"  images evaluated:           {summary['n_images']}")
        print(f"  MC samples per image:       {summary['n_samples_per_image']}")
        print(f"  overall accuracy:           {summary['overall_accuracy']:.4f}")
        print(f"  mean entropy (correct):     {summary['mean_uncertainty_correct']}")
        print(f"  mean entropy (incorrect):   {summary['mean_uncertainty_incorrect']}")
        print("  per-class accuracy:")
        for cls, acc in summary["per_class_accuracy"].items():
            print(f"    {cls:<20s} {acc}")
        print(f"  artifacts written to:       {output_dir}/")
    else:
        print(f"No summary written. results returned: {len(results)}")


if __name__ == "__main__":
    main()
