"""NeuroScan training CLI.

Usage:
    python training/train.py --config backend/config.yaml

The script accepts a config file path so experiments are reproducible: every
hyperparameter, dataset path, augmentation setting, class list, and target
weights path lives in the YAML file. To train a different stage, either edit
`training.stage` in the config (e.g. `stage1` or `stage2`) or pass a separate
YAML file via --config.

The legacy notebooks (training/train_8class_v5.ipynb,
training/train_glioma_subtype.ipynb) remain in the repo for reference; this
script is the canonical, version-controlled entry point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the backend/ package importable so we can use neuroscan.* from here.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from neuroscan.config import load_config  # noqa: E402
from neuroscan.logging_setup import setup_logging  # noqa: E402
from neuroscan.training import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NeuroScan classifier from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "backend" / "config.yaml"),
        help="Path to a YAML config file (default: backend/config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    log = setup_logging(config)
    log.info("Loaded config from %s", args.config)
    log.info("Training stage: %s", config["training"]["stage"])

    weights_path = train(config)
    log.info("Done. Weights saved to: %s", weights_path)


if __name__ == "__main__":
    main()
