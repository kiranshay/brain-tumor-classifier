"""Config loader for NeuroScan.

The whole system reads from a single config.yaml. This module loads it into
a plain dict and exposes a couple of helpers for resolving the active stage
in the training config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Default config path: backend/config.yaml, sibling of this package.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load a YAML config file. Defaults to backend/config.yaml."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_stage_config(config: dict[str, Any], stage_key: str) -> dict[str, Any]:
    """Return the model.stageN block for the given key (e.g. 'stage1')."""
    try:
        return config["model"][stage_key]
    except KeyError as e:
        raise KeyError(f"Missing model config for {stage_key!r}") from e


def resolve_weights_path(config: dict[str, Any], stage_key: str) -> Path:
    """Resolve a stage's weights path relative to the config file's directory."""
    stage = resolve_stage_config(config, stage_key)
    weights = stage["weights_path"]
    # Weights are stored next to config.yaml in the backend dir.
    return DEFAULT_CONFIG_PATH.parent / weights
