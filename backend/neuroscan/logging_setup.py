"""Logging setup for NeuroScan.

Single entry point: setup_logging(config). Idempotent — safe to call from
both the FastAPI startup path and the training CLI.
"""
from __future__ import annotations

import logging
from typing import Any

_CONFIGURED = False


def setup_logging(config: dict[str, Any] | None = None) -> logging.Logger:
    """Configure root logging from the config dict's `logging` block.

    Returns the 'neuroscan' logger.
    """
    global _CONFIGURED

    log_cfg = (config or {}).get("logging", {}) if config else {}
    level_name = log_cfg.get("level", "INFO")
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    level = getattr(logging, level_name.upper(), logging.INFO)

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(level)

    return logging.getLogger("neuroscan")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"neuroscan.{name}")
