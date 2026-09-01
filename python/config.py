"""Configuration loading and logging setup for the Wi-Fi CSI sensing project.

All runtime-tunable values live in ``config.yaml`` at the project root; this
module only defines defaults, merges user overrides, and resolves paths
relative to the project root (never absolute, never hardcoded elsewhere).
"""

from __future__ import annotations

import copy
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

#: Project root = the folder that contains config.yaml / python/ / data/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

#: Documented defaults. Keys mirror config.yaml; every value can be
#: overridden there or via CLI ``--set key=value`` arguments.
DEFAULTS: Dict[str, Any] = {
    # --- serial link to ESP32-B (spec section 26 keys) ---
    "serial_port": "auto",        # "auto" -> scan, or e.g. /dev/cu.usbmodem1101
    "baud_rate": 921600,          # must match SERIAL_BAUD in the receiver firmware
    "sample_rate": 60,            # nominal CSI rate in Hz (matches TRAFFIC_RATE_HZ)
    # --- room geometry ---
    "room_width": 5.0,            # metres between ESP32-A and ESP32-B
    "room_length": 4.0,
    # --- tracking ---
    "smoothing_factor": 0.35,     # position EMA alpha (higher = less smoothing)
    "presence_threshold": 0.5,
    "movement_threshold": 0.12,   # activity score threshold (calibration may override)
    # --- storage ---
    "model_path": "data/models",
    "dataset_path": "data/processed/dataset.csv",
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "log_dir": "logs",
    "log_level": "INFO",
    # --- CSI preprocessing (each step documented in preprocessing.py) ---
    "csi": {
        "expect_length_20mhz": 128,   # 64 subcarriers * 2 bytes (IDF CSI guide)
        "valid_subcarriers_20mhz": list(range(0, 27)) + list(range(32, 59)),
        "ht_only": False,             # keep both HT data frames and legacy beacons
        "drop_rx_state_errors": True,
        "drop_first_word_invalid": True,
        "phase_features": False,      # ESP32 phase is noisy; off by default
        "outlier_mad_threshold": 3.5,
        "smoothing": "ema",           # "ema" | "moving_avg" | "none"
        "smoothing_alpha": 0.25,
        "smoothing_window": 5,
        "lowpass_enabled": False,     # optional Butterworth along time
        "lowpass_cutoff_hz": 10.0,
        "lowpass_order": 4,
        "max_gap_seconds": 0.5,       # larger gaps reset smoothing state
    },
    # --- feature extraction ---
    "features": {
        "window_size": 64,            # samples per feature window (~1 s @60 Hz)
        "stride": 8,                  # new feature vector every N samples
        "spectral_bands": 4,
        "include_per_subcarrier": True,
    },
    # --- detection smoothing ---
    "presence": {"smoothing_alpha": 0.6},
    "movement": {
        "smoothing_alpha": 0.5,
        "auto_threshold_from_calibration": True,
        "calibration_file": "data/processed/calibration.json",
    },
    # --- GUI ---
    "gui": {
        "update_hz": 30,
        "debug_window": True,
        "debug_plot_samples": 300,
    },
    # --- training ---
    "train": {
        "seed": 42,
        "test_fraction": 0.2,
        "val_fraction": 0.2,
        "knn_neighbors": 7,
        "rf_estimators": 200,
        "nn_epochs": 60,
        "nn_batch_size": 64,
        "nn_learning_rate": 0.001,
        "nn_dropout": 0.2,
        "nn_hidden": [64, 32],
        "nn_patience": 8,
    },
}



def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load config.yaml, falling back to :data:`DEFAULTS`.

    Args:
        path: optional path to a YAML config file.
        overrides: highest-priority dict (e.g. from CLI arguments).

    Raises:
        ValueError: if the YAML file cannot be parsed.
        FileNotFoundError: if an explicitly given config file is missing.
    """
    cfg = copy.deepcopy(DEFAULTS)
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                user_cfg = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {cfg_path}: {exc}") from exc
        cfg = deep_merge(cfg, user_cfg)
    elif path:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    """Resolve a path-like config value relative to the project root."""
    value = config.get(key)
    if value is None:
        raise KeyError(f"Missing config key: {key}")
    p = Path(str(value))
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def get_nested(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``config["a"]["b"]`` via the dotted string ``"a.b"``."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def setup_logging(config: Dict[str, Any], name: str = "wifi_sensing") -> logging.Logger:
    """Configure console + rotating file logging under ``log_dir``."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured in this process
        return logger
    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                          "%H:%M:%S"))
    logger.addHandler(console)

    log_dir = resolve_path(config, "log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(file_handler)
    return logger
