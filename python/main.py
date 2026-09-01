"""Application entry point: live GUI for the Wi-Fi CSI human tracker.

Wires together serial reception, the inference engine and the PyQt6 UI:

    python main.py [--port ...] [--baud ...] [--models-dir ...] [--config ...]

Serial reading runs on a background thread; the GUI only polls snapshots
on a timer and therefore never blocks on I/O or inference.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("wifi_sensing.main")


def main(argv=None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Wi-Fi CSI human tracker (live GUI)")
    parser.add_argument("--port", default=None,
                        help="serial port (default: config value)")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--models-dir", default=None,
                        help="directory with trained models")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--debug", action="store_true", help="DEBUG logging")
    args = parser.parse_args(argv)

    from config import load_config, resolve_path, setup_logging
    cfg = load_config(args.config)
    cfg["log_level"] = "DEBUG" if args.debug else cfg.get("log_level", "INFO")
    setup_logging(cfg)

    from inference import InferenceEngine, load_models
    from serial_receiver import SerialReceiver

    models_dir = Path(args.models_dir) if args.models_dir \
        else resolve_path(cfg, "model_path")
    models = load_models(models_dir, cfg)
    if models is None:
        LOG.warning("no trained models found - heuristic mode, no position")

    engine = InferenceEngine(cfg, models)

    def on_sample(sample) -> None:
        engine.process_sample(sample)

    port = args.port or cfg.get("serial_port", "auto")
    baud = args.baud or int(cfg.get("baud_rate", 921600))
    try:
        receiver = SerialReceiver(port, baud, callback=on_sample)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    receiver.start()

    from PyQt6.QtWidgets import QApplication

    from visualization import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(engine, receiver, cfg)
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        return 0
    finally:
        receiver.stop()


if __name__ == "__main__":
    raise SystemExit(main())
