"""GUI smoke test — runs the full Qt interface WITHOUT hardware.

Feeds simulated CSI samples through the real inference engine, builds the
main window, drives the refresh timer manually and opens the debug window.
Offscreen by default (``QT_QPA_PLATFORM=offscreen`` is set automatically if
not already present).

    python tests/gui_smoke_test.py

Prints ``GUI SMOKE TEST OK`` on success. The sample data is synthetic and
only exercises code paths; it is never interpreted as real measurements.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np  # noqa: E402


def main() -> int:
    import argparse

    from config import load_config, resolve_path
    from inference import InferenceEngine, load_models
    from csi_parser import parse_line

    parser = argparse.ArgumentParser(description="GUI smoke test")
    parser.add_argument("--with-models", action="store_true",
                        help="also load data/models and feed a simulated person")
    args = parser.parse_args()

    cfg = load_config()
    models = None
    if args.with_models:
        models = load_models(resolve_path(cfg, "model_path"), cfg)
        if models is None:
            print("--with-models requested but no models found; run train.py first")
            return 1
    engine = InferenceEngine(cfg, models)

    rng = np.random.default_rng(1)

    class FakeReceiver:
        is_online = True
        packets = 100
        malformed = 2

        @property
        def packet_rate(self) -> float:
            return 55.0

        def stop(self) -> None:
            pass

        def join(self, timeout: float = 0.0) -> None:  # noqa: U100
            pass

    for k in range(400):
        i = np.arange(54) / 54
        amps = 40 + 6 * np.sin(2 * np.pi * (i + 0.3)) \
            + rng.normal(0, 0.5, 54)
        from csi_parser import VALID_SUBCARRIERS_20MHZ
        full = np.zeros(64, dtype=complex)
        full[list(VALID_SUBCARRIERS_20MHZ)] = amps * np.exp(1j * 0.3)
        raw = np.zeros(128, dtype=int)
        raw[0::2] = np.round(full.real).astype(int)
        raw[1::2] = np.round(full.imag).astype(int)
        obj = {
            "timestamp": k * 16000, "seq": k, "rssi": -50, "channel": 6,
            "secondary_channel": 0, "sig_mode": 1, "mcs": 5, "cwb": 0,
            "rate": 0, "aggregation": 0, "stbc": 0, "fec_coding": 0,
            "sgi": 0, "noise_floor": -96, "ampdu_cnt": 1, "sig_len": 64,
            "rx_state": 0, "ant": 0, "timestamp_wifi": k * 16000,
            "mac": "AABBCCDDEEFF", "first_word_invalid": 0,
            "csi_len": 128, "csi": raw.tolist(),
        }
        engine.process_sample(parse_line(json.dumps(obj), time.time() + k / 60))

    from PyQt6.QtWidgets import QApplication

    from visualization import MainWindow

    app = QApplication([])
    window = MainWindow(engine, FakeReceiver(), cfg)
    window.show()
    for _ in range(5):
        window._refresh()
        app.processEvents()
    window.debug_check.setChecked(True)
    app.processEvents()
    for _ in range(3):
        window._refresh()
        app.processEvents()
    print("GUI SMOKE TEST OK - presence:", engine.state.presence_label,
          "movement:", engine.state.movement_label)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
