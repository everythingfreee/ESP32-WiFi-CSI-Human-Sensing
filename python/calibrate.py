"""Guided calibration for the Wi-Fi CSI tracker.

The calibration walk records short sessions at fixed reference points and
derives the empty-room baseline used by the movement detector:

    1. empty room
    2. person at 0.00 (near ESP32-A)
    3. person at 0.25
    4. person at 0.50
    5. person at 0.75
    6. person at 1.00 (near ESP32-B)

Outputs:
  * ``data/processed/calibration.json`` — empty-room RSSI/activity baseline
    and the suggested movement threshold (auto-applied at inference when
    ``movement.auto_threshold_from_calibration`` is true);
  * ``data/processed/calibration_<timestamp>.csv`` — per-step summary;
  * raw CSI per step under ``data/raw/calib_<timestamp>_*.csv``.

Run this after every hardware move or room change.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from csi_parser import CsiSample
from config import load_config, resolve_path, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver

LOG = logging.getLogger("wifi_sensing.calibrate")


class CalibrationRun:
    """One guided calibration session."""

    def __init__(self, cfg: dict, positions: List[float], duration: float,
                 port: str, baud: int) -> None:
        self.cfg = cfg
        self.positions = positions
        self.duration = duration
        self.port = port
        self.baud = baud
        self.preproc_cfg = PreprocessConfig.from_config(cfg)
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.raw_dir = resolve_path(cfg, "raw_dir")
        self.processed_dir = resolve_path(cfg, "processed_dir")
        self.results: List[dict] = []

    # ------------------------------------------------------------------
    def _record_step(self, label: str, position: Optional[float],
                     prompt: str) -> dict:
        """Record one calibration step after the user confirms."""
        preprocessor = Preprocessor(self.preproc_cfg)
        extractor = FeatureExtractor(FeatureConfig.from_config(self.cfg),
                                     self.preproc_cfg)
        stats: dict = {"rssi": [], "activity": [], "n_samples": 0}

        def on_sample(sample: CsiSample) -> None:
            amp = preprocessor.process(sample)
            if amp is None:
                return
            stats["rssi"].append(float(sample.rssi))
            stats["n_samples"] += 1
            fv = extractor.update(amp, sample.rssi, sample.ts_local)
            if fv is None:
                return
            try:
                stats["activity"].append(
                    float(fv.values[fv.names.index("move_rel_activity")]))
            except ValueError:
                pass

        raw_path = self.raw_dir / (
            f"calib_{self.stamp}_{label}.csv")
        receiver = SerialReceiver(self.port, self.baud, callback=on_sample,
                                  record_path=raw_path)
        receiver.start()
        input(prompt)
        print(f"Recording ", end="", flush=True)
        width = 30
        start = time.time()
        last = -1
        while True:
            frac = min((time.time() - start) / self.duration, 1.0)
            filled = int(frac * width)
            if filled != last:
                print("#" * (filled - max(last, 0)), end="", flush=True)
                last = filled
            if frac >= 1.0:
                break
            time.sleep(0.15)
        receiver.stop()
        receiver.join(timeout=5.0)
        print(" 100%")

        rssi = np.asarray(stats["rssi"], dtype=np.float64)
        activity = np.asarray(stats["activity"], dtype=np.float64)
        result = {
            "label": label,
            "position": position,
            "raw_csv": str(raw_path),
            "n_samples": stats["n_samples"],
            "rssi_mean": float(rssi.mean()) if rssi.size else None,
            "rssi_std": float(rssi.std()) if rssi.size else None,
            "activity_mean": float(activity.mean()) if activity.size else None,
        }
        self.results.append(result)
        print(f"  -> samples={result['n_samples']}"
              + (f" rssi={result['rssi_mean']:.1f}±{result['rssi_std']:.1f} dBm"
                 if result["rssi_mean"] is not None else "")
              + (f" activity={result['activity_mean']:.4f}"
                 if result["activity_mean"] is not None else ""))
        return result


    # ------------------------------------------------------------------
    def run(self) -> int:
        """Execute all calibration steps and store the results."""
        print("=" * 64)
        print("Wi-Fi CSI tracker calibration")
        print("=" * 64)
        print(f"Steps: empty room, then person at {self.positions}")
        print(f"Each step records {self.duration:.0f} s. Keep the room as "
              "it will be during use.\n")

        self._record_step(
            "empty", None,
            "Leave the room empty (nobody between the ESP32 boards),\n"
            "then press Enter to start. ")
        for pos in self.positions:
            self._record_step(
                f"p{pos:.2f}", pos,
                f"Stand at position {pos:.2f} "
                f"({'near ESP32-A' if pos <= 0.01 else 'near ESP32-B' if pos >= 0.99 else f'{pos * 100:.0f}% of the link'}),\n"
                "then press Enter to start. ")

        self._save_summary()
        self._write_calibration_json()
        print("\nCalibration complete.")
        print(f"  summary         : {self.processed_dir / f'calibration_{self.stamp}.csv'}")
        print(f"  calibration.json: {self.processed_dir / 'calibration.json'}")
        return 0

    # ------------------------------------------------------------------
    def _save_summary(self) -> None:
        import csv
        path = self.processed_dir / f"calibration_{self.stamp}.csv"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "label", "position", "raw_csv", "n_samples", "rssi_mean",
                "rssi_std", "activity_mean"])
            writer.writeheader()
            writer.writerows(self.results)

    # ------------------------------------------------------------------
    def _write_calibration_json(self) -> None:
        empty = next((r for r in self.results if r["label"] == "empty"), None)
        payload = {
            "recorded_at": self.stamp,
            "positions": self.positions,
            "steps": self.results,
        }
        if empty is not None and empty["activity_mean"] is not None:
            baseline = empty["activity_mean"]
            payload["empty_activity"] = baseline
            payload["suggested_movement_threshold"] = baseline * 1.5
        if empty is not None and empty["rssi_mean"] is not None:
            payload["empty_rssi_mean"] = empty["rssi_mean"]
            payload["empty_rssi_std"] = empty["rssi_std"]
        (self.processed_dir / "calibration.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("calibration baseline written: empty_activity=%s",
                 payload.get("empty_activity"))


def main(argv=None) -> int:
    """CLI: ``python calibrate.py [--positions 0,0.25,0.5,0.75,1] ...``"""
    parser = argparse.ArgumentParser(
        description="Run the guided calibration procedure")
    parser.add_argument("--positions", default="0.0,0.25,0.5,0.75,1.0",
                        help="comma-separated normalized positions "
                             "(default: 0.0,0.25,0.5,0.75,1.0)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="seconds per step (default 15)")
    parser.add_argument("--port", default=None, help="serial port override")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--config", default=None, help="config.yaml path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    positions = [float(p) for p in args.positions.split(",") if p.strip()]
    for p in positions:
        if not 0.0 <= p <= 1.0:
            parser.error(f"position {p} outside [0.0, 1.0]")

    port = args.port or cfg.get("serial_port", "auto")
    baud = args.baud or int(cfg.get("baud_rate", 921600))
    run = CalibrationRun(cfg, positions, args.duration, port, baud)
    try:
        return run.run()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Connect the ESP32-B receiver over USB and try again (see --list).)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
