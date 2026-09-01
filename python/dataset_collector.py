"""Labeled dataset collection: stand at known positions and record.

Every session:
  * streams raw CSI from ESP32-B over serial (saved losslessly to
    ``data/raw/<session_id>.csv``);
  * runs the SAME preprocessing + feature extraction used at inference
    time;
  * appends one row per feature window to
    ``data/processed/dataset.csv`` with the label columns
    (``presence``, ``movement``, ``position``).

Usage examples::

    # empty room baseline
    python dataset_collector.py --label empty --duration 30

    # person standing at the midpoint (0.5)
    python dataset_collector.py --label person --position 0.5 \
        --mode static --duration 30

    # person walking along the link
    python dataset_collector.py --label person --position 0.5 \
        --mode moving --duration 30

Collect multiple sessions per position (repeat the same command); the
trainer splits by session, so several short sessions are better than one
long one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from csi_parser import CsiSample
from config import load_config, resolve_path, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver

LOG = logging.getLogger("wifi_sensing.collector")

DATASET_META = ("sample_id", "session_id", "timestamp")
DATASET_LABELS = ("presence", "movement", "position")


class SessionCollector:
    """Streams one labeled session into raw CSV + processed dataset."""

    def __init__(self, cfg: dict, session_id: str, label: str,
                 position: Optional[float], mode: str) -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.label = label
        self.position = position
        self.mode = mode
        if label == "person" and position is None:
            raise ValueError("--position is required when --label person")
        if label == "empty":
            self.position = None

        self.preproc_cfg = PreprocessConfig.from_config(cfg)
        self.preprocessor = Preprocessor(self.preproc_cfg)
        self.extractor = FeatureExtractor(
            FeatureConfig.from_config(cfg), self.preproc_cfg)
        self.feature_names = self.extractor.feature_names()

        processed_dir = resolve_path(cfg, "processed_dir")
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = resolve_path(cfg, "dataset_path")
        self.raw_path = resolve_path(cfg, "raw_dir") / f"{session_id}.csv"

        self.rows: list[list] = []
        self.next_sample_id = self._next_sample_id()

    # ------------------------------------------------------------------
    def _next_sample_id(self) -> int:
        if not self.dataset_path.exists():
            return 1
        try:
            existing = pd.read_csv(self.dataset_path, usecols=["sample_id"])
            return int(existing["sample_id"].max()) + 1
        except (ValueError, OSError):
            return int(time.time())

    # ------------------------------------------------------------------
    def feature_config_path(self) -> Path:
        return resolve_path(self.cfg, "processed_dir") / "feature_config.json"

    def ensure_feature_config(self) -> None:
        """Write/verify feature_config.json so training matches collection."""
        path = self.feature_config_path()
        current = {
            "feature_names": self.feature_names,
            "window_size": self.extractor.cfg.window_size,
            "stride": self.extractor.cfg.stride,
            "spectral_bands": self.extractor.cfg.spectral_bands,
            "include_per_subcarrier": self.extractor.cfg.include_per_subcarrier,
            "valid_subcarriers": list(self.preproc_cfg.valid_subcarriers),
            "preprocessing": {
                "smoothing": self.preproc_cfg.smoothing,
                "smoothing_alpha": self.preproc_cfg.smoothing_alpha,
                "ht_only": self.preproc_cfg.ht_only,
                "drop_rx_state_errors": self.preproc_cfg.drop_rx_state_errors,
                "drop_first_word_invalid":
                    self.preproc_cfg.drop_first_word_invalid,
            },
        }
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("feature_names") != current["feature_names"]:
                    LOG.warning(
                        "feature layout changed since the last collection "
                        "(old dataset has %d features, new has %d). "
                        "Consider re-collecting or deleting the old dataset.",
                        len(old.get("feature_names", [])),
                        len(current["feature_names"]))
            except json.JSONDecodeError:
                LOG.warning("unreadable feature_config.json; overwriting")
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")


    # ------------------------------------------------------------------
    def on_sample(self, sample: CsiSample) -> None:
        """Serial callback: preprocess -> features -> buffered row."""
        amp = self.preprocessor.process(sample)
        if amp is None:
            return
        fv = self.extractor.update(amp, sample.rssi, sample.ts_local)
        if fv is None:
            return
        try:
            rssi_mean = float(fv.values[fv.names.index("rssi_mean")])
            activity = float(fv.values[fv.names.index("move_rel_activity")])
        except ValueError:
            rssi_mean, activity = float(sample.rssi), 0.0
        self.rows.append(
            [self.next_sample_id + len(self.rows), self.session_id,
             fv.timestamp, rssi_mean]
            + fv.values.tolist()
            + [1 if self.label == "person" else 0,
               1 if (self.label == "person" and self.mode == "moving") else 0,
               self.position if self.label == "person" else np.nan])
        self._activity = activity

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Append buffered rows to the processed dataset CSV."""
        if not self.rows:
            LOG.warning("no feature rows collected - nothing to append")
            return
        header_needed = not self.dataset_path.exists()
        columns = (list(DATASET_META) + ["rssi"] + self.feature_names
                   + list(DATASET_LABELS))
        df = pd.DataFrame(self.rows, columns=columns)
        df.to_csv(self.dataset_path, mode="a",
                  header=header_needed, index=False)
        LOG.info("appended %d rows to %s", len(df), self.dataset_path)

    # ------------------------------------------------------------------
    def run(self, duration: float, port: str, baud: int) -> int:
        """Record for ``duration`` seconds with a progress display."""
        self.ensure_feature_config()
        receiver = SerialReceiver(port, baud, callback=self.on_sample,
                                  record_path=self.raw_path)
        receiver.start()
        print(f"\nSession '{self.session_id}': label={self.label}"
              + (f" position={self.position}" if self.position is not None
                 else "")
              + f" mode={self.mode} duration={duration:.0f}s")
        print("Recording ", end="", flush=True)
        start = time.time()
        width = 30
        last_shown = -1
        try:
            while True:
                elapsed = time.time() - start
                frac = min(elapsed / duration, 1.0)
                filled = int(frac * width)
                if filled != last_shown:
                    print("#" * (filled - max(last_shown, 0)), end="",
                          flush=True)
                    last_shown = filled
                if frac >= 1.0:
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\ninterrupted - keeping what was recorded so far")
        finally:
            receiver.stop()
            receiver.join(timeout=5.0)
        print(f" 100%")
        print(f"  raw samples : {receiver.packets} "
              f"(malformed {receiver.malformed}, io errors "
              f"{receiver.io_errors})")
        print(f"  raw CSV     : {self.raw_path}")
        print(f"  feature rows: {len(self.rows)}")
        self.flush()
        return 0 if self.rows else 1


def main(argv=None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect labeled CSI sessions for training")
    parser.add_argument("--label", required=True, choices=["empty", "person"],
                        help="what is happening in the room")
    parser.add_argument("--position", type=float, default=None,
                        help="normalized position 0.0..1.0 "
                             "(0.0=ESP32-A, 1.0=ESP32-B; required for person)")
    parser.add_argument("--mode", default="static", choices=["static", "moving"],
                        help="person standing still or walking")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="seconds to record (default 30)")
    parser.add_argument("--session-id", default=None,
                        help="session identifier (default: auto timestamp)")
    parser.add_argument("--port", default=None, help="serial port override")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--config", default=None, help="config.yaml path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.label == "person":
        if args.position is None:
            parser.error("--position is required when --label person "
                         "(e.g. --position 0.5)")
        if not 0.0 <= args.position <= 1.0:
            parser.error("--position must be within [0.0, 1.0]")

    session_id = args.session_id or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{args.label}"
        + (f"_p{args.position:.2f}" if args.position is not None else "")
        + f"_{args.mode}")
    collector = SessionCollector(cfg, session_id, args.label,
                                 args.position, args.mode)
    port = args.port or cfg.get("serial_port", "auto")
    baud = args.baud or int(cfg.get("baud_rate", 921600))
    try:
        return collector.run(args.duration, port, baud)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Connect the ESP32-B receiver over USB and try again (see --list.)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
