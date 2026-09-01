"""Real-time inference: model loading + streaming detection pipeline.

* :func:`load_models` loads the artifacts written by ``train.py`` (model
  manifest + scaler + feature configuration) once at startup. Training
  never runs at inference time.
* :class:`MovementDetector` — heuristic STATIC/MOVING from temporal CSI
  activity, exponentially smoothed.
* :class:`InferenceEngine` — full pipeline:
  CSI sample -> preprocessing -> windowed features -> presence
  classifier -> position model -> position filter -> :class:`SystemState`
  (a locked snapshot the GUI can poll safely).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from csi_parser import CsiSample

LOG = logging.getLogger("wifi_sensing.inference")


@dataclass
class SystemState:
    """Thread-safe view of the current tracking state (GUI-facing)."""

    connection: str = "OFFLINE"          # ONLINE / OFFLINE
    presence_label: str = "UNKNOWN"      # PERSON / EMPTY / UNKNOWN
    presence_probability: Optional[float] = None
    movement_label: str = "UNKNOWN"      # MOVING / STATIC / UNKNOWN
    movement_probability: Optional[float] = None
    position_raw: Optional[float] = None
    position_filtered: Optional[float] = None
    confidence: Optional[float] = None
    csi_rate: float = 0.0
    latency_ms: float = 0.0
    packets: int = 0
    malformed: int = 0
    model_name: str = "none (heuristic)"
    last_update: float = 0.0


@dataclass
class LoadedModels:
    """Everything inference needs from the training stage."""

    presence: object                 # .predict(features) -> P(person)
    position: Optional[object]       # .predict(features) -> [0, 1] or None
    scaler: Optional[StandardScaler] # shared feature scaler (fit on train)
    feature_names: list
    manifest: dict

    @property
    def position_available(self) -> bool:
        return self.position is not None


def load_models(models_dir: Path, cfg: dict) -> Optional[LoadedModels]:
    """Load the best models from ``models_dir``; ``None`` when absent."""
    manifest_path = models_dir / "model_manifest.json"
    if not manifest_path.exists():
        LOG.warning("no model manifest at %s - run train.py first",
                    manifest_path)
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = manifest.get("feature_names") or []
    scaler = None
    scaler_file = manifest.get("scaler")
    if scaler_file:
        scaler_path = models_dir / scaler_file
        if scaler_path.exists():
            import joblib
            scaler = joblib.load(scaler_path)

    from models.baseline_model import BaselinePositionModel, \
        BaselinePresenceModel

    def _load_presence():
        entry = manifest.get("presence")
        if not entry:
            return None
        path = models_dir / entry["file"]
        if entry.get("type") == "torch":
            from models import neural_network as nn_mod
            return nn_mod.TorchPresenceModel(
                nn_mod.load_torch_model("presence", len(feature_names),
                                        str(path), cfg),
                name=entry.get("name", "presence"))
        from models import baseline_model as bl_mod
        return BaselinePresenceModel(
            bl_mod.load_sklearn_model(str(path)),
            name=entry.get("name", "presence"))

    def _load_position():
        entry = manifest.get("position")
        if not entry:
            return None
        path = models_dir / entry["file"]
        if entry.get("type") == "torch":
            from models import neural_network as nn_mod
            return nn_mod.TorchPositionModel(
                nn_mod.load_torch_model("position", len(feature_names),
                                        str(path), cfg),
                name=entry.get("name", "position"))
        from models import baseline_model as bl_mod
        return BaselinePositionModel(
            bl_mod.load_sklearn_model(str(path)),
            name=entry.get("name", "position"))

    presence = _load_presence()
    position = _load_position()
    if presence is None:
        LOG.warning("manifest contains no usable presence model")
        return None
    LOG.info("models loaded: presence=%s%s",
             getattr(presence, "name", "?"),
             f", position={getattr(position, 'name', '?')}"
             if position is not None else " (no position model)")
    return LoadedModels(presence=presence, position=position, scaler=scaler,
                        feature_names=feature_names, manifest=manifest)


def apply_scaler(scaler: Optional[StandardScaler], features: np.ndarray) -> np.ndarray:
    """Scale features for models that expect scaled input (torch models).

    sklearn candidates embed their own scaler in the pipeline; applying the
    shared scaler again would be wrong there, so callers only use this for
    torch models (``kind == "torch"``).
    """
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if scaler is None:
        return x
    return scaler.transform(x)


class MovementDetector:
    """Heuristic STATIC/MOVING detector from temporal CSI activity.

    Uses the ``move_rel_activity`` feature (mean relative amplitude change
    over the window): a person walking changes the channel much faster
    than a static room.  One noisy sample never flips the state because
    the probability is exponentially smoothed over consecutive windows.
    The threshold comes from config, or from ``calibrate.py`` (empty-room
    baseline + margin) when auto-calibration is enabled.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.alpha = float(cfg.get("movement", {}).get("smoothing_alpha", 0.5))
        self.threshold = float(cfg.get("movement_threshold", 0.12))
        self.auto_cal = bool(cfg.get("movement", {}).get(
            "auto_threshold_from_calibration", True))
        self._prob: Optional[float] = None

    def update_threshold_from_calibration(self, cal_file) -> None:
        """Adopt the empty-room baseline measured by calibrate.py, if any."""
        try:
            cal = json.loads(Path(cal_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        baseline = float(cal.get("empty_activity", 0.0))
        if baseline > 0:
            self.threshold = baseline * 1.5
            LOG.info("movement threshold %.4f from calibration "
                     "(empty baseline %.4f)", self.threshold, baseline)

    def update(self, features: np.ndarray, names: list) -> float:
        """Feed one feature vector; return smoothed P(movement) in [0, 1]."""
        try:
            idx = names.index("move_rel_activity")
            activity = float(features[idx])
        except (ValueError, IndexError):
            activity = 0.0
        scale = max(self.threshold, 1e-6)
        raw = 1.0 / (1.0 + float(np.exp(-(activity - self.threshold)
                                        / (0.5 * scale))))
        self._prob = raw if self._prob is None \
            else self.alpha * raw + (1.0 - self.alpha) * self._prob
        return self._prob


class InferenceEngine:
    """Full real-time pipeline: CSI sample -> presence/movement/position.

    Thread model: :meth:`process_sample` is called from the serial
    receiver thread; the GUI polls the locked :attr:`state` snapshot and
    :meth:`snapshot_for_plot` from its own timer.
    """

    def __init__(self, cfg: dict, models: Optional[LoadedModels]) -> None:
        from feature_extraction import FeatureConfig, FeatureExtractor
        from position_filter import PositionFilter, PositionFilterConfig
        from preprocessing import PreprocessConfig, Preprocessor

        self.cfg = cfg
        self.models = models
        self.preproc_cfg = PreprocessConfig.from_config(cfg)
        self.preprocessor = Preprocessor(self.preproc_cfg)
        self.extractor = FeatureExtractor(
            FeatureConfig.from_config(cfg), self.preproc_cfg)
        self.feature_names = self.extractor.feature_names()
        self.position_filter = PositionFilter(
            PositionFilterConfig.from_config(cfg))
        self.presence_alpha = float(cfg.get("presence", {}).get(
            "smoothing_alpha", 0.6))
        self.presence_threshold = float(cfg.get("presence_threshold", 0.5))
        self.movement = MovementDetector(cfg)
        if self.movement.auto_cal:
            cal_file = cfg.get("movement", {}).get("calibration_file")
            if cal_file:
                self.movement.update_threshold_from_calibration(cal_file)
        if models is not None:
            self.model_name = getattr(models.presence, "name", "model")
        else:
            self.model_name = "none (heuristic fallback)"

        self.latency_ms = 0.0
        self._presence_prob: Optional[float] = None
        self._lock = threading.Lock()
        self._amplitude_history: Deque[np.ndarray] = deque(maxlen=300)
        self._rssi_history: Deque[float] = deque(maxlen=300)
        self._position_history: Deque[Optional[float]] = deque(maxlen=200)
        self.state = SystemState(model_name=self.model_name)

    # ------------------------------------------------------------------
    def process_sample(self, sample: CsiSample) -> None:
        """Consume one raw CSI sample (called from the serial thread)."""
        t0 = time.perf_counter()
        amp = self.preprocessor.process(sample)
        if amp is None:
            return
        self._amplitude_history.append(amp.copy())
        self._rssi_history.append(float(sample.rssi))
        fv = self.extractor.update(amp, sample.rssi, sample.ts_local)
        if fv is None:
            return

        movement_prob = self.movement.update(fv.values, fv.names)

        if self.models is not None:
            x = fv.values.reshape(1, -1)
            if getattr(self.models.presence, "kind", "") == "torch":
                x = apply_scaler(self.models.scaler, fv.values)
            presence_prob = float(self.models.presence.predict(x.reshape(-1)))
        else:
            # Honest fallback when no model is trained yet: motion implies
            # a person. Clearly reported as "heuristic" in the UI.
            presence_prob = min(max(movement_prob, 0.0), 1.0)
        if self._presence_prob is None:
            self._presence_prob = presence_prob
        else:
            a = self.presence_alpha
            self._presence_prob = (a * presence_prob
                                   + (1.0 - a) * self._presence_prob)

        person = self._presence_prob >= self.presence_threshold
        position_raw: Optional[float] = None
        position_filtered: Optional[float] = None
        if person and self.models is not None \
                and self.models.position_available:
            x = fv.values.reshape(1, -1)
            if getattr(self.models.position, "kind", "") == "torch":
                x = apply_scaler(self.models.scaler, fv.values)
            position_raw = float(self.models.position.predict(x.reshape(-1)))
            position_filtered = self.position_filter.filter(
                position_raw, fv.timestamp)
        if not person:
            self.position_filter.reset()

        latency = (time.perf_counter() - t0) * 1000.0
        self.latency_ms = (0.9 * self.latency_ms + 0.1 * latency
                           if self.latency_ms else latency)

        with self._lock:
            self._position_history.append(position_filtered)
            self.state.last_update = time.time()
            self.state.presence_label = "PERSON" if person else "EMPTY"
            self.state.presence_probability = self._presence_prob
            self.state.movement_label = \
                "MOVING" if movement_prob >= 0.5 else "STATIC"
            self.state.movement_probability = movement_prob
            self.state.position_raw = position_raw
            self.state.position_filtered = position_filtered
            self.state.confidence = self._presence_prob if person else None
            self.state.latency_ms = self.latency_ms

    # ------------------------------------------------------------------
    def update_link_status(self, online: bool, rate: float, packets: int,
                           malformed: int) -> None:
        """Refresh link-level stats (from the serial receiver thread)."""
        with self._lock:
            self.state.connection = "ONLINE" if online else "OFFLINE"
            self.state.csi_rate = rate
            self.state.packets = packets
            self.state.malformed = malformed

    # ------------------------------------------------------------------
    def snapshot_for_plot(self) -> dict:
        """Copy of debug-plot histories (GUI thread)."""
        with self._lock:
            return {
                "amplitudes": list(self._amplitude_history),
                "rssi": list(self._rssi_history),
                "positions": list(self._position_history),
            }



def main(argv=None) -> int:
    """CLI: headless real-time inference (no GUI)."""
    import argparse

    from config import load_config, resolve_path, setup_logging
    from serial_receiver import SerialReceiver

    parser = argparse.ArgumentParser(
        description="Run real-time presence/movement/position inference "
                    "from the ESP32 CSI stream (headless)")
    parser.add_argument("--port", default=None,
                        help="serial port (default: config value)")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--models-dir", default=None,
                        help="directory with trained models "
                             "(default: config value)")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["log_level"] = "DEBUG" if args.debug else cfg.get("log_level", "INFO")
    setup_logging(cfg)

    models_dir = Path(args.models_dir) if args.models_dir \
        else resolve_path(cfg, "model_path")
    models = load_models(models_dir, cfg)
    if models is None:
        print("WARNING: no trained models found - presence/movement run in "
              "heuristic mode and position is unavailable.\n"
              "Train models with: python dataset_collector.py && python "
              "train.py\n")

    engine = InferenceEngine(cfg, models)

    def on_sample(sample: CsiSample) -> None:
        engine.process_sample(sample)

    port = args.port or cfg.get("serial_port", "auto")
    baud = args.baud or int(cfg.get("baud_rate", 921600))
    receiver = SerialReceiver(port, baud, callback=on_sample)
    receiver.start()

    print("Real-time CSI inference (Ctrl+C to stop)")
    print("-" * 64)
    try:
        while True:
            time.sleep(0.5)
            engine.update_link_status(
                receiver.is_online, receiver.packet_rate,
                receiver.packets, receiver.malformed)
            s = engine.state
            pos = (f"{s.position_filtered:.2f}"
                   if s.position_filtered is not None else "  --")
            raw = (f"{s.position_raw:.2f}"
                   if s.position_raw is not None else "--")
            conf = (f"{s.confidence * 100:5.1f}%"
                    if s.confidence is not None else "  -- ")
            print(f"\r[{s.connection:7s}] {s.presence_label:6s} "
                  f"P={s.presence_probability if s.presence_probability is not None else 0:.2f} "
                  f"{s.movement_label:6s} pos={pos} (raw {raw}) "
                  f"conf={conf} csi={s.csi_rate:5.1f}/s "
                  f"lat={s.latency_ms:4.1f}ms ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        receiver.stop()
        receiver.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
