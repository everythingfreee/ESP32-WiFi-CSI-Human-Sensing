"""Feature extraction: from streaming CSI amplitudes to a fixed-size vector.

Pipeline (per config):

    CSI samples -> preprocessing -> sliding window -> Hampel outlier cleanup
        -> statistical + temporal + spectral features -> feature vector

All features are amplitude-based by default.  The extractor maintains a
ring buffer of the last ``window_size`` cleaned amplitude vectors and
emits a new feature vector every ``stride`` samples so the real-time
pipeline and the dataset collector produce identical features.

Feature groups (in output order):
  * per-subcarrier: mean, std, mean |first difference| over the window
    (captures the static + dynamic per-subcarrier behaviour a person
    causes on the link);
  * aggregate: mean, std, range, min, max across subcarriers of the
    window-averaged amplitude;
  * temporal: std / mean-abs-diff / energy / linear slope of the
    window-mean amplitude curve (movement sensitivity);
  * spectral: band energies of the FFT of the window-mean curve (slow
    drift vs. human-motion frequency bands);
  * RSSI: mean, std, last value (cheap complementary signal);
  * movement descriptors: relative activity and peak relative diff used
    by the heuristic movement detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from preprocessing import PreprocessConfig, hampel_filter



@dataclass
class FeatureConfig:
    """Feature extraction knobs (mirrors ``config.yaml: features``)."""

    window_size: int = 64
    stride: int = 8
    spectral_bands: int = 4
    include_per_subcarrier: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "FeatureConfig":
        feats = config.get("features", {})
        return cls(
            window_size=int(feats.get("window_size", 64)),
            stride=int(feats.get("stride", 8)),
            spectral_bands=int(feats.get("spectral_bands", 4)),
            include_per_subcarrier=bool(
                feats.get("include_per_subcarrier", True)),
        )


@dataclass
class FeatureVector:
    """A fixed-size feature vector plus its names and provenance."""

    values: np.ndarray
    names: List[str]
    timestamp: float          # local time of the newest sample in the window
    window_samples: int       # how many samples backed this vector

    @property
    def size(self) -> int:
        return len(self.values)


class FeatureExtractor:
    """Turns a stream of cleaned amplitude vectors into feature vectors."""

    def __init__(
        self,
        config: FeatureConfig,
        preprocess_config: Optional[PreprocessConfig] = None,
    ) -> None:
        from csi_parser import VALID_SUBCARRIERS_20MHZ  # avoid import cycle
        self.cfg = config
        n_sub = len(preprocess_config.valid_subcarriers) \
            if preprocess_config else len(VALID_SUBCARRIERS_20MHZ)
        self.n_subcarriers = n_sub
        self._amp_buf = np.zeros((0, n_sub), dtype=np.float64)
        self._rssi_buf: List[float] = []
        self._ts_buf: List[float] = []
        self._since_last = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop the ring buffers (after stream interruptions)."""
        self._amp_buf = np.zeros((0, self.n_subcarriers), dtype=np.float64)
        self._rssi_buf.clear()
        self._ts_buf.clear()
        self._since_last = 0

    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        """Deterministic feature name list (saved alongside models)."""
        names: List[str] = []
        if self.cfg.include_per_subcarrier:
            for i in range(self.n_subcarriers):
                names.append(f"sub_mean_{i}")
            for i in range(self.n_subcarriers):
                names.append(f"sub_std_{i}")
            for i in range(self.n_subcarriers):
                names.append(f"sub_diff_{i}")
        names += ["agg_mean", "agg_std", "agg_range", "agg_min", "agg_max"]
        names += ["temp_std", "temp_mean_abs_diff", "temp_energy", "temp_slope"]
        names += [f"spec_band_{i}" for i in range(self.cfg.spectral_bands)]
        names += ["rssi_mean", "rssi_std", "rssi_last"]
        names += ["move_rel_activity", "move_peak_rel_diff"]
        return names


    # ------------------------------------------------------------------
    def update(
        self,
        amp: np.ndarray,
        rssi: int,
        timestamp: float,
    ) -> Optional[FeatureVector]:
        """Feed one cleaned amplitude vector; return a FeatureVector every
        ``stride`` samples (or ``None`` while the window is filling)."""
        if len(amp) != self.n_subcarriers:
            raise ValueError(
                f"amplitude size {len(amp)} != {self.n_subcarriers} subcarriers")
        self._amp_buf = np.vstack([self._amp_buf, amp])
        self._rssi_buf.append(float(rssi))
        self._ts_buf.append(float(timestamp))
        if len(self._amp_buf) > self.cfg.window_size:
            self._amp_buf = self._amp_buf[-self.cfg.window_size:]
            self._rssi_buf = self._rssi_buf[-self.cfg.window_size:]
            self._ts_buf = self._ts_buf[-self.cfg.window_size:]
        self._since_last += 1
        if len(self._amp_buf) < self.cfg.window_size:
            return None
        if (self._since_last % self.cfg.stride) != 0:
            return None
        return self._compute(self._amp_buf.copy(), float(self._ts_buf[-1]))

    # ------------------------------------------------------------------
    def _compute(
        self,
        window: np.ndarray,
        timestamp: float,
    ) -> FeatureVector:
        """Compute all feature groups from a (window_size, n_sub) matrix."""
        window = hampel_filter(window, threshold=3.5, axis=0)
        rssi = np.asarray(self._rssi_buf[-len(window):], dtype=np.float64)
        values: List[float] = []
        names = self.feature_names()

        mean_curve = window.mean(axis=1)          # (window_size,)
        if self.cfg.include_per_subcarrier:
            values += window.mean(axis=0).tolist()
            values += window.std(axis=0).tolist()
            values += np.abs(np.diff(window, axis=0)).mean(axis=0).tolist()

        values += [float(mean_curve.mean()), float(mean_curve.std()),
                   float(mean_curve.max() - mean_curve.min()),
                   float(mean_curve.min()), float(mean_curve.max())]

        diff_curve = np.abs(np.diff(mean_curve)) \
            if len(mean_curve) > 1 else np.zeros(1)
        x = np.arange(len(mean_curve), dtype=np.float64)
        slope = float(np.polyfit(x, mean_curve, 1)[0])
        values += [float(mean_curve.std()), float(diff_curve.mean()),
                   float((mean_curve ** 2).mean()), slope]

        # Spectral band energies of the mean curve (movement bands).
        centered = mean_curve - mean_curve.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        if len(spectrum) > 1:
            spectrum[0] = 0.0  # ignore DC (already captured by mean)
        edges = np.linspace(0, len(spectrum), self.cfg.spectral_bands + 1,
                            dtype=int)
        for i in range(self.cfg.spectral_bands):
            lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
            values += [float((spectrum[lo:hi] ** 2).sum())]

        values += [float(rssi.mean()), float(rssi.std()), float(rssi[-1])]

        rel = diff_curve / (mean_curve[:-1].mean() + 1e-9)
        values += [float(rel.mean()), float(rel.max())]

        arr = np.nan_to_num(np.asarray(values, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) != len(names):
            raise RuntimeError(
                f"feature size mismatch: {len(arr)} values vs {len(names)} names")
        return FeatureVector(values=arr, names=list(names),
                             timestamp=timestamp,
                             window_samples=len(window))
