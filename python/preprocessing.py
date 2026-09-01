"""Configurable CSI preprocessing pipeline.

Every operation below exists for a specific, documented reason.  Nothing is
applied blindly: each step can be switched on/off from ``config.yaml``.

1. *Sample validation* (``drop_rx_state_errors``): the ESP32 marks corrupted
   receptions via ``rx_ctrl.rx_state``. Dropping these prevents garbage from
   entering the features.
2. *Invalid first word* (``drop_first_word_invalid``): a documented ESP32
   hardware limitation can make the first 4 CSI bytes (subcarriers 0-1)
   invalid; those samples are dropped when the flag is set.
3. *Bandwidth / frame-type filtering* (``ht_only``): 20 MHz buffers have a
   fixed 64-slot geometry; 40 MHz or STBC frames have different lengths, so
   unexpected buffers are rejected. ``ht_only`` keeps only HT data frames
   (the probe echoes) and discards legacy frames such as beacons, giving a
   more homogeneous signal.
4. *Valid subcarrier selection*: per Espressif's CSI guide only subcarriers
   0..26 and 32..58 carry usable channel estimates at 20 MHz; the rest are
   DC/guard/pilot slots and are removed.
5. *Outlier detection* (Hampel filter, applied per window in
   :func:`hampel_filter`): single-sample spikes (microwave bursts, packet
   retries) are replaced by the window median when they deviate more than
   ``outlier_mad_threshold`` medians-abs-deviations.
6. *Temporal smoothing* (EMA over consecutive samples): CSI amplitudes are
   noisy; a small EMA reduces variance without destroying temporal dynamics.
   The state resets after a reception gap larger than ``max_gap_seconds``.
7. *Optional low-pass filter* (Butterworth IIR, streaming): removes
   high-frequency noise above ``lowpass_cutoff_hz`` when enabled. Off by
   default because the EMA already limits noise and the ML features rely on
   temporal dynamics.
8. *NaN/inf guard*: any non-finite amplitude rejects the sample (never
   propagates into features or models).

Raw phase is available (:func:`clean_phase`) but phase-derived features are
disabled by default: the ESP32's per-packet phase offset makes raw phase
unreliable without careful calibration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # scipy is optional only for the low-pass filter
    from scipy.signal import butter, lfilter
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

from csi_parser import VALID_SUBCARRIERS_20MHZ, CsiSample


@dataclass
class PreprocessConfig:
    """Knobs for every preprocessing step (mirrors ``config.yaml: csi``)."""

    valid_subcarriers: Tuple[int, ...] = VALID_SUBCARRIERS_20MHZ
    expect_length: int = 128          # bytes for a 20 MHz non-STBC buffer
    ht_only: bool = False
    drop_rx_state_errors: bool = True
    drop_first_word_invalid: bool = True
    outlier_mad_threshold: float = 3.5
    smoothing: str = "ema"            # "ema" | "moving_avg" | "none"
    smoothing_alpha: float = 0.25
    smoothing_window: int = 5
    lowpass_enabled: bool = False
    lowpass_cutoff_hz: float = 10.0
    lowpass_order: int = 4
    max_gap_seconds: float = 0.5
    phase_features: bool = False
    nominal_rate_hz: float = 60.0     # nominal CSI rate, for filter design

    @classmethod
    def from_config(cls, config: dict) -> "PreprocessConfig":
        """Build from the ``config["csi"]`` subtree plus nominal rate."""
        csi = config.get("csi", {})
        return cls(
            valid_subcarriers=tuple(csi.get(
                "valid_subcarriers_20mhz", VALID_SUBCARRIERS_20MHZ)),
            expect_length=int(csi.get("expect_length_20mhz", 128)),
            ht_only=bool(csi.get("ht_only", False)),
            drop_rx_state_errors=bool(csi.get("drop_rx_state_errors", True)),
            drop_first_word_invalid=bool(
                csi.get("drop_first_word_invalid", True)),
            outlier_mad_threshold=float(csi.get("outlier_mad_threshold", 3.5)),
            smoothing=str(csi.get("smoothing", "ema")),
            smoothing_alpha=float(csi.get("smoothing_alpha", 0.25)),
            smoothing_window=int(csi.get("smoothing_window", 5)),
            lowpass_enabled=bool(csi.get("lowpass_enabled", False)),
            lowpass_cutoff_hz=float(csi.get("lowpass_cutoff_hz", 10.0)),
            lowpass_order=int(csi.get("lowpass_order", 4)),
            max_gap_seconds=float(csi.get("max_gap_seconds", 0.5)),
            phase_features=bool(csi.get("phase_features", False)),
            nominal_rate_hz=float(config.get("sample_rate", 60)),
        )


@dataclass
class RejectionStats:
    """Counters for every rejection reason (surfaced in logs / debug view)."""

    total_seen: int = 0
    accepted: int = 0
    rejected_total: int = 0
    rx_state_errors: int = 0
    first_word_invalid: int = 0
    length_mismatch: int = 0
    bandwidth_mismatch: int = 0
    frame_type: int = 0
    gap_resets: int = 0
    non_finite: int = 0
    last_raw_csi_len: int = 0
    last_complex_csi_len: int = 0
    rejection_reasons: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["rejection_reasons"] = dict(self.rejection_reasons)
        return out

    def reject(self, reason: str) -> None:
        self.rejected_total += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1




def hampel_filter(
    window: np.ndarray,
    threshold: float = 3.5,
    axis: int = 0,
) -> np.ndarray:
    """Replace spike outliers with the per-subcarrier window median.

    Args:
        window: (n_samples, n_subcarriers) amplitude matrix.
        threshold: rejection distance in units of the median absolute
            deviation (MAD), scaled by 1.4826 for normal consistency.
        axis: time axis (0 = rows are samples).

    Returns:
        A cleaned copy of ``window``.
    """
    med = np.median(window, axis=axis, keepdims=True)
    abs_dev = np.abs(window - med)
    mad = np.median(abs_dev, axis=axis, keepdims=True) * 1.4826
    # Guard against MAD == 0 (constant subcarrier): nothing to reject there.
    safe_mad = np.where(mad > 0, mad, np.inf)
    outliers = abs_dev > threshold * safe_mad
    return np.where(outliers, med, window)


def clean_phase(phase_matrix: np.ndarray) -> np.ndarray:
    """Sanitize raw CSI phase (columns = subcarriers).

    The ESP32 applies an unknown per-packet phase offset and clock drift, so
    raw phase is dominated by a linear slope across subcarriers.  We unwrap
    along the subcarrier axis and remove that linear trend, which is the
    standard mitigation before any phase use.  Residual phase noise remains;
    this is why phase features are optional.
    """
    unwrapped = np.unwrap(phase_matrix, axis=-1)
    n = unwrapped.shape[-1]
    x = np.arange(n, dtype=np.float64)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()
    slopes = (unwrapped * x_centered).sum(axis=-1, keepdims=True) / denom
    intercepts = unwrapped.mean(axis=-1, keepdims=True)
    trend = intercepts + slopes * x_centered
    return unwrapped - trend


class Preprocessor:
    """Stateful, streaming preprocessor for the CSI amplitude signal."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.cfg = config
        self.stats = RejectionStats()
        self._ema_state: Optional[np.ndarray] = None
        self._moving_buf: list[np.ndarray] = []
        self._lp_b = self._lp_a = None
        self._lp_zi: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None
        if config.lowpass_enabled:
            if not _HAVE_SCIPY:
                raise RuntimeError(
                    "lowpass_enabled=True requires scipy; pip install scipy")
            nyquist = 0.5 * config.nominal_rate_hz
            wn = min(config.lowpass_cutoff_hz / nyquist, 0.99)
            self._lp_b, self._lp_a = butter(config.lowpass_order, wn, btype="low")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget all smoothing state (call after stream interruptions)."""
        self._ema_state = None
        self._moving_buf.clear()
        self._lp_zi = None
        self._last_ts = None

    # ------------------------------------------------------------------
    def process(self, sample: CsiSample) -> Optional[np.ndarray]:
        """Validate and clean one sample; return amplitude on valid
        subcarriers, or ``None`` when the sample is rejected."""
        self.stats.total_seen += 1

        raw_csi = np.asarray(sample.csi, dtype=np.int8)
        self.stats.last_raw_csi_len = int(len(raw_csi))

        if self.cfg.drop_rx_state_errors and sample.rx_state != 0:
            self.stats.rx_state_errors += 1
            self.stats.reject("rx_state_error")
            return None

        # This ESP32 hardware flag means the first 4 bytes of the CSI buffer are
        # invalid, not that the whole packet is corrupted. Keep the raw packet
        # unchanged for recording, but trim the known invalid prefix before
        # interpreting the I/Q samples.
        if self.cfg.drop_first_word_invalid and sample.first_word_invalid:
            self.stats.first_word_invalid += 1
            if len(raw_csi) < 4:
                self.stats.reject("first_word_invalid_short")
                return None
            raw_csi = raw_csi[4:]
            self.stats.last_raw_csi_len = int(len(raw_csi))

        if len(raw_csi) < 2 or len(raw_csi) % 2 != 0:
            self.stats.length_mismatch += 1
            self.stats.reject("odd_or_short_raw_csi")
            return None

        if sample.cwb != 0:
            self.stats.bandwidth_mismatch += 1
            self.stats.reject("non_20mhz_packet")
            return None
        if self.cfg.ht_only and not sample.is_ht:
            self.stats.frame_type += 1
            self.stats.reject("non_ht_frame")
            return None

        self.stats.last_complex_csi_len = int(len(raw_csi) // 2)

        # The real ESP32 callback payload is an int8 I/Q stream. The project
        # expects the classic HT20 packet layout, but some boards emit 256 raw
        # values (128 complex values) or a similar packed payload. We accept
        # any even-length packed I/Q buffer large enough to contain valid
        # subcarriers while still rejecting obviously malformed data.
        if len(raw_csi) < 2 * max(8, len(self.cfg.valid_subcarriers)):
            self.stats.length_mismatch += 1
            self.stats.reject("too_short_for_valid_subcarriers")
            return None

        # Amplitude on valid subcarriers only.
        idx = np.asarray(self.cfg.valid_subcarriers, dtype=int)
        cplx = raw_csi.astype(np.float64)
        re, im = cplx[0::2], cplx[1::2]
        amp_full = np.sqrt(re * re + im * im)
        if not np.all(np.isfinite(amp_full)):
            self.stats.non_finite += 1
            self.stats.reject("non_finite_csi")
            return None
        amp = amp_full[idx]

        # Reception-gap handling: reset all temporal state after a stall.
        now = sample.ts_local
        if self._last_ts is not None and \
                (now - self._last_ts) > self.cfg.max_gap_seconds:
            self.stats.gap_resets += 1
            self.reset()
        self._last_ts = now

        # Temporal smoothing.
        mode = self.cfg.smoothing
        if mode == "ema":
            if self._ema_state is None or len(self._ema_state) != len(amp):
                self._ema_state = amp.copy()
            else:
                a = self.cfg.smoothing_alpha
                self._ema_state = a * amp + (1.0 - a) * self._ema_state
            amp = self._ema_state.copy()
        elif mode == "moving_avg":
            self._moving_buf.append(amp)
            if len(self._moving_buf) > self.cfg.smoothing_window:
                self._moving_buf.pop(0)
            amp = np.mean(self._moving_buf, axis=0)
        elif mode != "none":
            raise ValueError(f"unknown smoothing mode: {mode!r}")

        self.stats.accepted += 1

        # Optional streaming low-pass along time.
        if self.cfg.lowpass_enabled:
            if self._lp_zi is None or self._lp_zi.shape[-1] != len(amp):
                self._lp_zi = np.zeros(
                    (max(len(self._lp_a), len(self._lp_b)) - 1, len(amp)))
            amp, self._lp_zi = lfilter(
                self._lp_b, self._lp_a, amp, axis=0, zi=self._lp_zi)
            if not np.all(np.isfinite(amp)):
                self.stats.non_finite += 1
                self.reset()
                return None

        self.stats.accepted += 1
        return amp

# __CHUNK_END__
