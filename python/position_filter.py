"""Temporal position filtering for the 1-D position estimate.

Raw model predictions jump around; a filter makes the visible dot move
smoothly without hiding real motion.  Available filters:

* ``ema`` (default) — exponential moving average:
  ``filtered = alpha * raw + (1 - alpha) * previous``.  Lower ``alpha``
  means smoother output but more lag.  This is the V1 default because it
  is simple, causal and tunable.
* ``median`` — median of the last N accepted positions; robust against
  single-prediction outliers at the cost of a small fixed lag.

Both filters reset their state when no predictions arrive for longer than
``max_gap_seconds`` (e.g. the person left the link line), so a stale value
never leaks into a new detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class PositionFilterConfig:
    """Knobs (mirrors the top-level ``smoothing_factor`` etc.)."""

    filter_type: str = "ema"          # "ema" | "median"
    alpha: float = 0.35               # EMA weight of the new raw prediction
    median_window: int = 5
    max_gap_seconds: float = 2.0

    @classmethod
    def from_config(cls, config: dict) -> "PositionFilterConfig":
        return cls(
            filter_type="ema",
            alpha=float(config.get("smoothing_factor", 0.35)),
            median_window=5,
            max_gap_seconds=2.0,
        )


class PositionFilter:
    """Stateful filter mapping raw position predictions to display values."""

    def __init__(self, config: Optional[PositionFilterConfig] = None) -> None:
        self.cfg = config or PositionFilterConfig()
        if self.cfg.filter_type not in ("ema", "median"):
            raise ValueError(
                f"unknown position filter type: {self.cfg.filter_type!r}")
        if not 0.0 < self.cfg.alpha <= 1.0:
            raise ValueError("smoothing alpha must be in (0, 1]")
        self._state: Optional[float] = None
        self._history: Deque[float] = deque(maxlen=self.cfg.median_window)
        self._last_ts: Optional[float] = None

    def reset(self) -> None:
        """Forget the filtering state (person lost / stream restarted)."""
        self._state = None
        self._history.clear()
        self._last_ts = None

    def filter(self, position: float, timestamp: float) -> Optional[float]:
        """Feed one raw prediction in [0, 1]; return the filtered value."""
        if position is None:
            return None
        if not 0.0 <= float(position) <= 1.0:
            # Clamp instead of propagating impossible values.
            position = min(max(float(position), 0.0), 1.0)
        if self._last_ts is not None and \
                (timestamp - self._last_ts) > self.cfg.max_gap_seconds:
            self.reset()
        self._last_ts = timestamp

        if self.cfg.filter_type == "ema":
            if self._state is None:
                self._state = float(position)
            else:
                self._state = (self.cfg.alpha * float(position)
                               + (1.0 - self.cfg.alpha) * self._state)
            return self._state

        self._history.append(float(position))
        self._state = float(sorted(self._history)[len(self._history) // 2])
        return self._state
