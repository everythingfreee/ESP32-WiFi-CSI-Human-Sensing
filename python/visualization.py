"""PyQt6 live visualization: room view, status panel and debug window.

Threading model (the UI never blocks):

    SerialReceiver (thread) --callbacks--> InferenceEngine
                                   (serial thread)
    MainWindow (Qt main thread) polls SystemState + plot snapshots via a
    QTimer at ``gui.update_hz`` and repaints.

The room view shows the two ESP32 boards, the sensing link and a dot for
the estimated person position (normalized 0.0 = ESP32-A, 1.0 = ESP32-B)
with a fading trail of recent positions plus the raw (unfiltered)
prediction for debugging.  The optional debug window plots CSI amplitude,
RSSI, probabilities and position history with matplotlib.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel,
                             QMainWindow, QVBoxLayout, QWidget)

try:  # matplotlib Qt backend (debug window only)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except ImportError:  # pragma: no cover
    _HAVE_MPL = False




class RoomWidget(QWidget):
    """Top-down room view: ESP32 boards, link line and person dot."""

    def __init__(self, engine_state_getter, trail_getter, room_width: float,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._get_state = engine_state_getter
        self._get_trail = trail_getter
        self.room_width = max(room_width, 1.0)
        self.setMinimumHeight(220)

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        state = self._get_state()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin_x, mid_y = 60.0, h * 0.52
        span = w - 2 * margin_x

        # Sensing link
        painter.setPen(QPen(QColor(120, 120, 130), 2, Qt.PenStyle.DashLine))
        painter.drawLine(int(margin_x), int(mid_y), int(margin_x + span),
                         int(mid_y))

        # ESP32 boards
        for x, name in ((margin_x, "ESP32-A"), (margin_x + span, "ESP32-B")):
            painter.setPen(QPen(QColor(30, 30, 30), 2))
            painter.setBrush(QColor(60, 130, 246))
            painter.drawEllipse(int(x) - 9, int(mid_y) - 9, 18, 18)
            painter.setPen(QPen(QColor(90, 90, 90)))
            painter.drawText(int(x) - 30, int(mid_y) + 34, name)

        # Scale labels 0.0 / 0.5 / 1.0
        painter.setPen(QPen(QColor(140, 140, 140)))
        for frac, label in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
            x = margin_x + span * frac
            painter.drawText(int(x) - 10, int(mid_y) + 52, label)

        # Person dot + trail
        if state.presence_label == "PERSON" and \
                state.position_filtered is not None:
            confidence = state.confidence if state.confidence is not None \
                else state.presence_probability or 0.0
            trail = self._get_trail()
            for i, old in enumerate(trail):
                alpha = int(60 * (i + 1) / max(len(trail), 1))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(30, 100, 255, alpha))
                tx = margin_x + span * float(old)
                painter.drawEllipse(int(tx) - 4, int(mid_y) - 4, 8, 8)

            # raw prediction marker (debug)
            if state.position_raw is not None:
                rx = margin_x + span * float(state.position_raw)
                painter.setPen(QPen(QColor(255, 140, 0), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(rx) - 9, int(mid_y) - 9, 18, 18)

            px = margin_x + span * float(state.position_filtered)
            radius = 10 + 4 * float(np.clip(confidence, 0, 1))
            painter.setPen(QPen(QColor(20, 60, 160), 2))
            painter.setBrush(QColor(40, 110, 255))
            painter.drawEllipse(int(px - radius), int(mid_y - radius),
                                int(2 * radius), int(2 * radius))
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(int(px) - 12, int(mid_y) - 16,
                             f"{state.position_filtered:.2f}")
        else:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(int(w / 2) - 70, int(mid_y) - 24,
                             "no person detected")

        painter.end()


class StatusPanel(QWidget):
    """Text status block mirroring the specified UI layout."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(12, 8, 12, 8)
        self._labels: dict = {}
        rows = [
            ("presence", "Presence:"),
            ("movement", "Movement:"),
            ("position", "Position:"),
            ("position_raw", "Raw position:"),
            ("confidence", "Confidence:"),
            ("csi_rate", "CSI rate:"),
            ("latency", "Inference latency:"),
            ("connection", "Connection:"),
            ("model", "Model:"),
            ("packets", "Packets (ok/malformed):"),
        ]
        for i, (key, title) in enumerate(rows):
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("--")
            value_label.setStyleSheet("font-family: Menlo, monospace;")
            grid.addWidget(title_label, i, 0)
            grid.addWidget(value_label, i, 1)
            self._labels[key] = value_label

    # ------------------------------------------------------------------
    def update_state(self, s) -> None:
        """Refresh all values from a SystemState snapshot."""
        self._labels["presence"].setText(s.presence_label)
        self._labels["movement"].setText(s.movement_label)
        pos = f"{s.position_filtered:.2f}" \
            if s.position_filtered is not None else "--"
        raw = f"{s.position_raw:.2f}" \
            if s.position_raw is not None else "--"
        conf = f"{s.confidence * 100:.0f}%" \
            if s.confidence is not None else "--"
        self._labels["position"].setText(pos)
        self._labels["position_raw"].setText(raw)
        self._labels["confidence"].setText(conf)
        self._labels["csi_rate"].setText(f"{s.csi_rate:.1f} samples/s")
        self._labels["latency"].setText(f"{s.latency_ms:.1f} ms")
        self._labels["connection"].setText(s.connection)
        color = "#1a9c48" if s.connection == "ONLINE" else "#c62828"
        self._labels["connection"].setStyleSheet(
            f"color: {color}; font-weight: bold;")
        self._labels["model"].setText(s.model_name)
        self._labels["packets"].setText(f"{s.packets} / {s.malformed}")


class DebugWindow(QMainWindow):
    """Optional live debug plots (CSI amplitude, RSSI, probabilities,
    position). Requires matplotlib."""

    def __init__(self, engine, plot_samples: int = 300,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wi-Fi CSI Tracker — Debug")
        self.resize(860, 640)
        self._engine = engine
        self._plot_samples = plot_samples
        self._have_mpl = _HAVE_MPL
        central = QWidget(self)
        layout = QVBoxLayout(central)
        if not _HAVE_MPL:
            layout.addWidget(QLabel(
                "matplotlib is not installed - debug plots unavailable."
                " (pip install matplotlib)"))
        else:
            self._fig = Figure(figsize=(8, 6))
            self._canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._canvas)
            self._ax_amp = self._fig.add_subplot(2, 2, 1)
            self._ax_rssi = self._fig.add_subplot(2, 2, 2)
            self._ax_prob = self._fig.add_subplot(2, 2, 3)
            self._ax_pos = self._fig.add_subplot(2, 2, 4)
            self._prob_history = {"presence": [], "movement": []}
            self._fig.tight_layout()
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Redraw the plots from engine snapshots (called on a timer)."""
        if not self._have_mpl:
            return
        data = self._engine.snapshot_for_plot()
        amps = data["amplitudes"][-self._plot_samples:]
        rssi = data["rssi"][-self._plot_samples:]
        positions = data["positions"][-self._plot_samples:]

        self._ax_amp.clear()
        if amps:
            matrix = np.vstack(amps)
            for idx in np.linspace(0, matrix.shape[1] - 1, 6, dtype=int):
                self._ax_amp.plot(matrix[:, idx], linewidth=0.8)
        self._ax_amp.set_title("CSI amplitude (selected subcarriers)")
        self._ax_amp.set_xlabel("samples")

        self._ax_rssi.clear()
        self._ax_rssi.plot(rssi, color="tab:red", linewidth=0.9)
        self._ax_rssi.set_title("RSSI (dBm)")
        self._ax_rssi.set_xlabel("samples")

        self._ax_prob.clear()
        for key, color in (("presence", "tab:blue"), ("movement", "tab:green")):
            hist = self._prob_history[key][-self._plot_samples:]
            self._ax_prob.plot(hist, label=key, color=color, linewidth=0.9)
        self._ax_prob.set_ylim(-0.05, 1.05)
        self._ax_prob.set_title("presence / movement probability")
        self._ax_prob.legend(loc="upper right")
        self._ax_prob.set_xlabel("updates")

        self._ax_pos.clear()
        self._ax_pos.plot(positions, color="tab:purple", linewidth=1.0)
        self._ax_pos.set_ylim(-0.05, 1.05)
        self._ax_pos.set_title("filtered position")
        self._ax_pos.set_xlabel("updates")

        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def record_probabilities(self, state) -> None:
        """Append current probabilities to the plot history."""
        if not self._have_mpl:
            return
        self._prob_history["presence"].append(
            state.presence_probability if state.presence_probability
            is not None else 0.0)
        self._prob_history["movement"].append(
            state.movement_probability if state.movement_probability
            is not None else 0.0)



class MainWindow(QMainWindow):
    """Live tracker window (room view + status + optional debug view)."""

    def __init__(self, engine, receiver, cfg: dict) -> None:
        super().__init__()
        self.engine = engine
        self.receiver = receiver
        self.cfg = cfg
        self.setWindowTitle("Wi-Fi CSI Human Tracker")
        self.resize(760, 640)

        self._trail = []
        central = QWidget(self)
        outer = QVBoxLayout(central)

        if engine.models is None:
            warn = QLabel("No trained model loaded - presence/movement are "
                          "heuristic and position is disabled.\n"
                          "Collect data (dataset_collector.py) and train "
                          "(train.py) to enable position tracking.")
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "background-color: #fff3cd; color: #664d03; padding: 8px;"
                "border: 1px solid #ffe69c;")
            outer.addWidget(warn)

        self.room = RoomWidget(lambda: self.engine.state,
                               lambda: list(self._trail),
                               float(cfg.get("room_width", 5.0)))
        outer.addWidget(self.room, stretch=3)

        self.status_panel = StatusPanel()
        outer.addWidget(self.status_panel, stretch=2)

        toggle_row = QHBoxLayout()
        self.debug_check = QCheckBox("Show debug window")
        self.debug_check.toggled.connect(self._toggle_debug)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.debug_check)
        outer.addLayout(toggle_row)
        self.setCentralWidget(central)

        self.debug_window: Optional[DebugWindow] = None
        update_hz = int(cfg.get("gui", {}).get("update_hz", 30))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(max(16, int(1000 / max(update_hz, 1))))

    # ------------------------------------------------------------------
    def _toggle_debug(self, checked: bool) -> None:
        if checked:
            if self.debug_window is None:
                self.debug_window = DebugWindow(
                    self.engine,
                    int(self.cfg.get("gui", {}).get("debug_plot_samples", 300)),
                    parent=self)
            self.debug_window.show()
        elif self.debug_window is not None:
            self.debug_window.hide()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        """Timer callback: pull the latest state and repaint."""
        update_link_status_threadsafe(self.engine, self.receiver)
        state = self.engine.state
        if state.presence_label == "PERSON" and \
                state.position_filtered is not None:
            self._trail.append(state.position_filtered)
            if len(self._trail) > 60:
                self._trail.pop(0)
        else:
            if self._trail:
                self._trail.pop(0)
        self.status_panel.update_state(state)
        self.room.update()
        if self.debug_window is not None and self.debug_window.isVisible():
            self.debug_window.record_probabilities(state)
            self.debug_window.refresh()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Stop the serial thread when the window closes."""
        self._timer.stop()
        if self.receiver is not None:
            self.receiver.stop()
            self.receiver.join(timeout=3.0)
        if self.debug_window is not None:
            self.debug_window.close()
        super().closeEvent(event)


def update_link_status_threadsafe(engine, receiver) -> None:
    """Push the receiver's link stats into the engine state."""
    try:
        engine.update_link_status(
            receiver.is_online if receiver is not None else False,
            receiver.packet_rate if receiver is not None else 0.0,
            receiver.packets if receiver is not None else 0,
            receiver.malformed if receiver is not None else 0)
    except Exception:  # noqa: BLE001 - never crash the UI timer
        pass
