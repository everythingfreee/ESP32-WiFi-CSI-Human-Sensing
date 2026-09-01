"""USB-serial receiver for the ESP32-B CSI stream.

Responsibilities:
  * find or accept the ESP32 serial port (never hardcoded);
  * configure the baud rate;
  * read incoming lines in a background thread;
  * detect malformed packets and count them;
  * parse CSI packets (:mod:`csi_parser`);
  * add a local reception timestamp to every sample;
  * optionally record every raw sample losslessly to CSV;
  * pass valid packets to a callback (the processing pipeline).

CLI example::

    python serial_receiver.py --port /dev/cu.usbmodemXXXX --baud 921600
    python serial_receiver.py --list           # show candidate ports
    python serial_receiver.py --record --duration 30
    python serial_receiver.py --port /dev/cu.usbmodemXXXX --raw-debug
                                          # blindly print the first 30 raw lines
                                          # (binary-safe, before parsing).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import serial
from serial.tools import list_ports

from csi_parser import CsiSample, PacketError, parse_line

LOG = logging.getLogger("wifi_sensing.serial")

#: Port name patterns considered "probably an ESP32" during auto-detection.
_CANDIDATE_PATTERNS = ("usbmodem", "usbserial", "ttyusb", "ttyacm", "cu.usb")

CallbackFn = Callable[[CsiSample], None]


def find_serial_ports() -> List[str]:
    """Return candidate serial ports for the ESP32 (macOS/Linux/Windows)."""
    candidates: List[str] = []
    for info in list_ports.comports():
        name = (info.device or "").lower()
        if any(pat in name for pat in _CANDIDATE_PATTERNS):
            candidates.append(info.device)
    return sorted(candidates)


class RawSessionWriter:
    """Lossless, incremental raw-CSI CSV writer.

    The column layout is fixed by the first recorded packet (the 20 MHz
    format always carries the same length); any later packet with a
    different CSI length is redirected to an ``*_irregular.csv`` file so
    that no raw data is ever silently destroyed or truncated.
    """

    META_FIELDS = (
        "ts_local", "ts_esp32_us", "ts_wifi_us", "seq", "rssi", "channel",
        "secondary_channel", "sig_mode", "mcs", "cwb", "rate", "aggregation",
        "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "sig_len",
        "rx_state", "ant", "mac", "first_word_invalid", "csi_len",
    )

    def __init__(self, path: Path, first_sample: CsiSample) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._n_csi = len(first_sample.csi)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(
            list(self.META_FIELDS)
            + [f"csi_{i}" for i in range(self._n_csi)])
        self.rows_written = 0
        self.irregular_path: Optional[Path] = None
        self._irregular_fh = None
        self._irregular_writer = None

    def write(self, sample: CsiSample) -> None:
        """Append one raw sample (never blocks the serial reader for long)."""
        if len(sample.csi) != self._n_csi:
            if self._irregular_writer is None:
                self.irregular_path = self.path.with_name(
                    self.path.stem + "_irregular.csv")
                self._irregular_fh = open(
                    self.irregular_path, "w", newline="", encoding="utf-8")
                self._irregular_writer = csv.writer(self._irregular_fh)
                self._irregular_writer.writerow(
                    list(self.META_FIELDS)
                    + [f"csi_{i}" for i in range(len(sample.csi))])
            assert self._irregular_writer is not None
            self._irregular_writer.writerow(
                self._meta_row(sample) + list(int(v) for v in sample.csi))
            self._irregular_fh.flush()  # type: ignore[union-attr]
            return
        self._writer.writerow(
            self._meta_row(sample) + list(int(v) for v in sample.csi))
        if self.rows_written % 50 == 0:
            self._fh.flush()  # periodic flush; keeps crashes cheap to recover
        self.rows_written += 1

    @staticmethod
    def _meta_row(s: CsiSample) -> list:
        return [
            f"{s.ts_local:.6f}", s.ts_esp32_us, s.ts_wifi_us, s.seq, s.rssi,
            s.channel, s.secondary_channel, s.sig_mode, s.mcs, s.cwb, s.rate,
            s.aggregation, s.stbc, s.fec_coding, s.sgi, s.noise_floor,
            s.ampdu_cnt, s.sig_len, s.rx_state, s.ant, s.mac,
            int(s.first_word_invalid), len(s.csi),
        ]

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
        if self._irregular_fh and not self._irregular_fh.closed:
            self._irregular_fh.flush()
            self._irregular_fh.close()



class SerialReceiver(threading.Thread):
    """Background thread that streams parsed CSI samples to a callback.

    The thread survives serial disconnects (reconnects with backoff) and
    malformed packets (counted, never fatal).  The callback runs in the
    receiver thread; it must be fast or it will cause queue drops on the
    ESP32 side.
    """

    def __init__(
        self,
        port: str,
        baud: int = 921600,
        callback: Optional[CallbackFn] = None,
        record_path: Optional[Path] = None,
        reconnect_delay: float = 2.0,
        raw_debug: int = 0,
    ) -> None:
        super().__init__(name="serial_receiver", daemon=True)
        if port == "auto":
            found = find_serial_ports()
            if not found:
                raise RuntimeError(
                    "No serial port auto-detected. Connect the ESP32 or pass "
                    "--port explicitly (see --list).")
            port = found[0]
            LOG.info("auto-detected serial port: %s", port)
        self.port = port
        self.baud = int(baud)
        self.callback = callback
        self.reconnect_delay = reconnect_delay

        self.packets: int = 0
        self.malformed: int = 0
        self.io_errors: int = 0
        self.last_packet_ts: float = 0.0
        self.recent_timestamps: List[float] = []

        self._stop_event = threading.Event()
        self._ser: Optional[serial.Serial] = None
        self._writer: Optional[RawSessionWriter] = None
        self._record_path = record_path

    # ------------------------------------------------------------------
    @property
    def is_online(self) -> bool:
        """True while packets arrived within the last 3 seconds."""
        return (self.last_packet_ts > 0.0
                and (time.time() - self.last_packet_ts) < 3.0)

    @property
    def packet_rate(self) -> float:
        """Measured CSI packet rate (packets/s) over the last ~2 seconds."""
        now = time.time()
        self.recent_timestamps = [
            t for t in self.recent_timestamps if now - t <= 2.0]
        return len(self.recent_timestamps) / 2.0

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Ask the thread to stop and close the port (idempotent)."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    def run(self) -> None:
        buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                if self._ser is None or not getattr(self._ser, "is_open", False):
                    self._ser = serial.Serial(
                        self.port, self.baud, timeout=1)
                    buffer.clear()
                    LOG.info("serial connected: %s @ %d baud",
                             self.port, self.baud)
                chunk = self._ser.read(4096)
                if chunk:
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw_line, _, rest = buffer.partition(b"\n")
                        buffer = bytearray(rest)
                        self._handle_line(
                            raw_line.decode("utf-8", errors="replace"))
                else:
                    # timeout tick: nothing this second
                    pass
            except (serial.SerialException, OSError) as exc:
                self.io_errors += 1
                LOG.warning("serial I/O problem (%s); retrying in %.1fs",
                            exc, self.reconnect_delay)
                self._close_port()
                self._stop_event.wait(self.reconnect_delay)
            finally:
                if self._stop_event.is_set():
                    self._close_port()
        if self._writer is not None:
            self._writer.close()

    # ------------------------------------------------------------------
    def _handle_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        if text.startswith("#") and not text.startswith("#CSI "):
            LOG.debug("ignored: %s", text)
            return
        if text.startswith("#CSI "):
            json_part = text[len("#CSI "):]
            ts = time.time()
            try:
                sample = parse_line(json_part, ts)
            except PacketError as exc:
                self.malformed += 1
                LOG.warning("malformed packet #%d: %s", self.malformed, exc)
                return
            if sample is None:
                return

            # Now sample is valid
            self.packets += 1
            self.last_packet_ts = ts
            self.recent_timestamps.append(ts)

            # Initialize writer only on first valid packet
            if self._record_path is not None and self._writer is None:
                self._writer = RawSessionWriter(self._record_path, sample)
                LOG.info("recording raw samples to %s", self._record_path)
            if self._writer is not None:
                self._writer.write(sample)

            if self.callback is not None:
                try:
                    self.callback(sample)
                except Exception:
                    LOG.exception("callback failed on packet %d", self.packets)

            if self._record_path is not None and self._writer is None:
                self._writer = RawSessionWriter(self._record_path, sample)
                LOG.info("recording raw samples to %s", self._record_path)
            if self._writer is not None:
                self._writer.write(sample)

            if self.callback is not None:
                try:
                    self.callback(sample)
                except Exception:  # noqa: BLE001 - protect the reader thread
                    LOG.exception("callback failed on packet %d", self.packets)

    # ------------------------------------------------------------------
    def _close_port(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None



def main(argv: Optional[List[str]] = None) -> int:
    """CLI: live serial CSI reader with optional raw recording."""
    parser = argparse.ArgumentParser(
        description="Read the ESP32 CSI stream over USB serial")
    parser.add_argument("--port", default="auto",
                        help="serial port ('auto' scans, e.g. "
                             "/dev/cu.usbmodemXXXX)")
    parser.add_argument("--baud", type=int, default=921600,
                        help="baud rate (must match receiver firmware)")
    parser.add_argument("--list", action="store_true",
                        help="list candidate serial ports and exit")
    parser.add_argument("--record", action="store_true",
                        help="record every raw sample to data/raw/")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after N seconds (0 = run forever)")
    parser.add_argument("--max-packets", type=int, default=0,
                        help="stop after N valid packets (0 = unlimited)")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--debug", action="store_true",
                        help="DEBUG logging (noisy)")
    args = parser.parse_args(argv)

    from config import load_config, resolve_path, setup_logging
    cfg = load_config(args.config)
    cfg["log_level"] = "DEBUG" if args.debug else cfg.get("log_level", "INFO")
    setup_logging(cfg)

    if args.list:
        ports = find_serial_ports()
        if not ports:
            print("No candidate serial ports found. Is the ESP32 plugged in?")
            print("All ports:")
            for info in list_ports.comports():
                print(f"  {info.device}  ({info.description})")
        else:
            print("Candidate ESP32 ports:")
            for p in ports:
                print(f"  {p}")
        return 0

    record_path = None
    if args.record:
        record_path = resolve_path(cfg, "raw_dir") / (
            f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    receiver = SerialReceiver(args.port, args.baud, callback=None,
                              record_path=record_path)
    receiver.start()
    print(f"Reading {args.port} @ {args.baud} baud "
          f"{'[recording to data/raw] ' if args.record else ''}"
          f"(Ctrl+C to stop)...")
    start = time.time()
    try:
        while True:
            time.sleep(1.0)
            print(f"\rpackets={receiver.packets} rate={receiver.packet_rate:.1f}/s "
                  f"malformed={receiver.malformed} io_errors={receiver.io_errors}",
                  end="", flush=True)
            if args.duration and (time.time() - start) >= args.duration:
                break
            if args.max_packets and receiver.packets >= args.max_packets:
                break
    except KeyboardInterrupt:
        print()
    finally:
        receiver.stop()
        receiver.join(timeout=5.0)
        if record_path is not None:
            print(f"raw session saved: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
