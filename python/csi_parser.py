"""Parse the JSON Lines CSI stream produced by the ESP32-B receiver firmware.

The firmware emits one JSON object per CSI sample (see
``firmware/esp32_receiver/esp32_receiver.ino``). Every field below maps
directly to a documented member of the ESP-IDF structures
``wifi_csi_info_t`` / ``wifi_pkt_rx_ctrl_t`` -- nothing is invented:

===================  =======================================================
JSON field           meaning (ESP-IDF documentation)
===================  =======================================================
timestamp            int64 us from ``esp_timer_get_time()`` at callback entry
seq                  monotonically increasing receiver-side counter
rssi                 ``rx_ctrl.rssi`` (dBm)
channel              ``rx_ctrl.channel`` (primary Wi-Fi channel)
secondary_channel    ``rx_ctrl.secondary_channel`` (0 none / 1 above / 2 below)
sig_mode             0: 11b/g, 1: HT (11n), 3: VHT
mcs                  HT MCS index (valid for HT packets)
cwb                  channel bandwidth: 0 = 20 MHz, 1 = 40 MHz
rate                 PHY rate encoding (11b/g packets only)
aggregation          0: MPDU, 1: AMPDU
stbc                 0: non-STBC, >0: STBC
fec_coding           LDPC flag (HT packets)
sgi                  short guard interval flag
noise_floor          RF noise floor (dBm)
ampdu_cnt            number of aggregated subframes
sig_len              received packet length incl. FCS
rx_state             0 = no error, other values = reception error
ant                  antenna number
timestamp_wifi      ``rx_ctrl.timestamp`` in us (32-bit, wraps ~71.6 min)
mac                  source MAC address (hex string)
first_word_invalid   1 = the first 4 CSI bytes are invalid (HW limitation)
csi_len              number of valid bytes in ``csi``
csi                  int8 array: (real, imag) pair per subcarrier
===================  =======================================================

For a 20 MHz channel the CSI buffer is 128 bytes = 64 subcarrier slots;
subcarriers 0..26 and 32..58 are valid per Espressif's CSI guide.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Number of subcarrier slots the ESP32 reports for a 20 MHz channel.
SUBCARRIERS_20MHZ = 64
#: Valid subcarrier indices for 20 MHz (others are DC/guard/pilot slots).
VALID_SUBCARRIERS_20MHZ: Tuple[int, ...] = tuple(
    list(range(0, 27)) + list(range(32, 59))
)
#: Human-readable names for ``sig_mode`` values.
SIGNAL_MODES = {0: "11bg", 1: "HT", 3: "VHT"}

_REQUIRED_KEYS = ("timestamp", "rssi", "channel", "csi")


class PacketError(ValueError):
    """Raised when a received CSI line is malformed or inconsistent."""


@dataclass
class CsiSample:
    """One validated CSI measurement (raw values are preserved as-is)."""

    ts_esp32_us: int          # firmware esp_timer timestamp (us)
    ts_local: float           # local reception time (time.time(), seconds)
    seq: int
    rssi: int
    channel: int
    secondary_channel: int
    sig_mode: int
    mcs: int
    cwb: int
    rate: int
    aggregation: int
    stbc: int
    fec_coding: int
    sgi: int
    noise_floor: int
    ampdu_cnt: int
    sig_len: int
    rx_state: int
    ant: int
    ts_wifi_us: int
    mac: str
    first_word_invalid: bool
    csi: np.ndarray           # int8 array (real, imag pairs), raw values

    @property
    def n_subcarriers(self) -> int:
        """Number of subcarrier slots contained in the raw CSI buffer."""
        return len(self.csi) // 2

    @property
    def is_ht(self) -> bool:
        """True when the packet was an HT (11n) data frame."""
        return self.sig_mode == 1




def parse_line(line: str, local_ts: float) -> Optional[CsiSample]:
    """Parse one serial line into a :class:`CsiSample`.

    Returns ``None`` for blank lines and ``#``-prefixed status lines.
    Raises :class:`PacketError` for malformed or inconsistent packets.
    """
    text = line.strip()
    if not text:
        return None
    if text.startswith("#"):
        return None  # firmware status line, not a CSI sample

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PacketError(f"malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise PacketError("JSON line is not an object")

    for key in _REQUIRED_KEYS:
        if key not in obj:
            raise PacketError(f"missing required field: {key}")

    csi_raw = obj["csi"]
    if not isinstance(csi_raw, list) or len(csi_raw) == 0:
        raise PacketError("csi must be a non-empty array")
    if len(csi_raw) % 2 != 0:
        raise PacketError("csi array must contain (real, imag) pairs")

    csi_values: List[int] = []
    for value in csi_raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PacketError(f"non-numeric CSI value: {value!r}")
        if isinstance(value, float):
            if not math.isfinite(value) or value != int(value):
                raise PacketError(f"non-integer CSI value: {value!r}")
            value = int(value)
        if not (-128 <= value <= 127):
            raise PacketError(f"CSI value out of int8 range: {value}")
        csi_values.append(value)

    csi_len = obj.get("csi_len")
    if csi_len is not None and int(csi_len) != len(csi_values):
        raise PacketError(
            f"csi_len mismatch: header says {csi_len}, got {len(csi_values)}")

    try:
        sample = CsiSample(
            ts_esp32_us=int(obj["timestamp"]),
            ts_local=float(local_ts),
            seq=int(obj.get("seq", 0)),
            rssi=int(obj["rssi"]),
            channel=int(obj["channel"]),
            secondary_channel=int(obj.get("secondary_channel", 0)),
            sig_mode=int(obj.get("sig_mode", 0)),
            mcs=int(obj.get("mcs", 0)),
            cwb=int(obj.get("cwb", 0)),
            rate=int(obj.get("rate", 0)),
            aggregation=int(obj.get("aggregation", 0)),
            stbc=int(obj.get("stbc", 0)),
            fec_coding=int(obj.get("fec_coding", 0)),
            sgi=int(obj.get("sgi", 0)),
            noise_floor=int(obj.get("noise_floor", -100)),
            ampdu_cnt=int(obj.get("ampdu_cnt", 1)),
            sig_len=int(obj.get("sig_len", 0)),
            rx_state=int(obj.get("rx_state", 0)),
            ant=int(obj.get("ant", 0)),
            ts_wifi_us=int(obj.get("timestamp_wifi", 0)),
            mac=str(obj.get("mac", "")),
            first_word_invalid=bool(obj.get("first_word_invalid", 0)),
            csi=np.asarray(csi_values, dtype=np.int8),
        )
    except (TypeError, ValueError) as exc:
        raise PacketError(f"invalid field type: {exc}") from exc
    return sample


def raw_to_complex(csi: np.ndarray, first_word_invalid: bool = False) -> np.ndarray:
    """Convert the raw packed int8 I/Q stream to a complex float array.

    Classic ESP32 CSI stores real and imaginary values as consecutive int8
    values in the callback buffer. ``first_word_invalid`` indicates the first
    4 raw bytes are invalid for this platform and must be discarded before
    interpreting the packed complex samples. The original raw sequence is kept
    intact in :class:`CsiSample`; callers can opt into the documented invalid
    prefix trimming when building amplitude/phase arrays.
    """
    arr = np.asarray(csi, dtype=np.float64)
    if first_word_invalid:
        if len(arr) < 4:
            raise PacketError(
                "first_word_invalid set but CSI buffer is too short to strip "
                "the documented invalid prefix")
        arr = arr[4:]
    if len(arr) % 2 != 0:
        raise PacketError("CSI buffer length must be even after removing the invalid prefix")
    return arr[0::2] + 1j * arr[1::2]


def amplitude(sample: CsiSample) -> np.ndarray:
    """CSI amplitude for every packed complex sample in the raw buffer."""
    return np.abs(raw_to_complex(sample.csi, sample.first_word_invalid))


def phase(sample: CsiSample) -> np.ndarray:
    """CSI phase for every packed complex sample in the raw buffer."""
    return np.angle(raw_to_complex(sample.csi, sample.first_word_invalid))


def valid_amplitude(
    sample: CsiSample,
    valid_subcarriers: Sequence[int] = VALID_SUBCARRIERS_20MHZ,
) -> np.ndarray:
    """Amplitude restricted to the valid subcarrier indices."""
    amp = amplitude(sample)
    idx = np.asarray(valid_subcarriers, dtype=int)
    if idx.size and (idx.max() >= len(amp) or idx.min() < 0):
        raise PacketError(
            f"subcarrier index out of range for buffer of {len(amp)} slots")
    return amp[idx]
