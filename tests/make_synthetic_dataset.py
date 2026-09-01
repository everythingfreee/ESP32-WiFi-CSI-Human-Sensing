"""SYNTHETIC dataset generator — for pipeline testing ONLY.

*** The values produced by this script are NOT real Wi-Fi CSI measurements. ***

They are simulated channel patterns with a smooth position dependence plus
noise, written through the *real* parsing/preprocessing/feature pipeline so
that the full toolchain (parse -> preprocess -> features -> dataset ->
train -> evaluate) can be tested without hardware.

Never use the resulting dataset for real experiments, and never quote its
metrics as real-world results.

Usage:
    python tests/make_synthetic_dataset.py \
        --out data/processed/dataset_synthetic.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the python/ package importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from csi_parser import VALID_SUBCARRIERS_20MHZ, parse_line  # noqa: E402
from feature_extraction import FeatureConfig, FeatureExtractor  # noqa: E402
from preprocessing import PreprocessConfig, Preprocessor  # noqa: E402

RATE = 60  # samples per second
N_VALID = len(VALID_SUBCARRIERS_20MHZ)


def _csi_payload(rng: np.random.Generator, phase0: np.ndarray,
                 amps: np.ndarray) -> list:
    """Build a firmware-shaped 128-byte int8 CSI buffer for valid slots."""
    full = np.zeros(64, dtype=complex)
    full[list(VALID_SUBCARRIERS_20MHZ)] = amps * np.exp(1j * phase0)
    drift = rng.normal(0, 0.05)  # small per-packet phase noise
    full = full * np.exp(1j * drift)
    re = np.clip(np.round(full.real), -127, 127).astype(int)
    im = np.clip(np.round(full.imag), -127, 127).astype(int)
    raw = np.empty(128, dtype=int)
    raw[0::2], raw[1::2] = re, im
    return raw.tolist()


def _packet(rng: np.random.Generator, phase0: np.ndarray, pos: float,
            person: bool, seq: int, ts_us: int) -> str:
    """One simulated CSI sample as a firmware-style JSON line."""
    i = np.arange(N_VALID) / N_VALID
    # Static channel profile: smooth ripple across subcarriers.
    amps = 42.0 + 10.0 * np.sin(2 * np.pi * (i + 0.15 * pos))
    if person:
        # Person attenuates a band of subcarriers whose center slides with
        # the person's position along the link.
        center = 0.25 + 0.5 * pos
        depth = 9.0 + 2.0 * np.sin(2 * np.pi * (i + pos))
        amps = amps - depth * np.exp(-((i - center) ** 2) / 0.015)
    amps += rng.normal(0, 0.8, N_VALID)  # measurement noise
    rssi = int(round(-46 - 6 * abs(pos - 0.5) + rng.normal(0, 0.7)))
    obj = {
        "timestamp": ts_us,
        "seq": seq,
        "rssi": rssi,
        "channel": 6,
        "secondary_channel": 0,
        "sig_mode": 1,
        "mcs": 5,
        "cwb": 0,
        "rate": 0,
        "aggregation": 0,
        "stbc": 0,
        "fec_coding": 0,
        "sgi": 0,
        "noise_floor": -96,
        "ampdu_cnt": 1,
        "sig_len": 64,
        "rx_state": 0,
        "ant": 0,
        "timestamp_wifi": ts_us & 0xFFFFFFFF,
        "mac": "AABBCCDDEEFF",
        "first_word_invalid": 0,
        "csi_len": 128,
        "csi": _csi_payload(rng, phase0, np.abs(amps)),
    }
    return json.dumps(obj)



def simulate_session(session_id: str, label: str, position, mode: str,
                     seconds: float, seed: int) -> list:
    """Run one synthetic session through the real processing pipeline."""
    rng = np.random.default_rng(seed)
    phase0 = rng.uniform(-np.pi, np.pi, N_VALID)
    preproc_cfg = PreprocessConfig()
    preprocessor = Preprocessor(preproc_cfg)
    extractor = FeatureExtractor(FeatureConfig(), preproc_cfg)
    rows: list = []
    n_samples = int(seconds * RATE)
    for k in range(n_samples):
        if label == "person":
            pos = position
            if mode == "moving":
                pos = float(np.clip(position + 0.06 * np.sin(2 * np.pi * k /
                                                          (RATE * 8)), 0, 1))
            person = True
        else:
            pos, person = 0.5, False
        line = _packet(rng, phase0, pos, person, k,
                       int(k * 1e6 / RATE) + 1000 * seed)
        sample = parse_line(line, time.time() + k / RATE)
        amp = preprocessor.process(sample)
        if amp is None:
            continue
        fv = extractor.update(amp, sample.rssi, sample.ts_local)
        if fv is None:
            continue
        rows.append(
            [k, session_id, fv.timestamp,
             float(fv.values[fv.names.index("rssi_mean")])]
            + fv.values.tolist()
            + [1 if person else 0, 1 if mode == "moving" else 0,
               pos if person else np.nan])
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a SYNTHETIC dataset for pipeline testing")
    parser.add_argument("--out", default="data/processed/dataset_synthetic.csv")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="simulated seconds per session")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    sessions = []
    seed = args.seed
    for rep in range(2):  # two sessions per condition -> session split works
        sessions.append((f"syn_empty_{rep}", "empty", None, "static"))
        for p in positions:
            sessions.append((f"syn_p{p:.2f}_{rep}", "person", p, "static"))
    sessions.append(("syn_moving_0", "person", 0.5, "moving"))

    all_rows: list = []
    for session_id, label, pos, mode in sessions:
        seed += 1
        rows = simulate_session(session_id, label, pos, mode,
                                args.seconds, seed)
        if not rows:
            print(f"session {session_id}: no rows (pipeline rejected all?)",
                  file=sys.stderr)
            return 1
        print(f"{session_id}: {len(rows)} feature rows")
        all_rows.extend(rows)

    # Column order must match dataset_collector.py exactly.
    from feature_extraction import FeatureExtractor as _FE  # names source
    names = _FE(FeatureConfig(), PreprocessConfig()).feature_names()
    columns = (["sample_id", "session_id", "timestamp", "rssi"]
               + names + ["presence", "movement", "position"])
    df = pd.DataFrame(all_rows, columns=columns)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSYNTHETIC dataset written: {out} ({len(df)} rows, "
          f"{df['session_id'].nunique()} sessions)")
    print("REMINDER: this file contains simulated data, not measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
