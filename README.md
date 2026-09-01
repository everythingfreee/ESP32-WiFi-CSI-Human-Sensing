# ESP32 Wi-Fi CSI Human Position Sensing System

An **experimental** human presence / movement / approximate 1D-position
sensing system built from **two ESP32 DevKit V1 boards** and a Mac.

```
ESP32-A (AP)  ⇄ Wi-Fi link ⇄  ESP32-B (CSI receiver)
                                   │ USB serial (JSON Lines)
                                   ▼
                                Mac · Python
                           preprocess → features → ML models
                                   │
                    presence + movement + position (0.0…1.0)
                                   │
                                   ▼
                      live PyQt6 room visualization (🔵 dot)
```

> ⚠️ **Scientific honesty.** This is an experimental proof-of-concept.
> It estimates the person's position *along the line between the two
> ESP32s* as a normalized value in [0, 1]. It does **not** provide
> centimeter accuracy and cannot claim exact human coordinates — see
> [Known limitations](#known-limitations).

## Hardware

| Item                 | Qty | Role                                        |
|----------------------|----:|---------------------------------------------|
| ESP32 DevKit V1 (classic ESP32, ESP-WROOM-32) | 2 | #1 = AP + traffic generator, #2 = CSI receiver |
| USB cables           | 2  | power + serial to the Mac                   |
| Wi-Fi (2.4 GHz)      | 1  | the sensing link (dedicated AP, fixed channel) |

### Why classic ESP32 (and what differs on other ESP32s)

CSI support differs between ESP32 chips:

* **ESP32 (classic) — supported.** `esp_wifi_set_csi*` APIs, 20 MHz
  channel → 128-byte CSI buffer (64 subcarrier slots), HT20 data frames.
  Used here.
* **ESP32-S2 / ESP32-C3 / ESP32-S3** — CSI is *not* exposed in the same
  way / not available in their mainstream Wi-Fi stacks; a different
  experimental path would be needed. Not targeted here.
* **ESP32-C6** — has 802.11ax CSI support but a different API/board; not
  targeted here.

## Architecture

```
ESP32-A  AP "CSI_SENSOR" ch.6          ESP32-B  STA (ch.6)
   AP + UDP echo :8765   ←─────────────────  UDP probe :8888 → :8765
   echo (HT20 data frame) ──────────────────→  CSI captured per frame
                                                       │
         csi_parser.py ◄── JSON Lines (921600 baud) ────┘
              ▼
        preprocessing.py        (drop invalid, subcarrier select,
                                 Hampel outliers, EMA smoothing)
              ▼
        feature_extraction.py   (windowed amplitude stat/temporal/
                                 spectral features, fixed 180-dim vector)
              ▼
        presence model ▸ movement detector            position model
              ▼                                              ▼
         position_filter.py (EMA/median) ◄──── raw position
              ▼
        PyQt6 MainWindow + optional debug plots
```

Key design decisions:

* **All heavy computation runs on the Mac.** V1 does not run any ML on the
  ESP32.
* **Presence and position are separate models.** Position is only reported
  when presence says `PERSON`.
* **Models are saved once** (`data/models/`) and loaded at startup —
  training never runs at inference time.
* **Session-based data splits.** Neighbouring CSI samples from one recording
  are correlated; splitting by rows would leak information and inflate
  metrics. All train/val/test splits are made per **session** (see
  `python/dataset.py`).

## Project layout

```
ESP32_Sensing/
├── firmware/
│   ├── esp32_ap/            # ESP32-A: AP + UDP echo + heartbeats
│   └── esp32_receiver/      # ESP32-B: STA + CSI capture + serial stream
├── python/
│   ├── main.py              # live GUI launcher
│   ├── serial_receiver.py   # USB serial reader (+ raw CSV recorder)
│   ├── csi_parser.py        # JSON Lines → CsiSample (documented fields)
│   ├── preprocessing.py     # validation/subcarrier/Hampel/EMA/butterworth
│   ├── feature_extraction.py# fixed-size feature vector (180 dims)
│   ├── dataset.py           # load/validate/session-split/--stats
│   ├── dataset_collector.py # collect labeled sessions
│   ├── calibrate.py         # guided empty/person walk-through
│   ├── train.py             # baselines + NN, session split, save models
│   ├── evaluate.py          # honest test metrics from saved models
│   ├── inference.py         # engine: sample → presence/movement/position
│   ├── position_filter.py   # EMA / median position smoothing
│   ├── visualization.py     # PyQt6 room view + status + debug window
│   ├── config.py            # config.yaml loading / logging
│   └── models/              # baseline_model.py (sklearn), neural_network.py (PyTorch)
├── tests/                   # synthetic-data pipeline verification (NOT real data)
├── data/{raw,processed,models}/
├── logs/
├── config.yaml
└── requirements.txt
```

## CSI packet contract (critical for Python compatibility)

The Python pipeline expects the classic ESP32 CSI callback format that the
receiver firmware emits over USB serial. The raw callback payload is a packed
int8 I/Q stream, not a high-level numpy array.

The important contract is:

- `csi` is a flat int8 array of raw CSI bytes in the callback order
- `csi_len` is the raw callback byte count (`len(csi)`)
- On some classic ESP32 CSI callbacks, `first_word_invalid = 1` means the
  first 4 bytes of the raw CSI buffer are known-invalid prefix bytes; they
  are intentionally dropped before I/Q interpretation
- After removing that known invalid prefix, the remaining bytes are read as
  interleaved I/Q samples: `real0, imag0, real1, imag1, ...`
- The parser therefore converts the packed payload as
  `real = csi[0::2]`, `imag = csi[1::2]` after trimming the invalid prefix

The project intentionally verifies packet validity before feature extraction:

- `rx_state != 0` -> reject as corrupted reception
- `first_word_invalid` -> drop only the documented prefix, not the whole packet
- non-even or truncated raw payload -> reject as malformed
- non-20 MHz / non-HT packet -> reject with a diagnostic reason

For the classic ESP32 20 MHz HT20 flow, the expected raw callback length is
typically 128 bytes for the valid payload. The captured real packets in this
repository have shown `csi_len = 256` with `first_word_invalid = 1`, which is
still the same callback contract: a packed int8 I/Q stream with a known
invalid prefix. The Python parser now strips the invalid prefix and then
interprets the remaining raw bytes correctly.

## Hardware / environmental setup

1. Power both boards via USB.
2. Place them on opposite sides of the room, a few meters apart, elevated
   and facing the sensing area (antenna orientation affects CSI).
3. Keep the room **empty of moving objects** during calibration, and keep
   furniture/placement fixed between training and use — CSI depends on the
   environment (see [Known limitations](#known-limitations)).

## Arduino / firmware setup

### Board package (verified on macOS with arduino-cli)

| Component              | Version used in this project                    |
|------------------------|-------------------------------------------------|
| Arduino core          | `esp32:esp32` **2.0.17** (ESP-IDF v4.4.7) — also compiled against `espressif:esp32` 3.3.7 (ESP-IDF v5.5) |
| Board                 | **DOIT ESP32 DEVKIT V1** (FQBN `esp32:esp32:esp32doit-devkit-v1`; alias: "ESP32 Dev Module") |
| Required libraries    | none extra — `WiFi`, `WiFiUdp`, and the Wi-Fi stack's `esp_wifi.h` ship with the core |

How to install the package:

* **Arduino IDE:** Boards Manager → search "esp32" → install
  *"esp32 by Espressif Systems"* **2.0.17** (or 3.x). The classic 3.x core
  upstream package (used with `espressif:` FQBN prefix) is the official
  Espressif Arduino core.
* **arduino-cli:**
  ```bash
  arduino-cli core update-index
  arduino-cli core install esp32:esp32@2.0.17
  ```

### Why the ESP-IDF CSI API is called directly

The Arduino core does **not** wrap CSI (verified: `WiFiGeneric.h` has no
`setCsi`), so both sketches call the raw ESP-IDF functions
`esp_wifi_set_csi_config / esp_wifi_set_csi_rx_cb / esp_wifi_set_csi`
directly.  The core already ships these headers; **no extra library or
ESP-IDF installation is required.** The exact struct fields and callback
signature were verified against the headers shipped with cores 2.0.17 and
3.3.7 (see `firmware/esp32_receiver/README.md`).

### Board settings

For both sketches, the only required settings are:

* Board: **DOIT ESP32 DEVKIT V1**
* Partition scheme: default
* Flash size: default
* Upload speed: default (or 921600/115200 depending on your USB-UART chip)

### Flash ESP32-A

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
arduino-cli upload -p /dev/cu.usbmodemXXXX --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
```

(Arduino IDE: File → Open `firmware/esp32_ap/esp32_ap.ino`, select
"DOIT ESP32 DEVKIT V1", Upload.)

### Flash ESP32-B

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
arduino-cli upload -p /dev/cu.usbmodemXXXX --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
```

### Verify the link + CSI before touching the Mac

1. Open ESP32-A's serial (115200): expect `[AP]` startup info, then
   `stations=1` once B connects.
2. Open ESP32-B's serial: expect `[RX]` connection info and a fresh JSON
   line per probe with `"csi":[` entries at `TRAFFIC_RATE_HZ`.
3. Keep both serial monitors **closed** afterwards — only one process may
   own each serial port.

> **Baud rate:** ESP32-B streams at **921600** (matches `config.yaml`).
> If your board's USB-UART (CH340) does not reach it, lower
> `SERIAL_BAUD` in the receiver firmware **and** `baud_rate` in
> `config.yaml` together.

## Mac / Python setup

Requires **Python 3.10+** (developed on Python 3.14).

```bash
cd ESP32_Sensing
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # installs everything except torch
# optional but recommended for Model B:
pip install torch                       # CPU-only wheel is enough
```

Every CLI command below is run from the project root with the venv active
(`python`, or `.venv/bin/python` if you do not activate).

### Finding the Mac serial port

```bash
python python/serial_receiver.py --list
python -c "import serial.tools.list_ports as p; print([d.device for d in p.comports()])"
```

Typical macOS ports: `/dev/cu.usbmodem1101` (CP2102) or
`/dev/cu.usbserial-XXXX` (CH340). The port is configured in
`config.yaml` (`serial_port: auto` scans automatically) or passed with
`--port`.

## Usage workflow

### 0. (First run) verify raw CSI on the Mac

```bash
# see the live JSON Lines and packet rate (Ctrl+C to stop)
python python/serial_receiver.py --port /dev/cu.usbmodemXXXX --baud 921600
# optionally record raw samples losslessly to data/raw/
python python/serial_receiver.py --port /dev/cu.usbmodemXXXX --record --duration 30
```

Move a hand/person across the link and confirm you see activity in the raw
stream (e.g. via `--debug` or the debug window later).  **Do not proceed
to ML until this works** — this is the raw-CSI validation gate of the
project (stage 7 in the dev plan).

### 1. Calibration (recommended before collecting data)

```bash
python python/calibrate.py                # empty room, then 0.0/0.25/0.5/0.75/1.0
# shorter options:
python python/calibrate.py --duration 10
```

Follow the prompts. This writes `data/processed/calibration.json`
(empty-room activity + RSSI baseline and the derived movement threshold),
auto-applied by inference when
`movement.auto_threshold_from_calibration: true`.

### 2. Collect a labeled dataset

Repeat several **sessions** for each condition (2-3 sessions per value is
a good start; more is better):

```bash
# empty room
python python/dataset_collector.py --label empty --duration 60
# person standing at fixed positions
python python/dataset_collector.py --label person --position 0.0  --mode static --duration 60
python python/dataset_collector.py --label person --position 0.25 --mode static --duration 60
... 0.5, 0.75, 1.0 ...
# person moving (movement label)
python python/dataset_collector.py --label person --position 0.5 --mode moving --duration 60
```

Every invocation creates a **session** (raw file in `data/raw/`, rows
appended to `data/processed/dataset.csv`). Position semantics:
`0.0 = ESP32-A`, `0.5 = midpoint`, `1.0 = ESP32-B`.

### 3. Inspect the dataset

```bash
python python/dataset.py --stats
# columns + warnings; warns about imbalance / missing sessions
```

### 4. Train the models

```bash
python python/train.py --dataset data/processed/dataset.csv
# skip the neural network (Model B) if torch is not installed:
python python/train.py --no-nn
```

What happens: session-based train/val/test split → KNN + Random Forest
baselines (Model A) → small PyTorch MLP (Model B, optional) → best model
per task chosen on **validation** only → honest **test** metrics → artifacts
saved to `data/models/` (manifest, scaler, feature config, report).

### 5. Evaluate the saved models (optional, separate from train)

```bash
python python/evaluate.py --dataset data/processed/dataset.csv
```

### 6. Run in real time

```bash
# live GUI (room view + status + optional debug plots):
python python/main.py
# headless mode (prints state every 0.5 s):
python python/inference.py
```

The GUI shows `Connection / Presence / Movement / Position / Confidence /
CSI rate / latency` and a moving 🔵 dot along the ESP32-A .. ESP32-B line.
With no trained models it starts in *heuristic* mode (motion → presence,
position disabled) and shows a warning banner.

### 7. Continuous learning (manual, per project policy)

```
Collect new data → review/label → python python/train.py → evaluate.py →
only replaces the deployment artifacts; it never retrains automatically.
```

The safest update path is `train.py` — it writes fresh artifacts only
when the whole training run succeeds. To refuse a *worse* model, compare
`data/models/training_report.json` between runs before retraining.

## Debug view

Click **"Show debug window"** in the GUI to plot real-time CSI amplitude
(selected subcarriers), RSSI, presence/movement probabilities and the
filtered position — the primary tool for confirming that "human movement
changes CSI" before trusting any ML result.

## Automated (synthetic) pipeline tests

These run the **real** parsing/preprocessing/features/training/evaluation
code against clearly-labeled simulated CSI. They verify the toolchain, not
real-world accuracy — the synthetic profiles are built so the ML stages can
learn them, which they do almost perfectly by design.

```bash
# generate a synthetic dataset (simulated, never real measurements)
python tests/make_synthetic_dataset.py --seconds 8
# train + evaluate on it
python python/train.py --dataset data/processed/dataset_synthetic.csv --no-nn
python python/evaluate.py --dataset data/processed/dataset_synthetic.csv
# inference-with-models path (loads data/models, asserts position ≈ 0.25)
python tests/inference_model_test.py
# GUI smoke test (Qt offscreen, no hardware)
python tests/gui_smoke_test.py
```

## Performance notes (V1)

The real-time pipeline is streaming: each CSI sample goes through
`parse → preprocess → features (window/stride) → inference`, then the GUI
polls the engine state on a timer (`gui.update_hz`). Measurable quantities
are displayed live in the GUI / debug view:

* `CSI rate` — measured packets/s (target ≈ `TRAFFIC_RATE_HZ`).
* `Inference latency` — engine `process_sample` time (EMA), excludes any
  serial buffering; expect single-digit ms on a modern Mac for the 180-dim
  models.
* GUI update rate — Qt timer, configurable; independent of CSI rate.

Serial is the throughput bottleneck: 921600 baud ≈ 92 KB/s, and a 60 Hz
CSI stream needs roughly 60 × ~0.6 KB ≈ 36 KB/s — comfortably within
budget. If you see `dropped=` on the ESP32 or a stalled packet rate,
increase `baud`, raise `CSI_QUEUE_LEN`, or reduce `TRAFFIC_RATE_HZ`.

## Troubleshooting

| Problem                             | Fix                                                            |
|-------------------------------------|----------------------------------------------------------------|
| No serial port found                | install CP210x/CH340 driver; check USB System Report; use `--port` explicitly |
| Garbage/partial JSON lines          | baud mismatch; raise timeout; only one app may open the port    |
| `rx=0` while probes rise            | AP not echoing (check ESP32-A `echoed=`); channel/ssid mismatch  |
| Models missing                      | run `python python/train.py` first                              |
| Dataset missing                     | run `dataset_collector.py` first; check `config.yaml dataset_path`|
| GUI shows heuristic banner          | no `model_manifest.json` in `data/models/` — train first        |
| Position disabled                   | train again after collecting `person` sessions with `--position` |
| Ball jumps around                   | lower `smoothing_factor`; check room for other moving devices/channels |
| Presence flickers                   | raise `presence.smoothing_alpha`; move threshold; recalibrate    |
| Unreasonably good synthetic metrics | expected — see "Automated (synthetic) pipeline tests"            |

## Known limitations

This is an **experimental** system. Wi-Fi CSI sensing is affected by:

* Room geometry, walls, furniture, reflections, multipath;
* Other Wi-Fi networks/devices on the same channel;
* ESP32 antenna characteristics and hardware placement;
* Person orientation, clothing, body composition;
* Multiple people in the room (V1 assumes one);
* Temperature/humidity drift between sessions;
* Packet rate, channel choice, noise.

With two ESP32s, only limited spatial information is available, so **V1
treats position as an approximate coordinate along the ESP32-to-ESP32
link** (`0.0 = ESP32-A`, `1.0 = ESP32-B`). Do not claim exact human
coordinates unless your measurements demonstrate such accuracy.

Additional notes:

* Phase features are **off by default** — the ESP32's per-packet phase
  offset makes raw phase unusable without careful calibration
  (`config.yaml: csi.phase_features`).
* The synthetic dataset in `tests/` is **not** real data; do not use it
  for real experiments or quote its metrics as real results.
* Presence runs in *heuristic* mode (motion → person) when no trained
  model exists, trading accuracy for usability before training.

## Future improvements (V2+)

* Multiple receivers / CSI links → true 2D positioning;
* Multiple-person tracking; trajectory prediction;
* Temporal models (LSTM/TCN) over the windowed features;
* UDP streaming firmware instead of USB serial;
* Live auto-calibration and drift compensation;
* Spatial interpolation / offset calibration for room geometry.

The code is modular so these upgrades do not require rewriting: serial
transport, preprocessing, feature extraction, training, inference engine
and GUI are independent modules with documented interfaces.

## Development plan (the stages this project follows)

```
1  Identify ESP32 hardware                      ✓ (ESP32 DevKit V1)
2  Compile AP firmware                          ✓ (both cores, verified)
3  Compile receiver firmware                    ✓ (both cores, verified)
4  Verify Wi-Fi connection [on hardware]
5  Verify CSI capture [on hardware]
6  Stream CSI to Mac [on hardware]
7  Plot CSI                                   ✓ (debug window; live check on hardware)
8  Collect dataset [on hardware]
9  Train baselines [after stage 8]
10 Train neural network [after stage 8]
11 Evaluate models [after stage 8]
12 Real-time inference [after stage 8]
13 Live visualization                          ✓ (GUI smoke-tested)
14 Calibration                                ✓ (script, needs hardware)
15 Run complete experiment [needs hardware]
```

Stages marked **[on hardware]** depend on having the two boards connected;
everything else is verified in this repository (compile-verified firmware,
synthetic pipeline tests, GUI smoke test).

## Repository status

- [x] Firmware compile-verified (arduino-cli) on core 2.0.17 and core 3.3.7
- [x] Python pipeline verified end-to-end on synthetic data
- [x] GUI verified offscreen (with and without trained models)
- [x] Models load/save; inference reproduces the trained distribution
- [ ] **Requires your hardware** for real CSI, dataset collection, real
      training and the final live demo

## Licensing / third-party notices

Uses `arduino-esp32` (Espressif licensing), PyTorch / scikit-learn /
NumPy / SciPy / pandas / PyQt6 / matplotlib / PyYAML / PySerial — all
under their respective licenses. Firmware comments cite the Espressif
Wi-Fi CSI API documentation, and the CSI field mapping is documented in
`python/csi_parser.py`.