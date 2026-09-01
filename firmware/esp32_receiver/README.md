# ESP32-B: Wi-Fi Station + CSI Receiver + Serial Streamer

Target: **ESP32 DevKit V1 (classic ESP32, ESP-WROOM-32)**.

## Role

```
ESP32-A (AP)                    ESP32-B (this firmware)
   CSI_SENSOR  <------- Wi-Fi -------->  STA
                                sends UDP probe :8888 -> :8765
AP echoes probe back (HT20) -> CSI captured on EVERY received frame
                                "csi" -> one JSON Line per sample over USB serial
                                          -> Mac (921600 baud)
```

## CSI API verification (done against the actual installed cores)

The firmware was compile-verified against **both** installed Arduino cores,
using the **DOIT ESP32 DEVKIT V1** board FQBN:

| Core                  | ESP-IDF | API verified                            | Compile |
|-----------------------|---------|----------------------------------------|---------|
| `esp32:esp32` 2.0.17  | v4.4.7  | `wifi_csi_config_t`, `wifi_csi_info_t`, `wifi_csi_cb_t` | OK (55% flash) |
| `espressif:esp32` 3.3.7 | v5.5  | ditto                                   | OK (68% flash) |

Verified signatures (identical in both generations):

```c
typedef void (*wifi_csi_cb_t)(void *ctx, wifi_csi_info_t *data);
esp_err_t esp_wifi_set_csi_config(const wifi_csi_config_t *config);
esp_err_t esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx);
esp_err_t esp_wifi_set_csi(bool en);
```

`wifi_csi_info_t` provides (both cores): `rx_ctrl`, `mac[6]`,
`first_word_invalid`, `int8_t *buf`, `uint16_t len`.  (Core 3.x adds
`dmac/hdr/payload/payload_len/rx_seq`, which this firmware deliberately does
not depend on.)  The only version difference used is `dump_ack_en`
(added in ESP-IDF 5.x, guarded with `#if ESP_IDF_VERSION >= 5.0.0`).

The Arduino core's `WiFi` wrapper does **not** expose CSI, so the firmware
calls the raw ESP-IDF functions directly (they come with the Arduino core's
own `esp_wifi.h`; no extra library is needed).

## CSI buffer layout (20 MHz), per Espressif's CSI guide

CSI consists of (real, imag) **int8** pairs, one per subcarrier slot:
128 bytes = 64 slots at 20 MHz.  Subcarriers **0..26** and **32..58**
carry usable channel estimates; **27..31** and **59..63** are DC/guard/pilot
and are discarded by the Mac-side preprocessor.  STBC frames produce
double-length buffers; the firmware streams `info->len` verbatim and the
Mac drops unexpected lengths (or routes them to `*_irregular.csv` during
raw recording).

## Callback timing note

The CSI callback runs in the Wi-Fi task context and must be fast; the
firmware snapshots the packet into a FreeRTOS queue (depth 64) and the main
loop formats/emits it.  If the queue fills (`dropped=` rises / Mac packet
rate stalls), lower `TRAFFIC_RATE_HZ` or raise `CSI_QUEUE_LEN`/`SERIAL_BAUD`.

## Configuration (top of `esp32_receiver.ino`)

| Macro                | Meaning                                    | Default      |
|----------------------|--------------------------------------------|--------------|
| `WIFI_SSID`         | SSID of ESP32-A's AP                    | `CSI_SENSOR` |
| `WIFI_PASSWORD`     | AP password                                | `change_this`|
| `TRAFFIC_RATE_HZ`   | probe rate == expected CSI rate            | 60           |
| `SERIAL_BAUD`       | USB serial baud (`<=921600` on CP2102;  | 921600      |
|                      | CH340 allows 2000000)                   |              |
| `OUTPUT_MODE`        | `OUTPUT_SERIAL_JSON` / `OUTPUT_SERIAL_CSV` / `OUTPUT_NONE` | JSON |
| `CSI_MAX_BYTES`      | fixed snapshot buffer                        | 256        |
| `CSI_QUEUE_LEN`     | callback→loop queue depth                  | 64          |
| `RECONNECT_PERIOD_MS` | retry interval when the link drops       | 5000        |

## JSON Lines data format

One JSON object per line; every field maps to a documented ESP-IDF
member (full table in `python/csi_parser.py`).  Example:

```json
{"timestamp":1720000000000,"seq":42,"rssi":-47,"channel":6,
 "secondary_channel":0,"sig_mode":1,"mcs":5,"cwb":0,"rate":0,
 "aggregation":0,"stbc":0,"fec_coding":0,"sgi":0,"noise_floor":-90,
 "ampdu_cnt":1,"sig_len":64,"rx_state":0,"ant":0,
 "timestamp_wifi":217201234,"mac":"AABBCCDDEEFF","first_word_invalid":0,
 "csi_len":128,"csi":[12,-4,8,15,-3,7, ...]}
```

* `timestamp` — `esp_timer_get_time()` us at callback entry.
* `rssi`/`noise_floor`/... — copies of `rx_ctrl.*` members (see parser).
* `csi` — all raw int8 (real,imag) pairs, exactly as delivered.
* `first_word_invalid` — documented hardware limitation where the first
  4 CSI bytes may be invalid (samples with this flag are dropped by the
  Mac preprocessor by default).
* Lines beginning with `#` are human status (`#STAT ...`), ignored by
  parsers.

## Compiling with the Arduino CLI

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
arduino-cli upload  -p /dev/cu.usbmodemXXXX --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
```

## Verify CSI output without the Mac

```bash
python serial_receiver.py --list                 # find the port
python serial_receiver.py --port /dev/cu.usbmodemXXXX --baud 921600
# or use a terminal: screen /dev/cu.usbmodemXXXX 921600
```

Expected: a fresh JSON line per probe whose `"csi":[` list is present;
the `rx=` counter rises at ~`TRAFFIC_RATE_HZ` and `dropped=0`.  To record
raw samples for offline inspection:

```bash
python serial_receiver.py --record --duration 30
```

## Troubleshooting

| Symptom                  | Fix                                                        |
|--------------------------|------------------------------------------------------------|
| No serial output         | baud mismatch (`SERIAL_BAUD` vs monitor): use 921600; replug USB |
| `ERROR: connect timeout` | is ESP32-A powered? same SSID/password/channel on both?    |
| `rx=0` while probes rise | CSI captures received data frames: confirm the AP echoes probes (see esp32_ap README) |
| `dropped=` grows         | raise `SERIAL_BAUD` / `CSI_QUEUE_LEN`, lower `TRAFFIC_RATE_HZ` |
| `rx_errors=` grows       | environmental interference; try another channel              |
| `reconnects=` grows      | move boards closer; check AP restart loop                    |
| macOS shows no port      | install CP210x/CH340 driver; check System Report → USB        |