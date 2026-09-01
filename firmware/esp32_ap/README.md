# ESP32-A: Access Point + Traffic Generator

Target: **ESP32 DevKit V1 (classic ESP32, ESP-WROOM-32)**.

## Role

```
ESP32-A (this firmware)                  ESP32-B
  AP "CSI_SENSOR"  <---- Wi-Fi ---->  STA + CSI receiver
    UDP echo :8765                       sends probes
```

ESP32-A does **not** capture CSI. It:

1. Starts a dedicated Wi-Fi AP (`CSI_SENSOR`) on a fixed channel.
2. Runs a small UDP **echo** service on port `8765`. ESP32-B sends probe
   packets to the AP; every probe is echoed back as a **unicast data frame
   (HT20)**, which is exactly what ESP32-B captures CSI from.
3. Optionally broadcasts low-rate heartbeats so the channel always carries
   some AP-originated (legacy-rate) traffic even before a station joins.
4. Reports startup info and periodic status over USB serial; restarts the
   AP automatically if Wi-Fi reports a stop event.

## Why UDP echo instead of raw broadcast traffic?

The ESP32's CSI is captured *by the receiver* from any received packet.
Unicast HT data frames SAP→STA give the cleanest, most deterministic CSI
(HT-LTF based, fixed 128-byte buffer at 20 MHz), and the echo makes the
frame rate equal to the receiver's probe rate. This keeps traffic minimal
and predictable.

## Wiring

You only need to power the board over USB. No extra wiring is required.

## Configuration (top of `esp32_ap.ino`)

| Macro                | Meaning                                          | Default       |
|----------------------|--------------------------------------------------|---------------|
| `WIFI_SSID`          | AP network name                                  | `CSI_SENSOR`  |
| `WIFI_PASSWORD`     | AP password (`>= 8` chars for WPA2, `""`=open) | `change_this`|
| `WIFI_CHANNEL`      | fixed Wi-Fi channel                            | 6            |
| `UDP_ECHO_PORT`     | probe/echo port (must match receiver)          | 8765         |
| `HEARTBEAT_RATE_HZ` | AP-side broadcast rate (0 disables)            | 10           |
| `SERIAL_BAUD`       | USB serial baud                                | 921600       |

The AP bandwidth is pinned to **HT20** so CSI buffers always have the
expected 128-byte length at the receiver.

## Compiling with the Arduino CLI

```bash
# recommended core for this project (any ESP32 DevKit V1 FQBN)
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
# or with the upstream 3.x core
arduino-cli compile --fqbn espressif:esp32:esp32doit-devkit-v1 firmware/esp32_ap
```

`arduino-cli upload -p /dev/cu.usbmodemXXXX --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap`

See the main README for Arduino IDE instructions.

## Serial output / verification

After power-on open a terminal at 921600 baud:

```
[AP] ===============================================
[AP]  ESP32 Wi-Fi CSI sensing - ESP32-A access point
[AP]  firmware version: 1.0.0
[AP] ===============================================
[AP] ---------------- startup info ----------------
[AP] firmware version : 1.0.0
[AP] AP MAC address   : 30:AE:A4:...:XX
[AP] SSID             : CSI_SENSOR
[AP] Wi-Fi channel    : 6
[AP] AP IP address    : 192.168.4.1
[AP] UDP echo port    : 8765
[AP] heartbeat rate   : 10 Hz (0 = off)
[AP] ------------------------------------------------
```

Periodic `[AP] status:` lines show `stations=` (should become 1 once
ESP32-B connects), `echoed=` (probes replied) and `wifi_errors=`.

**Success checks**

* `stations=1` appears after ESP32-B connects.
* `echoed` increases when ESP32-B probes are being sent.
* No `ERROR` lines; on any `AP stopped` event the firmware restarts the AP
  automatically.

## Troubleshooting

| Symptom                     | Likely cause / fix                                    |
|-----------------------------|-------------------------------------------------------|
| AP not visible              | `WIFI_PASSWORD` too short → firmware starts an open AP; or channel congested (try channel 1 or 11) |
| No echoed counter           | receiver uses a different `UDP_ECHO_PORT`             |
| `softAP failed`             | check serial `ERROR` line; power-cycle the board      |
| `wifi_errors` climbing      | Wi-Fi stack restarted the AP; this is self-recovering |
| Board not detected          | install the proper USB-UART driver (CP210x/CH340)     |