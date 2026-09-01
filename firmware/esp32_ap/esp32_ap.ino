/*
 * ESP32 Wi-Fi CSI Human Position Sensing System
 * ---------------------------------------------
 * ESP32-A: Wi-Fi Access Point + controlled traffic generator.
 *
 * Role in the system:
 *   - Broadcasts a dedicated Wi-Fi AP ("CSI_SENSOR") on a fixed channel.
 *   - Runs a small UDP echo service: the CSI receiver (ESP32-B) sends UDP
 *     probe packets to this AP; every probe is echoed back as a unicast
 *     data frame. Those unicast reply frames are what ESP32-B captures
 *     CSI from (CSI is a receiver-side feature of the ESP32).
 *   - Optionally emits low-rate UDP heartbeats (broadcast) so the channel
 *     always carries some traffic from the AP side as well.
 *
 * Design rules:
 *   - No CSI is captured on this board; CSI capture happens on ESP32-B.
 *   - No unnecessary traffic: probe echoes + optional heartbeats + the
 *     normal beacon frames the AP must send anyway.
 *   - Recovers from Wi-Fi errors: events are logged and a stopped AP is
 *     restarted automatically.
 *
 * API compatibility (verified against the installed cores):
 *   - arduino-esp32 core 2.0.17 (ESP-IDF v4.4.7)
 *   - arduino-esp32 core 3.3.7  (ESP-IDF v5.5)
 * Target board: ESP32 DevKit V1 (DOIT ESP32 DEVKIT V1 / "ESP32 Dev Module").
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

// ------------------------- User configuration -------------------------
// Adjust these before flashing. The receiver (ESP32-B) must use the same
// SSID / password and must be able to reach this AP's channel.
#define WIFI_SSID            "CSI_SENSOR"
#define WIFI_PASSWORD        "change_this"    // >= 8 chars (WPA2) or "" for an open AP
#define WIFI_CHANNEL         6                // fixed experiment channel
#define AP_IPV4              192, 168, 4, 1   // softAP IP (default range)
#define UDP_ECHO_PORT        8765             // probe/echo service port
#define HEARTBEAT_RATE_HZ    10               // AP-side broadcast rate, 0 disables
#define HEARTBEAT_PAYLOAD    "CSI_AP_HEARTBEAT"
#define STATUS_PERIOD_MS     10000UL          // periodic status line on Serial
#define SERIAL_BAUD          921600
#define FIRMWARE_VERSION     "1.0.0"
// ----------------------------------------------------------------------

static WiFiUDP   s_udp;
static bool      s_ap_running  = false;
static uint32_t  s_echoed      = 0;   // UDP probes echoed back to the sender
static uint32_t  s_heartbeats  = 0;   // broadcast heartbeats sent
static uint32_t  s_wifi_errors = 0;   // AP stop / UDP errors observed
static uint32_t  s_last_status_ms = 0;
static uint32_t  s_last_beat_ms   = 0;
static uint32_t  s_last_restart_ms = 0;

static uint8_t   s_rx_buffer[512];

static const IPAddress s_ap_ip(AP_IPV4);
static const IPAddress s_ap_mask(255, 255, 255, 0);


// ---------------------------------------------------------------------
// Wi-Fi event logging / error tracking
// ---------------------------------------------------------------------
static void onWifiEvent(arduino_event_id_t event) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_AP_START:
      s_ap_running = true;
      Serial.println("[AP] event: AP started");
      break;
    case ARDUINO_EVENT_WIFI_AP_STOP:
      s_ap_running = false;
      s_wifi_errors++;
      Serial.println("[AP] event: AP stopped (will restart)");
      break;
    case ARDUINO_EVENT_WIFI_AP_STACONNECTED:
      Serial.println("[AP] event: station connected");
      break;
    case ARDUINO_EVENT_WIFI_AP_STADISCONNECTED:
      Serial.println("[AP] event: station disconnected");
      break;
    default:
      break;  // ignore unrelated events to keep the log clean
  }
}

// ---------------------------------------------------------------------
// AP lifecycle
// ---------------------------------------------------------------------
static void startAccessPoint() {
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAPConfig(s_ap_ip, s_ap_ip, s_ap_mask)) {
    Serial.println("[AP] ERROR: softAPConfig failed");
  }
  bool ok;
  if (strlen(WIFI_PASSWORD) >= 8) {
    ok = WiFi.softAP(WIFI_SSID, WIFI_PASSWORD, WIFI_CHANNEL);
  } else {
    Serial.println("[AP] password empty/short -> starting an OPEN AP");
    ok = WiFi.softAP(WIFI_SSID, nullptr, WIFI_CHANNEL);
  }
  if (!ok) {
    Serial.println("[AP] ERROR: softAP failed to start");
    s_wifi_errors++;
    return;
  }
  // Pin the AP bandwidth to HT20 so the receiver always sees a
  // deterministic CSI length (128 bytes for 20 MHz).
  esp_err_t err = esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20);
  if (err != ESP_OK) {
    Serial.printf("[AP] WARNING: esp_wifi_set_bandwidth -> %d\n", (int)err);
  }
  s_ap_running = true;
}

static void restartAccessPointIfNeeded() {
  // Only attempt a restart if an AP_STOP event occurred; back off so a
  // persistent failure does not turn into a tight restart loop.
  if (s_ap_running) {
    return;
  }
  const uint32_t now = millis();
  if (now - s_last_restart_ms < 5000UL) {
    return;
  }
  s_last_restart_ms = now;
  Serial.println("[AP] recovery: restarting softAP");
  startAccessPoint();
}


// ---------------------------------------------------------------------
// Traffic generation
// ---------------------------------------------------------------------
// Echo every received UDP datagram back to its sender as a unicast data
// frame. These unicast replies are the packets the receiver captures CSI
// from (HT data frames ESP32-A -> ESP32-B).
static void pumpUdpEcho() {
  const int packet_size = s_udp.parsePacket();
  if (packet_size < 0) {
    s_wifi_errors++;
    return;
  }
  if (packet_size == 0) {
    return;
  }
  int len = packet_size;
  if (len > (int)sizeof(s_rx_buffer)) {
    len = (int)sizeof(s_rx_buffer);
  }
  const int got = s_udp.read(s_rx_buffer, len);
  if (got <= 0) {
    return;
  }
  s_udp.beginPacket(s_udp.remoteIP(), s_udp.remotePort());
  s_udp.write(s_rx_buffer, got);
  if (s_udp.endPacket() != 1) {
    s_wifi_errors++;
  }
  s_echoed++;
}

// Low-rate broadcast heartbeats keep some AP-originated traffic on the
// channel (legacy-rate frames -> L-LTF based CSI on the receiver) even
// when no station is registered yet. Set HEARTBEAT_RATE_HZ to 0 to disable.
static void pumpHeartbeat() {
#if HEARTBEAT_RATE_HZ > 0
  if (HEARTBEAT_RATE_HZ > 1000) {
    return;  // guard against nonsensical configuration
  }
  const uint32_t period_ms = 1000UL / (uint32_t)HEARTBEAT_RATE_HZ;
  const uint32_t now = millis();
  if (now - s_last_beat_ms < period_ms) {
    return;
  }
  s_last_beat_ms = now;
  IPAddress bcast(s_ap_ip[0], s_ap_ip[1], s_ap_ip[2], 255);
  s_udp.beginPacket(bcast, UDP_ECHO_PORT);
  s_udp.write((const uint8_t *)HEARTBEAT_PAYLOAD, strlen(HEARTBEAT_PAYLOAD));
  if (s_udp.endPacket() != 1) {
    s_wifi_errors++;
  }
  s_heartbeats++;
#endif
}

static void pumpStatus() {
  const uint32_t now = millis();
  if (now - s_last_status_ms < STATUS_PERIOD_MS) {
    return;
  }
  s_last_status_ms = now;
  Serial.printf(
      "[AP] status: uptime_s=%lu stations=%u echoed=%lu heartbeats=%lu "
      "wifi_errors=%lu heap=%u\n",
      (unsigned long)(now / 1000UL),
      (unsigned)WiFi.softAPgetStationNum(),
      (unsigned long)s_echoed,
      (unsigned long)s_heartbeats,
      (unsigned long)s_wifi_errors,
      (unsigned)ESP.getFreeHeap());
}

static void printStartupInfo() {
  Serial.println("[AP] ---------------- startup info ----------------");
  Serial.printf("[AP] firmware version : %s\n", FIRMWARE_VERSION);
  Serial.printf("[AP] AP MAC address   : %s\n", WiFi.softAPmacAddress().c_str());
  Serial.printf("[AP] SSID             : %s\n", WIFI_SSID);
  Serial.printf("[AP] Wi-Fi channel    : %d\n", WiFi.channel());
  Serial.printf("[AP] AP IP address    : %s\n", WiFi.softAPIP().toString().c_str());
  Serial.printf("[AP] UDP echo port    : %u\n", (unsigned)UDP_ECHO_PORT);
  Serial.printf("[AP] heartbeat rate   : %u Hz (0 = off)\n", (unsigned)HEARTBEAT_RATE_HZ);
  Serial.println("[AP] ------------------------------------------------");
}

// ---------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  Serial.println();
  Serial.println("[AP] ===============================================");
  Serial.println("[AP]  ESP32 Wi-Fi CSI sensing - ESP32-A access point");
  Serial.printf("[AP]  firmware version: %s\n", FIRMWARE_VERSION);
  Serial.println("[AP] ===============================================");

  WiFi.onEvent(onWifiEvent);
  startAccessPoint();

  if (s_udp.begin(UDP_ECHO_PORT) != 1) {
    Serial.println("[AP] ERROR: UDP socket could not be opened");
    s_wifi_errors++;
  }

  printStartupInfo();
}

void loop() {
  pumpUdpEcho();
  pumpHeartbeat();
  pumpStatus();
  restartAccessPointIfNeeded();
  delay(2);  // yield to the WiFi/UDP stack
}
