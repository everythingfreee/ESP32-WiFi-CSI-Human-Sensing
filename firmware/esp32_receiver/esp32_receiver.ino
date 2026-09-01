/*
 * ESP32 Wi-Fi CSI Human Position Sensing System
 * ---------------------------------------------
 * ESP32-B: Wi-Fi station + CSI receiver / serial streamer.
 * (Corrected version – no early Wi-Fi driver calls)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "esp_timer.h"
#include "esp_idf_version.h"
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// ------------------------- User configuration -------------------------
#define WIFI_SSID           "CSI_SENSOR"
#define WIFI_PASSWORD       "change_this"
#define WIFI_CHANNEL        6               // informational only
#define TRAFFIC_RATE_HZ     60              // probe rate
#define TRAFFIC_PAYLOAD     "CSI_PROBE"
#define SERIAL_BAUD         921600          // use 115200 for debugging (later increase)
#define CONNECT_TIMEOUT_MS  30000UL
#define RECONNECT_PERIOD_MS 5000UL
#define RESTART_AFTER_MS    120000UL
#define STATUS_PERIOD_MS    5000UL
#define CSI_MAX_BYTES       256
#define CSI_QUEUE_LEN       64
#define OUTPUT_MODE         OUTPUT_SERIAL_JSON
#define FIRMWARE_VERSION    "1.0.0"
// ----------------------------------------------------------------------

enum OutputMode {
  OUTPUT_SERIAL_JSON = 0,
  OUTPUT_SERIAL_CSV  = 1,
  OUTPUT_NONE        = 2
};

typedef struct {
  int64_t  ts_us;
  uint32_t seq;
  int8_t   csi[CSI_MAX_BYTES];
  uint16_t csi_len;
  int8_t   rssi;
  int8_t   noise_floor;
  uint8_t  channel;
  uint8_t  secondary_channel;
  uint8_t  sig_mode;
  uint8_t  mcs;
  uint8_t  cwb;
  uint8_t  rate;
  uint8_t  aggregation;
  uint8_t  stbc;
  uint8_t  fec_coding;
  uint8_t  sgi;
  uint8_t  ampdu_cnt;
  uint16_t sig_len;
  uint8_t  rx_state;
  uint8_t  ant;
  uint32_t ts_wifi_us;
  uint8_t  mac[6];
  uint8_t  first_word_invalid;
} CsiPacket;

static QueueHandle_t s_csi_queue   = nullptr;
static WiFiUDP       s_udp;
static uint32_t      s_rx_seq      = 0;
static uint32_t      s_dropped     = 0;
static uint32_t      s_cb_errors   = 0;
static uint32_t      s_rx_errors   = 0;
static uint32_t      s_probes_sent = 0;
static uint32_t      s_reconnects  = 0;
static uint32_t      s_last_probe_ms     = 0;
static uint32_t      s_last_status_ms    = 0;
static uint32_t      s_last_conn_ms      = 0;
static uint32_t      s_last_reconnect_ms = 0;
static uint16_t      s_probe_id    = 0;
static bool          s_csi_enabled = false;   // flag to avoid re-configuring

// ---------------------------------------------------------------------
// CSI callback (must be fast)
// ---------------------------------------------------------------------
static void csi_rx_callback(void *ctx, wifi_csi_info_t *data) {
  (void)ctx;
  if (data == nullptr || data->buf == nullptr) {
    s_cb_errors++;
    return;
  }
  if (data->rx_ctrl.rx_state != 0) {
    s_rx_errors++;
    return;
  }
  if (!s_csi_queue) return;

  CsiPacket pkt;
  memset(&pkt, 0, sizeof(pkt));
  pkt.ts_us  = esp_timer_get_time();
  pkt.seq    = s_rx_seq++;
  pkt.rssi   = (int8_t)data->rx_ctrl.rssi;
  pkt.noise_floor = (int8_t)data->rx_ctrl.noise_floor;
  pkt.channel           = (uint8_t)data->rx_ctrl.channel;
  pkt.secondary_channel = (uint8_t)data->rx_ctrl.secondary_channel;
  pkt.sig_mode    = (uint8_t)data->rx_ctrl.sig_mode;
  pkt.mcs         = (uint8_t)data->rx_ctrl.mcs;
  pkt.cwb         = (uint8_t)data->rx_ctrl.cwb;
  pkt.rate        = (uint8_t)data->rx_ctrl.rate;
  pkt.aggregation = (uint8_t)data->rx_ctrl.aggregation;
  pkt.stbc        = (uint8_t)data->rx_ctrl.stbc;
  pkt.fec_coding  = (uint8_t)data->rx_ctrl.fec_coding;
  pkt.sgi         = (uint8_t)data->rx_ctrl.sgi;
  pkt.ampdu_cnt   = (uint8_t)data->rx_ctrl.ampdu_cnt;
  pkt.sig_len     = (uint16_t)data->rx_ctrl.sig_len;
  pkt.rx_state    = (uint8_t)data->rx_ctrl.rx_state;
  pkt.ant         = (uint8_t)data->rx_ctrl.ant;
  pkt.ts_wifi_us  = (uint32_t)data->rx_ctrl.timestamp;
  memcpy(pkt.mac, data->mac, 6);
  pkt.first_word_invalid = data->first_word_invalid ? 1 : 0;
  uint16_t n = data->len;
  if (n > CSI_MAX_BYTES) n = CSI_MAX_BYTES;
  pkt.csi_len = n;
  memcpy(pkt.csi, data->buf, n);

  if (xQueueSend(s_csi_queue, &pkt, 0) != pdTRUE) {
    s_dropped++;
  }
}

// ---------------------------------------------------------------------
// CSI configuration – safe to call only after WiFi is connected
// ---------------------------------------------------------------------
static bool configureCsi() {
  wifi_csi_config_t csi_config;
  memset(&csi_config, 0, sizeof(csi_config));
  csi_config.lltf_en           = true;
  csi_config.htltf_en          = true;
  csi_config.stbc_htltf2_en    = true;
  csi_config.ltf_merge_en      = true;
  csi_config.channel_filter_en = false;
  csi_config.manu_scale        = false;
  csi_config.shift             = 0;
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
  csi_config.dump_ack_en       = false;
#endif

  esp_err_t err = esp_wifi_set_csi_config(&csi_config);
  if (err != ESP_OK) {
    Serial.printf("[RX] ERROR: esp_wifi_set_csi_config -> %d\n", (int)err);
    return false;
  }
  err = esp_wifi_set_csi_rx_cb(&csi_rx_callback, nullptr);
  if (err != ESP_OK) {
    Serial.printf("[RX] ERROR: esp_wifi_set_csi_rx_cb -> %d\n", (int)err);
    return false;
  }
  err = esp_wifi_set_csi(true);
  if (err != ESP_OK) {
    Serial.printf("[RX] ERROR: esp_wifi_set_csi(true) -> %d\n", (int)err);
    return false;
  }
  s_csi_enabled = true;
  return true;
}

// ---------------------------------------------------------------------
// Connection management (no early Wi-Fi driver calls!)
// ---------------------------------------------------------------------
static void printConnectionInfo() {
  Serial.println("[RX] ---------------- connection info ----------------");
  Serial.printf("[RX] firmware version : %s\n", FIRMWARE_VERSION);
  Serial.printf("[RX] station MAC      : %s\n", WiFi.macAddress().c_str());
  Serial.printf("[RX] SSID             : %s\n", WIFI_SSID);
  Serial.printf("[RX] Wi-Fi channel    : %d\n", WiFi.channel());
  Serial.printf("[RX] IP address       : %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("[RX] gateway (AP)     : %s\n", WiFi.gatewayIP().toString().c_str());
  Serial.printf("[RX] probe rate       : %u Hz\n", (unsigned)TRAFFIC_RATE_HZ);
  Serial.printf("[RX] output mode      : %s\n",
                OUTPUT_MODE == OUTPUT_SERIAL_JSON ? "SERIAL_JSON" :
                OUTPUT_MODE == OUTPUT_SERIAL_CSV  ? "SERIAL_CSV"  : "NONE");
  Serial.println("[RX] -------------------------------------------------");
}

static bool connectToAp() {
  Serial.printf("[RX] connecting to SSID '%s'...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  // No esp_wifi_set_protocol/bandwidth here – let the stack handle it
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > CONNECT_TIMEOUT_MS) {
      Serial.println("[RX] ERROR: connection timeout");
      return false;
    }
    delay(100);
    // Optional status print
    // Serial.printf("[RX] Wi-Fi status: %d\n", WiFi.status());
  }
  s_last_conn_ms = millis();
  printConnectionInfo();
  return true;
}

static void maintainConnection() {
  if (WiFi.status() == WL_CONNECTED) {
    s_last_conn_ms = millis();
    // If we are connected but CSI not enabled, enable it now
    if (!s_csi_enabled) {
      Serial.println("[RX] Re‑enabling CSI after reconnect...");
      if (configureCsi()) {
        Serial.println("[RX] CSI re‑enabled.");
      } else {
        Serial.println("[RX] ERROR: CSI re‑configuration failed.");
      }
    }
    return;
  }
  const uint32_t now = millis();
  if (now - s_last_reconnect_ms >= RECONNECT_PERIOD_MS) {
    s_last_reconnect_ms = now;
    s_reconnects++;
    Serial.println("[RX] WARNING: link down, attempting reconnect");
    WiFi.reconnect();
  }
  if (now - s_last_conn_ms > RESTART_AFTER_MS) {
    Serial.println("[RX] ERROR: offline too long, restarting");
    delay(100);
    ESP.restart();
  }
}

// ---------------------------------------------------------------------
// Traffic generation (UDP probes) – only if connected
// ---------------------------------------------------------------------
static void pumpTraffic() {
  if (WiFi.status() != WL_CONNECTED || TRAFFIC_RATE_HZ == 0) return;
  const uint32_t period_ms = 1000UL / (uint32_t)TRAFFIC_RATE_HZ;
  const uint32_t now = millis();
  if (now - s_last_probe_ms < period_ms) return;
  s_last_probe_ms = now;
  char payload[32];
  snprintf(payload, sizeof(payload), "%s|%u", TRAFFIC_PAYLOAD, (unsigned)(s_probe_id++));
  s_udp.beginPacket(WiFi.gatewayIP(), 8765);
  s_udp.write((const uint8_t *)payload, strlen(payload));
  if (s_udp.endPacket() != 1) {
    s_cb_errors++;
  }
  s_probes_sent++;
}

static void drainUdp() {
  while (s_udp.parsePacket() > 0) {
    s_udp.flush();
  }
}

// ---------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------
static size_t format_json(const CsiPacket &p, char *out, const size_t cap) {
  // ... (keep your original formatting code, unchanged) ...
  // I'm copying the original for completeness; you can keep yours.
  size_t n = (size_t)snprintf(
      out, cap,
      "{\"timestamp\":%lld,\"seq\":%lu,\"rssi\":%d,\"channel\":%u,"
      "\"secondary_channel\":%u,\"sig_mode\":%u,\"mcs\":%u,\"cwb\":%u,"
      "\"rate\":%u,\"aggregation\":%u,\"stbc\":%u,\"fec_coding\":%u,"
      "\"sgi\":%u,\"noise_floor\":%d,\"ampdu_cnt\":%u,\"sig_len\":%u,"
      "\"rx_state\":%u,\"ant\":%u,\"timestamp_wifi\":%lu,"
      "\"mac\":\"%02X%02X%02X%02X%02X%02X\",\"first_word_invalid\":%u,"
      "\"csi_len\":%u,\"csi\":[",
      (long long)p.ts_us, (unsigned long)p.seq, (int)p.rssi,
      (unsigned)p.channel, (unsigned)p.secondary_channel,
      (unsigned)p.sig_mode, (unsigned)p.mcs, (unsigned)p.cwb,
      (unsigned)p.rate, (unsigned)p.aggregation, (unsigned)p.stbc,
      (unsigned)p.fec_coding, (unsigned)p.sgi, (int)p.noise_floor,
      (unsigned)p.ampdu_cnt, (unsigned)p.sig_len, (unsigned)p.rx_state,
      (unsigned)p.ant, (unsigned long)p.ts_wifi_us,
      p.mac[0], p.mac[1], p.mac[2], p.mac[3], p.mac[4], p.mac[5],
      (unsigned)p.first_word_invalid, (unsigned)p.csi_len);
  for (uint16_t i = 0; i < p.csi_len && n + 16 < cap; i++) {
    n += (size_t)snprintf(out + n, cap - n, "%s%d",
                          (i == 0) ? "" : ",", (int)p.csi[i]);
  }
  if (n + 3 < cap) {
    out[n++] = ']';
    out[n++] = '}';
    out[n++] = '\n';
    out[n]   = '\0';
  }
  return n;
}

static size_t format_csv(const CsiPacket &p, char *out, const size_t cap) {
  // ... (keep your CSV format) ...
  size_t n = (size_t)snprintf(
      out, cap,
      "%lld,%lu,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%d,%u,%u,%u,%u,%lu,"
      "%02X%02X%02X%02X%02X%02X,%u,%u",
      (long long)p.ts_us, (unsigned long)p.seq, (int)p.rssi,
      (unsigned)p.channel, (unsigned)p.secondary_channel,
      (unsigned)p.sig_mode, (unsigned)p.mcs, (unsigned)p.cwb,
      (unsigned)p.rate, (unsigned)p.aggregation, (unsigned)p.stbc,
      (unsigned)p.fec_coding, (unsigned)p.sgi, (int)p.noise_floor,
      (unsigned)p.ampdu_cnt, (unsigned)p.sig_len, (unsigned)p.rx_state,
      (unsigned)p.ant, (unsigned long)p.ts_wifi_us,
      p.mac[0], p.mac[1], p.mac[2], p.mac[3], p.mac[4], p.mac[5],
      (unsigned)p.first_word_invalid, (unsigned)p.csi_len);
  for (uint16_t i = 0; i < p.csi_len && n + 16 < cap; i++) {
    n += (size_t)snprintf(out + n, cap - n, ",%d", (int)p.csi[i]);
  }
  if (n + 1 < cap) {
    out[n++] = '\n';
    out[n]   = '\0';
  }
  return n;
}

static void emit_packet(const CsiPacket &p) {
  static char line_buffer[4096];
  size_t n = 0;

  // Emit the full JSON object in the exact shape the Python parser expects:
  //   #CSI {...,"csi":[...]}\n
  n = (size_t)snprintf(
      line_buffer, sizeof(line_buffer),
      "#CSI {\"timestamp\":%lld,\"seq\":%lu,\"rssi\":%d,"
      "\"channel\":%u,\"secondary_channel\":%u,\"sig_mode\":%u,"
      "\"mcs\":%u,\"cwb\":%u,\"rate\":%u,\"aggregation\":%u,"
      "\"stbc\":%u,\"fec_coding\":%u,\"sgi\":%u,"
      "\"noise_floor\":%d,\"ampdu_cnt\":%u,\"sig_len\":%u,"
      "\"rx_state\":%u,\"ant\":%u,\"timestamp_wifi\":%lu,"
      "\"mac\":\"%02X%02X%02X%02X%02X%02X\",\"first_word_invalid\":%u,"
      "\"csi_len\":%u,\"csi\":[",
      (long long)p.ts_us, (unsigned long)p.seq, (int)p.rssi,
      (unsigned)p.channel, (unsigned)p.secondary_channel,
      (unsigned)p.sig_mode, (unsigned)p.mcs, (unsigned)p.cwb,
      (unsigned)p.rate, (unsigned)p.aggregation, (unsigned)p.stbc,
      (unsigned)p.fec_coding, (unsigned)p.sgi, (int)p.noise_floor,
      (unsigned)p.ampdu_cnt, (unsigned)p.sig_len, (unsigned)p.rx_state,
      (unsigned)p.ant, (unsigned long)p.ts_wifi_us,
      p.mac[0], p.mac[1], p.mac[2], p.mac[3], p.mac[4], p.mac[5],
      (unsigned)p.first_word_invalid, (unsigned)p.csi_len);

  for (uint16_t i = 0; i < p.csi_len && n + 16 < sizeof(line_buffer); i++) {
    n += (size_t)snprintf(line_buffer + n, sizeof(line_buffer) - n, "%s%d",
                          (i == 0) ? "" : ",", (int)p.csi[i]);
  }

  if (n + 4 < sizeof(line_buffer)) {
    line_buffer[n++] = ']';
    line_buffer[n++] = '}';
    line_buffer[n++] = '\n';
    line_buffer[n] = '\0';
  } else {
    // If the buffer is too small, avoid emitting a partial JSON packet.
    return;
  }

  Serial.write((const uint8_t *)line_buffer, n);
  Serial.flush();
}
static void pumpStatus() {
  const uint32_t now = millis();
  if (now - s_last_status_ms < STATUS_PERIOD_MS) return;
  s_last_status_ms = now;
  Serial.printf(
      "#STAT uptime_s=%lu rx=%lu dropped=%lu rx_errors=%lu cb_errors=%lu "
      "probes=%lu reconnects=%lu rssi=%d heap=%u csi=%d\n",
      (unsigned long)(now / 1000UL), (unsigned long)s_rx_seq,
      (unsigned long)s_dropped, (unsigned long)s_rx_errors,
      (unsigned long)s_cb_errors, (unsigned long)s_probes_sent,
      (unsigned long)s_reconnects,
      WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0,
      (unsigned)ESP.getFreeHeap(), (int)s_csi_enabled);
}

// ---------------------------------------------------------------------
// Arduino setup / loop
// ---------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(3000);  // allow time to open Serial Monitor
  Serial.println();
  Serial.println("[RX] ===============================================");
  Serial.println("[RX]  ESP32 Wi-Fi CSI sensing - ESP32-B CSI receiver");
  Serial.printf("[RX]  firmware version: %s\n", FIRMWARE_VERSION);
  Serial.println("[RX] ===============================================");

  s_csi_queue = xQueueCreate(CSI_QUEUE_LEN, sizeof(CsiPacket));
  if (s_csi_queue == nullptr) {
    Serial.println("[RX] ERROR: could not create CSI queue");
  }

  // Connect to AP first – this initialises the whole stack
  if (!connectToAp()) {
    Serial.println("[RX] will keep retrying in loop()...");
  } else {
    // Now open UDP socket (safe because lwIP is ready)
    if (s_udp.begin(8888) != 1) {
      Serial.println("[RX] WARNING: UDP socket could not be opened");
    }
    // Configure and enable CSI
    if (!configureCsi()) {
      Serial.println("[RX] ERROR: CSI configuration failed, stopping");
      while (true) delay(1000);
    }
    Serial.println("[RX] CSI capture enabled");
  }
}

void loop() {
  maintainConnection();   // handles reconnects and re‑enables CSI if needed
  pumpTraffic();
  drainUdp();

  if (s_csi_queue != nullptr) {
    CsiPacket pkt;
    while (xQueueReceive(s_csi_queue, &pkt, 0) == pdTRUE) {
      static uint32_t debug_count = 0;
      debug_count++;
      if (debug_count % 50 == 0) {
          Serial.printf("#DBG dequeue count=%u\n", debug_count);
      }
      emit_packet(pkt);
    }
  }

  pumpStatus();
  delay(2);
}