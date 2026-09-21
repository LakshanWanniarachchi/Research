/*
 * ECG heart-attack screening  -  ESP32 + AD8232 firmware
 * ======================================================
 *
 *   AD8232 sensor -> ESP32 -> [ Flask server -> 1D CNN ] -> dashboard
 *   ^^^^^^^^^^^^^^^^^^^^^^^
 *   this sketch
 *
 * STATUS: uploaded to a bare ESP32 on 2026-09-10 (placeholder Wi-Fi, no
 * sensor attached). The live AD8232 acquisition path is the USB-serial sketch
 * ecg_esp32_serial/ecg_esp32_serial.ino; this Wi-Fi sketch has not delivered
 * a window to the server from real hardware. Its software contract is tested
 * with esp32_simulator.py, which sends byte-for-byte the same JSON to the same
 * endpoint.
 *
 * WIRING  (as physically built; pins changed from 34/32/33 on 2026-09-21)
 *   AD8232 OUTPUT -> ESP32 VP = GPIO36  (ADC1_CH0; input-only pin, ADC1 is the
 *                                   half that keeps working while WiFi is on -
 *                                   ADC2 does not, which is a classic ESP32 trap)
 *   AD8232 LO+    -> D32 = GPIO32  (leads-off detection; moved off the
 *                                   GPIO12 strapping pin - see ecg_esp32_serial.ino)
 *   AD8232 LO-    -> D13 = GPIO13
 *   AD8232 3.3V   -> 3V3           (NOT 5V - the AD8232 output would exceed
 *                                   the ESP32's 3.3V ADC range and clip)
 *   AD8232 GND    -> GND
 *
 * ELECTRODES (Lead I - the model is trained on Lead I and nothing else)
 *   Right Arm (red)    -> right collarbone
 *   Left  Arm (yellow) -> left collarbone
 *   Right Leg (green)  -> lower left ribs   (driven reference)
 *
 * WHAT IT SENDS
 *   Exactly 1000 raw samples at 100 Hz = one 10-second window, matching the
 *   PTB-XL recordings the model was trained on. Raw on purpose: the server
 *   z-scores the window with the same preprocess() used in training, so the
 *   device never has to agree with the server about any statistics. Getting
 *   that normalisation subtly different on the device is a whole class of bug
 *   that cannot happen if the device never does it.
 *
 *   POST /api/predict
 *   {"signal":[...1000 numbers...], "mode":"high_recall", "device_id":"ESP32-001"}
 *
 *   device_id is how the server knows whose heart this is. Re-assigning the
 *   device to a different patient is a database edit, not a reflash.
 *
 * DEPENDENCIES  (Arduino IDE -> Library Manager)
 *   ArduinoJson by Benoit Blanchon (v7+)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------------
// Settings  -  the only lines you should need to change
// ---------------------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// The machine running app.py. Must be reachable from the ESP32's network, so
// 127.0.0.1 will NOT work here - use the server's LAN address. Remember Flask
// binds to 127.0.0.1 by default; run it with host="0.0.0.0" to accept the
// device.
const char* SERVER_URL = "http://192.168.1.100:5000/api/predict";

const char* DEVICE_ID = "ESP32-001";   // must match the patient's device_id
const char* MODE      = "high_recall"; // catch as many real MIs as possible

// ---------------------------------------------------------------------------
// Signal capture
// ---------------------------------------------------------------------------
const int   ECG_PIN      = 36;   // VP
const int   LO_PLUS_PIN  = 32;   // D32 (moved off GPIO12, a boot strapping pin)
const int   LO_MINUS_PIN = 13;   // D13

const int   SAMPLE_RATE  = 100;                    // Hz - must match training
const int   N_SAMPLES    = 1000;                   // 10 seconds
const int   SAMPLE_US    = 1000000 / SAMPLE_RATE;  // 10,000 us between samples

const unsigned long SEND_INTERVAL_MS = 30000;      // one window every 30 s

int   ecgBuffer[N_SAMPLES];
float voltBuffer[N_SAMPLES];


void setup() {
  Serial.begin(115200);
  delay(200);

  // Pull-ups: an unplugged AD8232 must read as lead-off, not as two floating
  // pins that happen to read LOW (the limitation found on 2026-09-10).
  pinMode(LO_PLUS_PIN, INPUT_PULLUP);
  pinMode(LO_MINUS_PIN, INPUT_PULLUP);

  // 12-bit ADC (0..4095) over the full 0-3.3 V swing. ADC_11db is the widest
  // attenuation, which matches the AD8232's output range; a narrower setting
  // would clip the peaks of the QRS complex - the most diagnostic part of the
  // waveform.
  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  connectWiFi();
}


void connectWiFi() {
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nConnected. Device IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi failed - will retry before the next send.");
  }
}


// ---------------------------------------------------------------------------
// Are the electrodes actually on the patient?
// ---------------------------------------------------------------------------
// The AD8232 pulls LO+ or LO- HIGH when an electrode loses contact. Sending a
// window captured with a loose electrode would feed the model a flat line or
// pure noise, and it would dutifully return a probability for it. Checking
// first means the system says "check the electrodes" instead of quietly
// producing a meaningless verdict - the difference between a device that fails
// loudly and one that fails silently.
bool leadsConnected() {
  return digitalRead(LO_PLUS_PIN) == LOW && digitalRead(LO_MINUS_PIN) == LOW;
}


// ---------------------------------------------------------------------------
// Capture one 10-second window
// ---------------------------------------------------------------------------
// Timed with micros() rather than delay(), because delay() would ignore the
// time the ADC read itself takes and the true sample rate would drift below
// 100 Hz. The model expects 100 Hz exactly - a window captured at 94 Hz is a
// stretched heartbeat, and the shapes it learned would no longer line up.
bool captureWindow() {
  unsigned long nextSample = micros();

  for (int i = 0; i < N_SAMPLES; i++) {
    if (!leadsConnected()) {
      Serial.println("Electrode came loose - discarding this window.");
      return false;
    }

    while ((long)(micros() - nextSample) < 0) { /* wait for the tick */ }
    ecgBuffer[i] = analogRead(ECG_PIN);
    nextSample += SAMPLE_US;
  }

  // Convert the raw ADC counts to volts. This is a linear rescale, so it
  // cannot change what the model sees (the server z-scores the window, and
  // z-scoring removes any linear scale). It is done purely so the numbers in
  // the JSON are physically meaningful if a human reads them.
  for (int i = 0; i < N_SAMPLES; i++) {
    voltBuffer[i] = ecgBuffer[i] * (3.3f / 4095.0f);
  }
  return true;
}


// ---------------------------------------------------------------------------
// Send the window to the server
// ---------------------------------------------------------------------------
void sendWindow() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi dropped - reconnecting.");
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(20000);        // inference plus network; be generous

  // 1000 floats do not fit in a JsonDocument on this chip alongside WiFi
  // buffers, so the body is streamed out as text instead of being built in
  // memory first. Six significant figures is far finer than the ADC's own
  // resolution, so nothing is lost.
  String body;
  body.reserve(N_SAMPLES * 9 + 100);
  body = "{\"device_id\":\"";
  body += DEVICE_ID;
  body += "\",\"mode\":\"";
  body += MODE;
  body += "\",\"signal\":[";
  for (int i = 0; i < N_SAMPLES; i++) {
    if (i) body += ',';
    body += String(voltBuffer[i], 6);
  }
  body += "]}";

  int status = http.POST(body);

  if (status == 200) {
    String reply = http.getString();

    JsonDocument doc;
    if (deserializeJson(doc, reply) == DeserializationError::Ok) {
      const char* label = doc["label"] | "?";
      float prob        = doc["probability"] | 0.0f;
      const char* who   = doc["patient_name"] | "(unattributed)";

      Serial.printf("%s  p=%.4f  patient=%s\n", label, prob, who);

      // A flagged window is where a real deployment would raise an alarm -
      // buzzer, SMS to a carer, or an entry in a clinician's queue. Alerting
      // is named as future work in the thesis and is deliberately not
      // implemented here.
      if ((int)(doc["prediction"] | 0) == 1) {
        Serial.println(">>> MI PATTERN FLAGGED - seek medical advice <<<");
      }
    }
  } else if (status > 0) {
    Serial.printf("Server returned %d: %s\n", status, http.getString().c_str());
  } else {
    Serial.printf("Request failed: %s\n", http.errorToString(status).c_str());
  }

  http.end();
}


void loop() {
  if (!leadsConnected()) {
    Serial.println("Electrodes not detected - check the pads.");
    delay(2000);
    return;
  }

  Serial.println("Capturing 10 s ...");
  if (captureWindow()) {
    sendWindow();
  }

  delay(SEND_INTERVAL_MS);
}
