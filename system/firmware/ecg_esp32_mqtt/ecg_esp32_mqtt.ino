/*
 * ECG acquisition  -  ESP32 + AD8232, streaming over Wi-Fi and MQTT
 * =================================================================
 *
 *   AD8232 -> ESP32 --Wi-Fi--> [ MQTT broker ] --> Flask backend -> 1D CNN
 *   ^^^^^^^^^^^^^^^^^^^^^^^^^
 *   this sketch
 *
 * This replaces the earlier version of this sketch, which captured a whole
 * ten-second window on the device and published it as one 9 KB message every
 * 30 seconds. That cannot drive a live waveform in a browser: nothing is seen
 * until a window is finished. This version publishes one second of samples at
 * a time, so the backend can draw the trace as it arrives, and assembles the
 * ten-second model window itself - using the same window rules as the USB
 * path, in one place, on the PC.
 *
 * USB is now only power, flashing and the debug console. The ECG itself
 * travels over Wi-Fi.
 *
 * WIRING  (unchanged, as verified on 2026-09-21 and 2026-09-23)
 *   AD8232 OUTPUT -> VP  = GPIO36   ADC1_CH0, input-only. ADC1 keeps working
 *                                   while Wi-Fi is on; ADC2 does not, which is
 *                                   why this pin matters for this sketch.
 *   AD8232 LO+    -> D32 = GPIO32   leads-off detection. Moved off GPIO12 on
 *                                   2026-09-21: GPIO12 is the MTDI strapping
 *                                   pin, and LO+ holding it HIGH at reset made
 *                                   the board watchdog-reset and an upload fail.
 *   AD8232 LO-    -> D13 = GPIO13   leads-off detection
 *   AD8232 3.3V   -> 3V3            NOT 5V: the output would exceed the 3.3 V
 *                                   ADC range and clip the QRS peaks.
 *   AD8232 GND    -> GND
 *
 * TOPICS   (device id only - never a patient name, see mosquitto/acl)
 *   ecg/devices/<DEVICE_ID>/data     published, QoS 0: one second of samples
 *   ecg/devices/<DEVICE_ID>/status   published retained, QoS 1: online/offline
 *   ecg/devices/<DEVICE_ID>/event    published, QoS 1: reconnects, overruns
 *   ecg/devices/<DEVICE_ID>/result   subscribed, QoS 1: the server's verdict
 *
 * CHUNK PAYLOAD
 *   {"device_id":"ESP32-001","seq":41,"t_us":123456789,"fs":100,"n":100,
 *    "lop":"000...0","lom":"000...0","adc":[2071,2068,...]}
 *   seq  increments by one per chunk, so the backend can see a lost chunk
 *   t_us micros() at the first sample of the chunk, for rate measurement
 *   lop  one character per sample, '1' = LO+ was high (that electrode off)
 *   lom  the same for LO-, so the backend keeps the per-lead detail
 *   adc  raw 12-bit counts. The device never normalises: the server z-scores
 *        with the same code the model was trained with, so the two cannot
 *        disagree about a statistic.
 *
 * CREDENTIALS
 *   In secrets.h, which is not in version control. Copy secrets_example.h.
 *
 * DEPENDENCIES  (Arduino IDE -> Library Manager)
 *   PubSubClient by Nick O'Leary
 */

#include <WiFi.h>
#include <PubSubClient.h>

#include "secrets.h"

#define FIRMWARE_VERSION "ecg-mqtt-2.0"

// ---------------------------------------------------------------------------
// Signal capture  -  unchanged from the verified USB sketch
// ---------------------------------------------------------------------------
const int ECG_PIN      = 36;   // VP
const int LO_PLUS_PIN  = 32;   // D32
const int LO_MINUS_PIN = 13;   // D13

const int SAMPLE_RATE = 100;                     // Hz - must match training
const int SAMPLE_US   = 1000000 / SAMPLE_RATE;   // 10,000 us between samples

const int CHUNK_SAMPLES = 100;                   // one second per message

// 100 readings of up to 4 characters plus the envelope and the lead-off
// string. PubSubClient's default buffer is 256 bytes, which would silently
// drop every chunk: publish() simply returns false. 2 KB is comfortable.
const uint16_t MQTT_BUFFER = 2048;

uint16_t chunkAdc[CHUNK_SAMPLES];
char     chunkLoP[CHUNK_SAMPLES + 1];   // LO+ per sample, '1' = electrode off
char     chunkLoM[CHUNK_SAMPLES + 1];   // LO- per sample

int           chunkCount  = 0;
unsigned long chunkStartUs = 0;
unsigned long nextSample   = 0;
unsigned long seq          = 0;
unsigned long overruns     = 0;
unsigned long publishFails = 0;

char topicData[72];
char topicStatus[72];
char topicEvent[72];
char topicResult[72];

WiFiClient   net;
PubSubClient mqtt(net);

// Reconnection is attempted on a timer rather than in a blocking loop, so
// sampling never stops while the link is down. A device that freezes for
// twenty seconds because the broker went away is a device that loses the
// heartbeat it was supposed to be watching.
unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;
const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long MQTT_RETRY_MS = 3000;
bool wasConnected = false;


// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);

  // Pull-ups so that a missing or unplugged AD8232 reads as lead-off rather
  // than as two floating pins that read LOW at random.
  pinMode(LO_PLUS_PIN, INPUT_PULLUP);
  pinMode(LO_MINUS_PIN, INPUT_PULLUP);

  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  snprintf(topicData,   sizeof(topicData),   "ecg/devices/%s/data",   DEVICE_ID);
  snprintf(topicStatus, sizeof(topicStatus), "ecg/devices/%s/status", DEVICE_ID);
  snprintf(topicEvent,  sizeof(topicEvent),  "ecg/devices/%s/event",  DEVICE_ID);
  snprintf(topicResult, sizeof(topicResult), "ecg/devices/%s/result", DEVICE_ID);

  Serial.printf("# %s  device=%s  fs=%d Hz  chunk=%d samples\n",
                FIRMWARE_VERSION, DEVICE_ID, SAMPLE_RATE, CHUNK_SAMPLES);
  Serial.printf("# pins: ecg=%d lo+=%d lo-=%d\n",
                ECG_PIN, LO_PLUS_PIN, LO_MINUS_PIN);
  Serial.printf("# broker: %s:%d as %s\n", MQTT_HOST, MQTT_PORT, MQTT_USER);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);          // sleep adds latency and drops chunks
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWifiAttempt = millis();

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setBufferSize(MQTT_BUFFER);
  mqtt.setKeepAlive(30);

  nextSample = micros();
  chunkStartUs = nextSample;
}


// ---------------------------------------------------------------------------
// The server's verdict comes back here
// ---------------------------------------------------------------------------
// Printed to the console only. Acting on it - a buzzer, an SMS to a carer - is
// named as future work in the thesis and is deliberately not implemented.
void onMessage(char* topic, byte* payload, unsigned int length) {
  Serial.print("# result: ");
  for (unsigned int i = 0; i < length && i < 200; i++) Serial.write(payload[i]);
  Serial.println();
}


// ---------------------------------------------------------------------------
// Connectivity, without blocking the sampler
// ---------------------------------------------------------------------------
void serviceWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  if (millis() - lastWifiAttempt < WIFI_RETRY_MS) return;

  lastWifiAttempt = millis();
  Serial.println("# wifi: reconnecting");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}


void publishStatus(const char* state) {
  char payload[200];
  snprintf(payload, sizeof(payload),
           "{\"device_id\":\"%s\",\"state\":\"%s\",\"fw\":\"%s\",\"fs\":%d,"
           "\"ip\":\"%s\",\"rssi\":%d}",
           DEVICE_ID, state, FIRMWARE_VERSION, SAMPLE_RATE,
           WiFi.localIP().toString().c_str(), (int)WiFi.RSSI());
  mqtt.publish(topicStatus, payload, true);      // retained
}


void publishEvent(const char* kind, unsigned long value) {
  char payload[160];
  snprintf(payload, sizeof(payload),
           "{\"device_id\":\"%s\",\"event\":\"%s\",\"value\":%lu,\"seq\":%lu}",
           DEVICE_ID, kind, value, seq);
  mqtt.publish(topicEvent, payload);
}


void serviceMqtt() {
  if (mqtt.connected()) return;
  if (WiFi.status() != WL_CONNECTED) return;
  if (millis() - lastMqttAttempt < MQTT_RETRY_MS) return;

  lastMqttAttempt = millis();

  // The last will is registered with the broker now and published BY THE
  // BROKER if this device disappears without saying goodbye, so a device whose
  // battery dies does not appear online forever. Retained, so a dashboard that
  // connects later still learns the current state.
  char willPayload[120];
  snprintf(willPayload, sizeof(willPayload),
           "{\"device_id\":\"%s\",\"state\":\"offline\"}", DEVICE_ID);

  bool ok = mqtt.connect(DEVICE_ID, MQTT_USER, MQTT_PASS,
                         topicStatus, 1, true, willPayload);
  if (ok) {
    Serial.printf("# mqtt: connected to %s:%d\n", MQTT_HOST, MQTT_PORT);
    publishStatus("online");
    mqtt.subscribe(topicResult, 1);
    if (wasConnected) publishEvent("reconnected", 1);
    wasConnected = true;
  } else {
    // rc -4 timeout, -2 network, 4 bad credentials, 5 not authorised
    Serial.printf("# mqtt: connect failed rc=%d\n", mqtt.state());
  }
}


// ---------------------------------------------------------------------------
// Publish one second of ECG
// ---------------------------------------------------------------------------
// Built directly into the client's buffer rather than into a String: repeated
// String concatenation fragments the heap on this chip, and the payload size
// is known in advance.
bool publishChunk() {
  if (!mqtt.connected()) return false;

  char payload[MQTT_BUFFER];
  int n = snprintf(payload, sizeof(payload),
                   "{\"device_id\":\"%s\",\"seq\":%lu,\"t_us\":%lu,\"fs\":%d,"
                   "\"n\":%d,\"lop\":\"%s\",\"lom\":\"%s\",\"adc\":[",
                   DEVICE_ID, seq, chunkStartUs, SAMPLE_RATE,
                   chunkCount, chunkLoP, chunkLoM);

  for (int i = 0; i < chunkCount && n < (int)sizeof(payload) - 8; i++) {
    n += snprintf(payload + n, sizeof(payload) - n, i ? ",%u" : "%u",
                  chunkAdc[i]);
  }
  n += snprintf(payload + n, sizeof(payload) - n, "]}");

  bool ok = mqtt.publish(topicData, (const uint8_t*)payload, n, false);
  if (!ok) {
    publishFails++;
    Serial.printf("# publish failed (seq %lu, %d bytes, total fails %lu)\n",
                  seq, n, publishFails);
  }
  return ok;
}


// ---------------------------------------------------------------------------
// Main loop: sample on the tick, publish once a second, reconnect in the gaps
// ---------------------------------------------------------------------------
// Timed against micros() rather than delay(): the next tick is scheduled from
// the previous tick, so the time spent reading, publishing and reconnecting
// does not accumulate as drift. The model expects 100 Hz exactly; a window
// captured at a drifted rate is a stretched heartbeat whose shapes it has
// never seen.
void loop() {
  mqtt.loop();
  serviceWifi();
  serviceMqtt();

  if ((long)(micros() - nextSample) < 0) return;      // not time yet

  if (chunkCount == 0) chunkStartUs = micros();

  chunkAdc[chunkCount] = (uint16_t)analogRead(ECG_PIN);
  chunkLoP[chunkCount] = (digitalRead(LO_PLUS_PIN) == HIGH) ? '1' : '0';
  chunkLoM[chunkCount] = (digitalRead(LO_MINUS_PIN) == HIGH) ? '1' : '0';
  chunkCount++;

  nextSample += SAMPLE_US;

  if (chunkCount >= CHUNK_SAMPLES) {
    chunkLoP[chunkCount] = '\0';
    chunkLoM[chunkCount] = '\0';
    publishChunk();
    seq++;
    chunkCount = 0;
  }

  // If the loop fell more than one tick behind - a slow publish, a reconnect -
  // re-anchor instead of firing a burst of late samples, and say so. The
  // backend sees the same event as a timing gap and refuses the window rather
  // than scoring a signal with a hole in it.
  if ((long)(micros() - nextSample) > SAMPLE_US) {
    overruns++;
    nextSample = micros() + SAMPLE_US;
    if (overruns % 10 == 1) {
      Serial.printf("# overrun %lu\n", overruns);
      publishEvent("overrun", overruns);
    }
  }
}

/*
 * SECURITY NOTE, stated in the thesis rather than glossed over
 * -----------------------------------------------------------
 * Port 1883 is unencrypted. On a private hotspot with an authenticated broker
 * and per-device ACLs that is acceptable for a prototype: every client must
 * present credentials, and this device may publish only under its own device
 * id. It is not acceptable across the public internet, where the samples and
 * the credentials would be readable by anything on the path. A deployment uses
 * port 8883 with WiFiClientSecure, the broker's CA certificate, and per-device
 * credentials issued at registration.
 */
