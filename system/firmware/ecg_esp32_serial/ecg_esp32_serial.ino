/*
 * ECG acquisition  -  ESP32 + AD8232, streaming over USB serial
 * =============================================================
 *
 *   AD8232 -> ESP32 --USB serial--> ecg_serial_monitor.py -> model.py (1D CNN)
 *   ^^^^^^^^^^^^^^^
 *   this sketch
 *
 * The Wi-Fi sketches (ecg_esp32.ino, ecg_esp32_mqtt.ino) capture a whole
 * 10-second window on the device and send it as one JSON body, so nothing can
 * be seen until the window is finished. This sketch sends every sample the
 * moment it is taken, which is what a live waveform display needs. Windowing,
 * lead-off gating of the window, preprocessing and inference all happen on the
 * PC, in the same code the server uses.
 *
 * WIRING  (as physically built, ESP32 DevKit with DOIT-style silkscreen)
 *   AD8232 OUTPUT -> VP  = GPIO36   ADC1_CH0, input-only. ADC1 keeps working
 *                                   while Wi-Fi is on; ADC2 does not.
 *   AD8232 LO+    -> D32 = GPIO32   leads-off detection. Originally wired to
 *                                   D12 = GPIO12, the MTDI strapping pin. On
 *                                   2026-09-21 LO+ held GPIO12 HIGH at reset
 *                                   (boot:0x33), the board watchdog-reset and an
 *                                   upload failed, so LO+ was moved to GPIO32.
 *   AD8232 LO-    -> D13 = GPIO13   leads-off detection
 *   AD8232 3.3V   -> 3V3            NOT 5V: the output would exceed the 3.3 V
 *                                   ADC range and clip.
 *   AD8232 GND    -> GND
 *
 * SERIAL FORMAT  (115200 baud, one line per sample, 100 lines per second)
 *   seq,t_us,ecg_adc,lo_plus,lo_minus
 *     seq       sample counter from boot; a gap means a sample was lost
 *     t_us      micros() at the moment of the ADC read
 *     ecg_adc   raw 12-bit reading, 0..4095
 *     lo_plus   1 = LO+ HIGH (electrode off), 0 = connected
 *     lo_minus  1 = LO- HIGH (electrode off), 0 = connected
 *   Any line starting with '#' is a status message, never a sample.
 *
 * GPIO36 has no internal pull resistor, so with the AD8232 unplugged it reads
 * noise. That is harmless because the LO pins are pulled up: an unplugged
 * sensor always reports lead-off, and the PC never treats it as an ECG.
 *
 * Samples taken while a lead is off are still sent, flagged, so the PC can say
 * "lead off" instead of the waveform simply stopping. The PC never builds a
 * model window from a flagged sample.
 */

// ---------------------------------------------------------------------------
// Signal capture
// ---------------------------------------------------------------------------
const int ECG_PIN      = 36;   // VP
const int LO_PLUS_PIN  = 32;   // D32 (moved off GPIO12, a boot strapping pin)
const int LO_MINUS_PIN = 13;   // D13

const int SAMPLE_RATE = 100;                     // Hz - must match training
const int SAMPLE_US   = 1000000 / SAMPLE_RATE;   // 10,000 us between samples

const long SERIAL_BAUD = 115200;

unsigned long nextSample;
unsigned long seq = 0;
unsigned long overruns = 0;


void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  // Pull-ups, so that a missing or unplugged AD8232 reads as LEAD OFF instead
  // of two floating pins that read LOW at random (seen on 2026-09-21 with the
  // sensor disconnected). The AD8232 drives LO+/LO- actively, so it overrides
  // the weak internal pull-up when it is present. Set after boot, so it does
  // not affect any strapping level sampled at reset.
  pinMode(LO_PLUS_PIN, INPUT_PULLUP);
  pinMode(LO_MINUS_PIN, INPUT_PULLUP);

  // 12-bit ADC (0..4095) with ADC_11db, the widest attenuation, as in the
  // Wi-Fi sketches; a narrower setting would clip the QRS peaks.
  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  Serial.println("# ECG_SERIAL v2 fs_hz=100 adc_bits=12 atten=11db ecg_pin=36 lo_plus_pin=32 lo_minus_pin=13");
  Serial.println("# columns: seq,t_us,ecg_adc,lo_plus,lo_minus");

  nextSample = micros();
}


// ---------------------------------------------------------------------------
// Sampling loop
// ---------------------------------------------------------------------------
// Timed against micros() rather than delay(), exactly as captureWindow() in the
// Wi-Fi sketches: the next tick is scheduled from the previous tick, not from
// "now", so the time spent reading and printing does not accumulate as drift.
void loop() {
  if ((long)(micros() - nextSample) < 0) return;   // not time yet

  unsigned long t = micros();
  int ecg = analogRead(ECG_PIN);
  int loPlus  = digitalRead(LO_PLUS_PIN);
  int loMinus = digitalRead(LO_MINUS_PIN);

  Serial.printf("%lu,%lu,%d,%d,%d\n", seq, t, ecg, loPlus, loMinus);
  seq++;

  nextSample += SAMPLE_US;

  // If the loop fell more than one tick behind (e.g. the serial buffer was
  // full), re-anchor instead of firing a burst of late samples, and say so.
  if ((long)(micros() - nextSample) > SAMPLE_US) {
    overruns++;
    Serial.printf("# overrun %lu\n", overruns);
    nextSample = micros() + SAMPLE_US;
  }
}
