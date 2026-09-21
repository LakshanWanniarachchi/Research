/*
 * Copy this file to secrets.h in the same folder and fill in the four values
 * below. secrets.h is listed in .gitignore and must never be committed: it
 * holds the Wi-Fi password and the broker credentials.
 *
 *   copy secrets_example.h secrets.h        (Windows)
 *   cp   secrets_example.h secrets.h        (macOS / Linux)
 *
 * The ESP32 cannot join WPA2-Enterprise networks (eduroam and most university
 * Wi-Fi), so use a home network or a phone hotspot. The laptop running the
 * broker must be on that same network, and MQTT_HOST below is that laptop's
 * address on it - not 127.0.0.1, which on the ESP32 would mean the ESP32.
 * Find it with `ipconfig` (Windows) or `ip addr` (Linux).
 */
#pragma once

// ---------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------
#define WIFI_SSID      "your-network-name"
#define WIFI_PASSWORD  "your-network-password"

// ---------------------------------------------------------------------------
// MQTT broker  -  the machine running mosquitto and app.py
// ---------------------------------------------------------------------------
#define MQTT_HOST      "192.168.1.100"
#define MQTT_PORT      1883

// Broker account for this device, created with:
//   mosquitto_passwd mosquitto/passwd esp32-001
#define MQTT_USER      "esp32-001"
#define MQTT_PASS      "the-password-you-set"

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------
// Must match the device id registered in the web application, and the topics
// allowed for this device in mosquitto/acl. Topics are case-sensitive.
#define DEVICE_ID      "ESP32-001"
