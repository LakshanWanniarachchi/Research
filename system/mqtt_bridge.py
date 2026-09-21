"""
MQTT bridge  -  receiving readings from devices over the internet
================================================================

    AD8232 -> ESP32 --WiFi--> [ MQTT broker ] --> this bridge -> 1D CNN -> SQLite
                                                        ^^^^ this file

Why MQTT rather than the HTTP endpoint
--------------------------------------
The existing /api/predict endpoint needs the device to reach the server
directly. That works on a home network and stops working the moment the server
is behind a router, which it always is. MQTT inverts the direction: both the
device and the server open OUTBOUND connections to a broker, so neither needs a
public address or a forwarded port. It also keeps working when a device drops
off WiFi mid-transmission, because the broker holds the connection state rather
than the application.

The trade is an extra component that has to be running. If the broker is down,
readings are not delayed, they are lost, unless the device buffers them. That
limitation is stated in the thesis rather than hidden.

Topics
------
    ecg/<device_id>/reading    device  -> server   a 10-second window
    ecg/<device_id>/result     server  -> device   the verdict
    ecg/<device_id>/status     device  -> server   online / offline (retained)

A wildcard subscription to ecg/+/reading picks up every device without the
server needing to know in advance which devices exist.

Design note: transport and logic are separated
----------------------------------------------
`handle_reading()` takes bytes and returns a dict. It touches no network, so it
can be tested exhaustively without a broker running. The paho client is a thin
shell around it. Mixing the two would mean the only way to test a malformed
payload was to stand up a broker and publish to it.
"""

import json
import os
import threading
import time

import paho.mqtt.client as mqtt

import config
import model as mdl
import models
import mqtt_stream

# ---------------------------------------------------------------------------
# Configuration, all overridable from the environment
# ---------------------------------------------------------------------------
MQTT_ENABLED = os.environ.get("MQTT_ENABLED", "0") == "1"
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME") or None
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD") or None
MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "ecg-server")

# Refuse a device that does not present a matching token. Off by default so the
# existing HTTP simulator keeps working unchanged; the thesis records this as a
# deliberate default rather than an oversight.
REQUIRE_DEVICE_TOKEN = os.environ.get("REQUIRE_DEVICE_TOKEN", "0") == "1"

TOPIC_READING = "ecg/+/reading"
RESULT_TEMPLATE = "ecg/{device_id}/result"

# The streaming contract added in 2026-09: the device publishes one second of
# samples at a time instead of a finished window, so a browser can watch the
# trace arrive. The legacy whole-window topic above is kept because the
# simulator and its tests speak it, and because a device that only ever sends
# complete windows is still a valid client of this system.
TOPIC_DATA = "ecg/devices/+/data"
TOPIC_STATUS = "ecg/devices/+/status"
TOPIC_EVENT = "ecg/devices/+/event"
STREAM_RESULT_TEMPLATE = "ecg/devices/{device_id}/result"

# Filled in by start(); handed the completed windows to score.
STREAMS = mqtt_stream.StreamRegistry()

# Set by start(); the bridge reuses the model the web application already
# loaded rather than loading a second copy into the same process.
_CNN = None
_THRESHOLDS = {}
_INPUT_LENGTH = 1000


class PayloadError(ValueError):
    """The message was not something this bridge can act on."""


# ---------------------------------------------------------------------------
# 1. Message handling  (no network here, so this is directly testable)
# ---------------------------------------------------------------------------
def parse_payload(raw, input_length=None):
    """Turn raw MQTT bytes into a validated reading.

    Raises PayloadError with a specific reason for anything malformed. A device
    on a public broker can send whatever it likes, including nothing, so every
    field is checked before it reaches the model.
    """
    expected = input_length or _INPUT_LENGTH

    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PayloadError("payload is not valid UTF-8 JSON") from exc

    if not isinstance(body, dict):
        raise PayloadError("payload must be a JSON object")

    signal = body.get("signal")
    if not isinstance(signal, list):
        raise PayloadError("missing 'signal' list")
    if len(signal) != expected:
        raise PayloadError(
            f"signal must be exactly {expected} samples, got {len(signal)}")
    if not all(isinstance(v, (int, float)) for v in signal):
        raise PayloadError("signal must contain only numbers")

    mode = body.get("mode", "high_recall")
    device_id = body.get("device_id")
    if not device_id or not isinstance(device_id, str):
        raise PayloadError("missing 'device_id'")

    return {
        "device_id": device_id,
        "token": body.get("token"),
        "signal": signal,
        "mode": mode,
    }


def device_id_from_topic(topic):
    """Pull the device id out of ecg/<device_id>/reading."""
    parts = topic.split("/")
    if len(parts) != 3 or parts[0] != "ecg" or parts[2] != "reading":
        raise PayloadError(f"unexpected topic '{topic}'")
    return parts[1]


def handle_reading(topic, raw, cnn=None, thresholds=None,
                   require_token=None):
    """Score one incoming reading and store it. Returns the result dict.

    This is the whole of the bridge's behaviour. The paho callback does nothing
    except call it and publish what it returns.
    """
    cnn = cnn if cnn is not None else _CNN
    thresholds = thresholds if thresholds is not None else _THRESHOLDS
    if require_token is None:
        require_token = REQUIRE_DEVICE_TOKEN

    topic_device = device_id_from_topic(topic)
    reading = parse_payload(raw)

    # The topic and the body must agree. If they do not, one device is
    # publishing under another device's topic, which is either a bug or an
    # attempt to file readings against somebody else's record.
    if reading["device_id"] != topic_device:
        raise PayloadError(
            f"device_id '{reading['device_id']}' does not match topic "
            f"'{topic_device}'")

    if reading["mode"] not in thresholds:
        raise PayloadError(f"mode must be one of {sorted(thresholds)}")

    conn = models.connect()
    try:
        if require_token:
            patient = models.get_patient_by_device_token(
                conn, topic_device, reading["token"])
            if patient is None:
                raise PayloadError("device id or token not recognised")
        else:
            patient = models.get_patient_by_device(conn, topic_device)

        start = time.perf_counter()
        probability, was_warmup = mdl.predict_probability(cnn, reading["signal"])
        inference_ms = (time.perf_counter() - start) * 1000

        threshold = thresholds[reading["mode"]]
        prediction = 1 if probability >= threshold else 0

        models.save_prediction(
            conn,
            source="device",
            probability=probability,
            threshold=threshold,
            mode=reading["mode"],
            prediction=prediction,
            inference_ms=inference_ms,
            patient_id=patient["id"] if patient else None,
            is_warmup=was_warmup,
        )
    finally:
        conn.close()

    result = {
        "device_id": topic_device,
        "probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "mode": reading["mode"],
        "prediction": prediction,
        "label": "MI PATTERN DETECTED" if prediction else "NORMAL",
        "inference_ms": round(inference_ms, 1),
        "patient_id": patient["id"] if patient else None,
        "patient_name": patient["name"] if patient else None,
    }
    if patient is None:
        result["warning"] = (f"device '{topic_device}' is not registered to any "
                             f"patient; the reading was stored unattributed")
    return result


# ---------------------------------------------------------------------------
# 2. Transport
# ---------------------------------------------------------------------------
def _on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        # Subscribing inside on_connect rather than once after connect() is
        # what makes a reconnect self-healing: paho re-runs this callback and
        # the subscriptions come back with it.
        client.subscribe([(TOPIC_READING, 1), (TOPIC_DATA, 0),
                          (TOPIC_STATUS, 1), (TOPIC_EVENT, 1)])
        print(f"  MQTT connected to {MQTT_HOST}:{MQTT_PORT}, subscribed to "
              f"{TOPIC_DATA}, {TOPIC_STATUS}, {TOPIC_EVENT}, {TOPIC_READING}")
    else:
        print(f"  MQTT connection refused: {reason_code}")


def _on_disconnect(client, userdata, flags=None, reason_code=None,
                   properties=None):
    """Report a dropped broker connection instead of going quiet.

    paho reconnects by itself from loop_forever(); what it does not do is tell
    anyone it lost the link, and a dashboard that keeps showing the last known
    state of a device it can no longer hear from is worse than one that admits
    the connection is down.
    """
    print(f"  MQTT disconnected (reason {reason_code}); reconnecting")


def _on_message(client, userdata, msg):
    if msg.topic.startswith("ecg/devices/"):
        _handle_stream_message(client, msg)
        return

    try:
        result = handle_reading(msg.topic, msg.payload)
    except PayloadError as exc:
        # A bad message from one device must not stop the bridge serving the
        # others, so it is reported and discarded rather than raised.
        print(f"  MQTT rejected a message on {msg.topic}: {exc}")
        return
    except Exception as exc:                      # pragma: no cover
        print(f"  MQTT handler failed on {msg.topic}: {exc!r}")
        return

    client.publish(RESULT_TEMPLATE.format(device_id=result["device_id"]),
                   json.dumps(result), qos=1)
    flag = "MI" if result["prediction"] else "normal"
    print(f"  MQTT {result['device_id']}: p={result['probability']:.4f} "
          f"-> {flag} ({result['inference_ms']:.0f} ms)")


def _handle_stream_message(client, msg):
    """One chunk, status or event from a streaming device.

    The registry never raises: a device sending nonsense is counted and
    ignored so that the others keep being served. A window that comes back
    completed has already been scored and stored by the callback given to the
    registry in start(); all that is left is to tell the device.
    """
    window = STREAMS.handle_message(msg.topic, msg.payload)
    if window is None:
        return
    device_id = msg.topic.split("/")[2]
    stream = STREAMS.get(device_id, create=False)
    result = stream.last_result if stream else None
    if not result or "error" in result:
        return
    client.publish(STREAM_RESULT_TEMPLATE.format(device_id=device_id),
                   json.dumps(result), qos=1)
    flag = "MI" if result.get("prediction") else "normal"
    print(f"  MQTT {device_id}: window scored p={result.get('probability')} "
          f"-> {flag}")


def build_client():
    """Construct a paho client configured for this bridge.

    paho-mqtt 2.x requires the callback API version to be declared. Passing
    VERSION2 selects the signature used by the callbacks above; omitting it
    raises at construction on 2.x and silently uses the old signatures on 1.x.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=MQTT_CLIENT_ID, clean_session=True)
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.on_disconnect = _on_disconnect
    return client


def start(cnn, thresholds, input_length=1000, on_window=None):
    """Start the bridge in a background thread. Returns the client, or None.

    Failure to reach the broker is not fatal. The web application still serves
    the dashboard and the HTTP endpoint, and says so, rather than refusing to
    start because an optional transport is unavailable.
    """
    global _CNN, _THRESHOLDS, _INPUT_LENGTH
    _CNN, _THRESHOLDS, _INPUT_LENGTH = cnn, thresholds, input_length
    if on_window is not None:
        STREAMS.on_window = on_window

    if not MQTT_ENABLED:
        print("  MQTT bridge disabled (set MQTT_ENABLED=1 to turn it on)")
        return None

    client = build_client()
    try:
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    except OSError as exc:
        print(f"  MQTT broker unreachable at {MQTT_HOST}:{MQTT_PORT} ({exc}). "
              f"The HTTP endpoint still works.")
        return None

    thread = threading.Thread(target=client.loop_forever, daemon=True,
                              name="mqtt-bridge")
    thread.start()
    return client
