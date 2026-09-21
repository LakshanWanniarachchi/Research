"""
MQTT device simulator  -  stands in for the ESP32 that is not built yet
======================================================================

    [ this script ] --> MQTT broker --> mqtt_bridge.py --> 1D CNN --> SQLite

Speaks exactly the contract firmware/ecg_esp32_mqtt.ino implements: it
publishes a 1000-sample raw window to ecg/<device_id>/reading and listens on
ecg/<device_id>/result for the verdict. The bridge cannot tell the two apart.

The signals are real PTB-XL recordings drawn from fold 10 only, the fold the
model never saw during training or threshold tuning. Replaying training data
would produce beautiful numbers that mean nothing.

Usage
-----
    python mqtt_device_simulator.py                       # 5 readings
    python mqtt_device_simulator.py --n 20 --interval 1
    python mqtt_device_simulator.py --type mi --noise 0.05
    python mqtt_device_simulator.py --host broker.example.com --port 1883

Requires a broker to be running. See mosquitto/mosquitto.conf.
"""

import argparse
import json
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

import models

TEST_FOLD = 10

_results = []


def on_message(client, userdata, msg):
    """The server's verdict comes back here."""
    try:
        r = json.loads(msg.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("  <- unreadable reply")
        return
    _results.append(r)
    verdict = "MI" if r.get("prediction") else "normal"
    print(f"  <- p={r.get('probability'):.4f}  said {verdict:<6} "
          f"{r.get('inference_ms', 0):6.1f} ms  "
          f"patient={r.get('patient_name') or '(unattributed)'}")
    if r.get("warning"):
        print(f"     !! {r['warning']}")


def add_sensor_noise(signal, level, rng):
    """Roughen a clean hospital signal to imitate a cheap electrode.

    Two effects, both real: slow baseline wander from breathing and movement,
    and gaussian hash from the amplifier and mains.
    """
    n = len(signal)
    t = np.arange(n) / 100.0
    wander = level * 3 * np.sin(2 * np.pi * rng.uniform(0.15, 0.5) * t)
    hiss = rng.normal(0, level, n)
    return signal + wander.astype(np.float32) + hiss.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Pretend to be an ESP32 on MQTT.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--device", default="ESP32-001")
    p.add_argument("--token", default="", help="shared secret for this device")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--mode", default="high_recall",
                   choices=["default", "best_f1", "high_recall"])
    p.add_argument("--type", default="any", choices=["any", "mi", "normal"])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    label = {"mi": 1, "normal": 0}.get(args.type)

    # Pull the windows before connecting, so a database problem surfaces before
    # a broker problem and the two are not confused.
    conn = models.connect()
    windows = []
    for _ in range(args.n):
        row = models.random_recording(conn, fold=TEST_FOLD, label=label)
        if row is None:
            conn.close()
            sys.exit(f"No '{args.type}' recordings in fold {TEST_FOLD}. "
                     f"Is ecg.db populated?")
        windows.append((models.blob_to_signal(row["signal"]),
                        row["label"], row["id"]))
    conn.close()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=args.device)
    if args.username:
        client.username_pw_set(args.username, args.password)
    client.on_message = on_message

    status_topic = f"ecg/{args.device}/status"
    client.will_set(status_topic, "offline", qos=1, retain=True)

    try:
        client.connect(args.host, args.port, keepalive=60)
    except OSError as exc:
        sys.exit(f"Cannot reach the broker at {args.host}:{args.port}\n  {exc}\n\n"
                 f"Start one first. See mosquitto/mosquitto.conf.")

    client.publish(status_topic, "online", qos=1, retain=True)
    client.subscribe(f"ecg/{args.device}/result", qos=1)
    client.loop_start()

    print("=" * 66)
    print(f"  MQTT DEVICE SIMULATOR   '{args.device}'  ->  "
          f"{args.host}:{args.port}")
    print("=" * 66)
    print(f"  topic     : ecg/{args.device}/reading")
    print(f"  operating : {args.mode}")
    print(f"  noise     : {args.noise if args.noise else 'none (clean signal)'}")
    print()

    topic = f"ecg/{args.device}/reading"
    for i, (signal, truth, rec_id) in enumerate(windows, 1):
        if args.noise:
            signal = add_sensor_noise(signal, args.noise, rng)

        payload = json.dumps({
            "device_id": args.device,
            "token": args.token,
            "mode": args.mode,
            "signal": [float(v) for v in signal],
        })

        info = client.publish(topic, payload, qos=1)
        info.wait_for_publish(timeout=10)
        actual = "MI" if truth else "normal"
        print(f"  [{i:3d}/{args.n}] -> rec {rec_id:>6}  truly {actual:<6} "
              f"({len(payload)} bytes)")

        if i < len(windows):
            time.sleep(args.interval)

    # Give the last verdict time to come back before disconnecting.
    time.sleep(2.0)
    client.publish(status_topic, "offline", qos=1, retain=True)
    client.loop_stop()
    client.disconnect()

    print()
    print("-" * 66)
    print(f"  published : {len(windows)}")
    print(f"  verdicts  : {len(_results)} returned")
    if _results:
        flagged = sum(r.get("prediction", 0) for r in _results)
        warm = [r["inference_ms"] for r in _results if r.get("inference_ms", 0) < 1000]
        print(f"  flagged MI: {flagged}")
        if warm:
            print(f"  model time: mean {np.mean(warm):.1f} ms over {len(warm)} warm calls")
    print()
    print("  Agreement over a handful of readings bounces around and is not a")
    print("  result. The thesis figure is the full fold-10 evaluation.")
    print("-" * 66)


if __name__ == "__main__":
    main()
