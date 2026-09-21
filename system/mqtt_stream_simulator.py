"""
A stand-in for the streaming ESP32
==================================

Publishes one-second ECG chunks to the broker in exactly the format
firmware/ecg_esp32_mqtt prints, so the backend cannot tell this apart from the
device. It exists for two reasons: the server path can be tested when the
hardware is not to hand, and a fault can be attributed - if the device fails
but this does not, the fault is in the device, not in the server.

It replays real PTB-XL recordings from fold 10, the fold the model never saw
while training, converted to ADC counts so the numbers look like a device's.
It is a transport test, not an evaluation: agreement on a handful of replayed
recordings says nothing about the model's accuracy, which is Chapter 6's job.

    python mqtt_stream_simulator.py --host 127.0.0.1 --username esp32-001 \
        --password ... --device ESP32-001 --seconds 12

Every message it sends says it came from a simulator in the `sim` field, so a
reading captured during a demonstration can never be mistaken for one from a
person.
"""

import argparse
import json
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

import models

TEST_FOLD = 10
FS = 100
CHUNK = 100                      # samples per message, i.e. one second


def to_adc_counts(signal):
    """Millivolts to 12-bit counts, the way the AD8232 and the ESP32 would.

    The model z-scores every window, so this scaling cannot change a verdict.
    It exists so the payload carries the integers a real device sends, and so
    the backend's range checks are exercised for real.
    """
    signal = np.asarray(signal, dtype=np.float64)
    spread = np.abs(signal).max() or 1.0
    counts = 2048 + (signal / spread) * 700
    return np.clip(counts, 0, 4095).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--username", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--device", default="ESP32-001")
    ap.add_argument("--seconds", type=int, default=12,
                    help="how many one-second chunks to publish")
    ap.add_argument("--type", choices=["any", "mi", "normal"], default="any")
    ap.add_argument("--realtime", action="store_true",
                    help="wait a second between chunks, as the device does")
    ap.add_argument("--drop", type=int, default=-1,
                    help="skip this chunk number, to prove a gap is detected")
    args = ap.parse_args()

    label = {"mi": 1, "normal": 0}.get(args.type)
    conn = models.connect()
    rows = []
    while sum(len(r) for r in rows) < args.seconds * CHUNK:
        row = models.random_recording(conn, fold=TEST_FOLD, label=label)
        if row is None:
            sys.exit("no recordings in the database; run 1_extract_data.py first")
        rows.append(to_adc_counts(models.blob_to_signal(row["signal"])))
    conn.close()
    samples = np.concatenate(rows)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.device)
    if args.username:
        client.username_pw_set(args.username, args.password)
    status_topic = f"ecg/devices/{args.device}/status"
    client.will_set(status_topic,
                    json.dumps({"device_id": args.device, "state": "offline"}),
                    qos=1, retain=True)
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as exc:
        sys.exit(f"cannot reach the broker at {args.host}:{args.port} ({exc})")
    client.loop_start()
    client.publish(status_topic, json.dumps(
        {"device_id": args.device, "state": "online", "fw": "simulator",
         "fs": FS, "sim": True}), qos=1, retain=True)

    print(f"publishing {args.seconds} chunks as {args.device} -> {args.host}:{args.port}")
    t_us = 1_000_000
    sent = 0
    for seq in range(args.seconds):
        if seq == args.drop:
            print(f"  seq {seq}: deliberately not sent (gap test)")
            t_us += CHUNK * (1_000_000 // FS)
            continue
        block = samples[seq * CHUNK:(seq + 1) * CHUNK]
        payload = json.dumps({
            "device_id": args.device, "seq": seq, "t_us": t_us, "fs": FS,
            "n": len(block), "lop": "0" * len(block), "lom": "0" * len(block),
            "adc": [int(v) for v in block], "sim": True,
        })
        info = client.publish(f"ecg/devices/{args.device}/data", payload, qos=0)
        info.wait_for_publish(timeout=5)
        sent += 1
        t_us += CHUNK * (1_000_000 // FS)
        if args.realtime:
            time.sleep(1.0)

    time.sleep(2.0)             # let any verdict come back
    client.publish(status_topic, json.dumps(
        {"device_id": args.device, "state": "offline"}), qos=1, retain=True)
    client.loop_stop()
    client.disconnect()
    print(f"sent {sent} chunk(s) = {sent * CHUNK} samples")
    print("This is replayed PTB-XL data over MQTT, not a recording from a person.")


if __name__ == "__main__":
    main()
