"""
The ESP32 stand-in  -  pretends to be the hardware that is not built yet
=======================================================================

    AD8232 sensor -> ESP32 -> [ Flask server -> 1D CNN ] -> dashboard
    ^^^^^^^^^^^^^^^^^^^^^^^
    this script stands in for the two boxes on the left

Why this exists
---------------
The physical device is future work, but the SOFTWARE contract between the
device and the server is finished and testable today. This script speaks that
contract exactly as the firmware in firmware/ecg_esp32.ino does: it collects
1000 samples, wraps them in JSON with its own device id, POSTs them to
/api/predict, and prints what came back.

So the untested part of the system is only the analog front end. Everything
from the HTTP request onwards is exercised, end to end, every time you run
this - including the bit that matters most for the write-up: a reading
arriving from a named device and being filed against the right patient.

Where the signals come from
---------------------------
Real PTB-XL recordings out of ecg.db, taken ONLY from fold 10 - the fold the
model never saw during training or threshold tuning. Replaying training data
would produce beautiful numbers that mean nothing.

The signals are sent RAW (as stored). The server does the z-scoring, using the
same preprocess() the model was trained with, which is exactly what a real
device would rely on: an ESP32 should not have to know any statistics.

Usage
-----
    python esp32_simulator.py                          # 10 readings, 2s apart
    python esp32_simulator.py --n 30 --interval 0.5
    python esp32_simulator.py --device ESP32-001 --type mi
    python esp32_simulator.py --noise 0.05             # add sensor noise

First register the device to a patient, or the readings arrive unattributed:
    on the dashboard, "+ enrol" -> set Device id to the same string.
"""

import argparse
import sys
import time

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("This script needs the 'requests' package:\n    pip install requests")

import models


DEFAULT_URL = "http://127.0.0.1:5000"
TEST_FOLD = 10          # the unseen fold, same constant the app uses


def fetch_signals(n, want, seed):
    """Pull n random fold-10 recordings out of the database.

    Returns a list of (signal, true_label) pairs. The true label is only used
    to print how the system did - it is never sent to the server, because a
    real device has no idea what it is looking at.
    """
    label = {"mi": 1, "normal": 0}.get(want)      # None means "any"

    conn = models.connect()
    rng = np.random.default_rng(seed)
    signals = []
    for _ in range(n):
        row = models.random_recording(conn, fold=TEST_FOLD, label=label)
        if row is None:
            conn.close()
            sys.exit(f"No '{want}' recordings in fold {TEST_FOLD} - is ecg.db populated?")
        signals.append((models.blob_to_signal(row["signal"]), row["label"], row["id"]))
    conn.close()
    return signals


def add_sensor_noise(signal, level, rng):
    """Roughen a clean hospital signal to imitate a cheap electrode.

    The model was trained on clinical recordings, so this is the honest way to
    ask "what happens when the input is worse than the training data?" - a
    limitation the thesis names explicitly. Two effects, both real:

      baseline wander  slow drift from the patient breathing and moving
      gaussian noise   electrical hash from the amplifier and mains
    """
    n = len(signal)
    t = np.arange(n) / 100.0                     # seconds, at 100 Hz
    wander = level * 3 * np.sin(2 * np.pi * rng.uniform(0.15, 0.5) * t)
    hiss = rng.normal(0, level, n)
    return signal + wander.astype(np.float32) + hiss.astype(np.float32)


def post_reading(url, signal, device_id, mode, timeout):
    """Send one reading, exactly as the firmware does. Returns the parsed reply."""
    payload = {
        # .tolist() because numpy floats are not JSON-serialisable.
        "signal": [float(v) for v in signal],
        "mode": mode,
        "device_id": device_id,
    }
    r = requests.post(f"{url}/api/predict", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main():
    p = argparse.ArgumentParser(description="Pretend to be an ESP32 sending ECG readings.")
    p.add_argument("--url", default=DEFAULT_URL, help="where the Flask app is listening")
    p.add_argument("--device", default="ESP32-001", help="this device's id")
    p.add_argument("--n", type=int, default=10, help="how many readings to send")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between readings")
    p.add_argument("--mode", default="high_recall",
                   choices=["default", "best_f1", "high_recall"])
    p.add_argument("--type", default="any", choices=["any", "mi", "normal"],
                   help="what kind of recording to replay")
    p.add_argument("--noise", type=float, default=0.0,
                   help="sensor noise level, e.g. 0.05; 0 = clean hospital signal")
    p.add_argument("--seed", type=int, default=None, help="fix the choice of recordings")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Fail early and clearly if the server is not up - much friendlier than a
    # connection traceback ten seconds into a demo.
    try:
        health = requests.get(f"{args.url}/api/health", timeout=10).json()
    except requests.RequestException as e:
        sys.exit(f"Cannot reach the server at {args.url}\n  {e}\n\n"
                 f"Start it first:  python app.py")

    print("=" * 66)
    print(f"  ESP32 SIMULATOR   device '{args.device}'  ->  {args.url}")
    print("=" * 66)
    print(f"  server model    : {health.get('model')}")
    print(f"  operating point : {args.mode} (threshold {health['thresholds'][args.mode]})")
    print(f"  sending         : {args.n} reading(s), {args.interval}s apart, "
          f"type={args.type}")
    print(f"  sensor noise    : {args.noise if args.noise else 'none (clean signal)'}")
    print()

    signals = fetch_signals(args.n, args.type, args.seed)

    n_flagged = n_correct = 0
    latencies = []
    attributed = None

    for i, (signal, truth, rec_id) in enumerate(signals, 1):
        if args.noise:
            signal = add_sensor_noise(signal, args.noise, rng)

        started = time.perf_counter()
        try:
            reply = post_reading(args.url, signal, args.device, args.mode, timeout=30)
        except requests.RequestException as e:
            print(f"  [{i:3d}/{args.n}]  FAILED to send: {e}")
            continue
        round_trip_ms = (time.perf_counter() - started) * 1000

        latencies.append(round_trip_ms)
        n_flagged += reply["prediction"]
        n_correct += int(reply["prediction"] == truth)

        # The server tells us who it filed this under. Report it once.
        if attributed is None:
            attributed = reply.get("patient_name") or "(nobody)"
            if reply.get("warning"):
                print(f"  !! {reply['warning']}\n")

        verdict = "MI" if reply["prediction"] else "normal"
        actual = "MI" if truth else "normal"
        mark = "ok " if reply["prediction"] == truth else "MISS" if truth else "fa  "

        print(f"  [{i:3d}/{args.n}]  rec {rec_id:>6}  "
              f"p={reply['probability']:.4f}  said {verdict:<6}  "
              f"truly {actual:<6}  {mark}  "
              f"{reply['inference_ms']:5.1f} ms model / {round_trip_ms:6.1f} ms round trip")

        if i < len(signals):
            time.sleep(args.interval)

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    sent = len(latencies)
    print()
    print("-" * 66)
    if not sent:
        print("  Nothing was sent successfully.")
        return

    print(f"  sent            : {sent} reading(s)")
    print(f"  filed under     : {attributed}")
    print(f"  flagged as MI   : {n_flagged}")
    print(f"  agreed with the dataset label: {n_correct}/{sent} "
          f"({100 * n_correct / sent:.1f}%)")
    print(f"  round trip      : mean {np.mean(latencies):.1f} ms, "
          f"min {np.min(latencies):.1f}, max {np.max(latencies):.1f}")
    print()
    print("  Note: this accuracy is over a small random sample, so it will")
    print("  bounce around. The thesis figure is the full fold-10 evaluation")
    print("  in test_results.txt, not this number.")
    print("-" * 66)


if __name__ == "__main__":
    main()
