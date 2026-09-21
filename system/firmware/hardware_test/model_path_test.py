"""
Model-input compatibility test for the USB-serial path (T11 / T12).

Part A - HARDWARE: replays a CSV recorded by ecg_serial_monitor.py --save
through the same WindowBuilder the live monitor uses, and reports how many
10-second windows it would hand to the model. With no electrodes on a person
the expected answer is zero: every window must be refused.

Part B - REPLAY (not hardware): pushes held-out PTB-XL fold-10 recordings from
ecg.db through WindowBuilder -> Predictor sample by sample, exactly as live
samples would arrive, and checks that the probability equals the one the
server path (model.predict_probability on the stored recording) gives. Equal
numbers mean the live path uses the same preprocessing and the same model.

    python model_path_test.py --csv ../../recordings/REC-....csv --n 3 --out result.json
"""
import argparse
import csv
import json
import os
import queue
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SYSTEM)

import ecg_serial_monitor as mon  # noqa: E402


def part_a(csv_path):
    wb = mon.WindowBuilder()
    n = 0
    windows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            s = mon.Sample(int(row["seq"]), int(row["t_us"]), int(row["ecg_adc"]),
                           int(row["lo_plus"]), int(row["lo_minus"]))
            w = wb.add(s)
            if w is not None:
                windows.append(w.first_seq)
    return {"source": "HARDWARE", "csv": os.path.basename(csv_path),
            "samples": n, "windows_passed_to_model": len(windows),
            "windows_discarded": dict(wb.rejected)}


def part_b(n_per_class):
    import model
    import models

    conn = models.connect()
    rows = []
    for label in (0, 1):
        rows += conn.execute(
            "SELECT id, signal, label FROM recordings WHERE fold = 10 AND label = ? "
            "ORDER BY id LIMIT ?", (label, n_per_class)).fetchall()
    conn.close()

    wq = queue.Queue()
    pred = mon.Predictor(wq, mode="high_recall")
    # PTB-XL values are millivolts, not 12-bit ADC counts, so the ADC-rail
    # saturation check does not apply (every |v| < 5 mV would look "railed").
    # max_rail_fraction=1.0 disables only that check; lead-off, sequence and
    # timing checks still run.
    wb = mon.WindowBuilder(max_rail_fraction=1.0)
    signals = []
    seq = 0
    for r in rows:
        sig = models.blob_to_signal(r["signal"])
        signals.append((r["id"], int(r["label"]), sig))
        for v in sig:                     # one sample at a time, like the stream
            w = wb.add(mon.Sample(seq, 1_000_000 + seq * 10_000, float(v), 0, 0))
            seq += 1
            if w is not None:
                wq.put(w)
    wq.put(None)
    pred.run()                            # same code the thread runs, in-line

    loaded, _ = model.load_trained_model()
    out = []
    for (rid, truth, sig), res in zip(signals, pred.results):
        direct, _ = model.predict_probability(loaded, sig)
        out.append({
            "recording_id": rid, "truth": "MI" if truth else "NORMAL",
            "monitor_path_probability": res["probability"],
            "server_path_probability": round(direct, 4),
            "identical": abs(res["probability"] - round(direct, 4)) < 1e-4,
            "threshold": res["threshold"], "mode": res["mode"],
            "screening": res["label"], "input_shape": res["input_shape"],
        })
    return {"source": "REPLAY of PTB-XL fold 10 (not hardware)",
            "model": pred.model_name, "windows_scored": len(pred.results),
            "results": out, "disclaimer": mon.DISCLAIMER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = {"part_a_hardware": part_a(a.csv), "part_b_replay": part_b(a.n)}
    print(json.dumps(result, indent=2))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
