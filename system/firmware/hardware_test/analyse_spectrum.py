"""
Where does the no-electrode signal come from?

With the AD8232 powered and no electrodes on a person, the recorded ADC trace
alternates rail to rail on almost every sample under a slow beat envelope. A
tone just off 50 Hz, sampled at exactly 100 Hz, does exactly that: it folds to
near the 50 Hz Nyquist limit, and multiplying by (-1)^n demodulates it to a
slow tone at |f - 50| Hz. This script measures those numbers from a CSV saved
by ecg_serial_monitor.py --save.

    python analyse_spectrum.py ../../recordings/REC-....csv out.json
"""
import csv
import json
import sys

import numpy as np


def main(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    x = np.array([int(r["ecg_adc"]) for r in rows], float)
    lo = np.array([int(r["lead_off"]) for r in rows])
    n = np.arange(len(x))

    fx = np.fft.rfftfreq(len(x), 0.01)
    px = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
    kx = np.argmax(px[1:]) + 1

    y = (x - x.mean()) * (-1.0) ** n
    f = np.fft.rfftfreq(len(y), 0.01)
    p = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    k = np.argmax(p[1:]) + 1

    d = np.sign(np.diff(x))
    res = {
        "csv": csv_path.replace("\\", "/").split("/")[-1],
        "samples": len(x),
        "dominant_freq_of_raw_signal_hz": round(float(fx[kx]), 3),
        "share_of_consecutive_diffs_that_alternate_sign": round(float(np.mean(d[1:] * d[:-1] < 0)), 4),
        "beat_freq_after_(-1)^n_demodulation_hz": round(float(f[k]), 3),
        "implied_interference_freq_hz": [round(50 - float(f[k]), 3), round(50 + float(f[k]), 3)],
        "lead_off_vs_adc_high_correlation": round(float(np.corrcoef(lo, x > 2048)[0, 1]), 3),
    }
    print(json.dumps(res, indent=2))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
