"""
Live ECG monitor  -  PC side of the USB-serial acquisition path
===============================================================

    AD8232 -> ESP32 --USB serial--> [ this file ] -> model.py (same 1D CNN)

Reads the sample stream from firmware/ecg_esp32_serial/ecg_esp32_serial.ino,
plots it live, shows lead-off status and the measured sampling rate, and can
save every sample to CSV. With --predict it also cuts the stream into
10-second windows and scores each one with the SAME model and the SAME
preprocess() the server uses (model.load_trained_model /
model.predict_probability). Nothing here re-implements preprocessing.

Stream format, one line per sample, lines starting with '#' are status text:
    seq,t_us,ecg_adc,lo_plus,lo_minus

Usage
    python ecg_serial_monitor.py                     # auto-detect port, live plot
    python ecg_serial_monitor.py --port COM7 --save  # also write system/recordings/REC-*.csv
    python ecg_serial_monitor.py --predict           # + experimental MI screening
    python ecg_serial_monitor.py --headless --duration 60 --report r.json --snapshot s.png

The screening output is a research result, not a clinical diagnosis.
"""

import argparse
import collections
import csv
import datetime
import json
import os
import queue
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(HERE, "recordings")

# Must match the firmware and the model (model.SAMPLE_RATE_HZ / INPUT_LENGTH).
# Not imported from model.py, because importing it loads TensorFlow and the
# monitor must start instantly when no prediction is wanted.
FS_HZ = 100
WINDOW_LEN = 1000
ADC_MAX = 4095

# CP210x USB-UART bridge on the ESP32 DevKit (Windows: "Silicon Labs CP210x").
ESP32_USB_IDS = {(0x10C4, 0xEA60)}

DISCLAIMER = "Experimental research screening result - not a clinical diagnosis."

_SAMPLE_RE = re.compile(r"^(\d+),(\d+),(\d+),([01]),([01])$")


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    seq: int
    t_us: int
    adc: int
    lo_plus: int
    lo_minus: int
    pc_time: float = 0.0

    @property
    def lead_off(self):
        return bool(self.lo_plus or self.lo_minus)


def parse_line(line):
    """Classify one line from the device.

    Returns ("sample", Sample), ("status", text), ("text", text),
    ("malformed", text), or None for a blank line. "text" is anything that is
    not trying to be a sample - chiefly the ESP32 boot ROM banner, printed
    whenever opening the port resets the board. "malformed" is a line that
    starts like a sample but is corrupted. Only well-formed samples are ever
    plotted or windowed.
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith("#"):
        return ("status", line[1:].strip())
    m = _SAMPLE_RE.match(line)
    if not m:
        return ("malformed" if line[0].isdigit() else "text", line)
    seq, t_us, adc, lop, lom = (int(g) for g in m.groups())
    if adc > ADC_MAX:
        return ("malformed", line)
    return ("sample", Sample(seq, t_us, adc, lop, lom))


# ---------------------------------------------------------------------------
# 2. Signal quality
# ---------------------------------------------------------------------------
def signal_quality(adc):
    """Simple checks on raw ADC counts. Not a clinical quality index.

    rail_fraction: share of samples pinned at either end of the ADC range,
    which means the AD8232 output is saturated (typical with a loose or
    floating electrode) and the waveform is not an ECG.
    """
    a = np.asarray(adc, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    rail = np.mean((a <= 5) | (a >= ADC_MAX - 5))
    return {
        "n": int(a.size),
        "min": int(a.min()), "max": int(a.max()),
        "mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1),
        "rail_fraction": round(float(rail), 4),
        "saturated": bool(rail > 0.01),
        "flat": bool(a.std() < 5),
    }


def detect_beat(adc, fs_hz=FS_HZ):
    """Is there a plausible heartbeat here? Returns (bpm, strength 0-1).

    An acquisition gate, not part of the research preprocessing: the model is
    still given the raw window exactly as before. It exists because the other
    quality checks pass on a signal that is electrically healthy but contains
    no ECG at all - on 2026-09-21, 50 Hz interference with no QRS complexes was
    scored "MI PATTERN, 48%".

    QRS detection in the style of Pan-Tompkins, reduced to what 100 Hz and a
    ten-second window need: average adjacent samples (which also removes the
    50 Hz component sitting exactly at Nyquist), subtract a half-second moving
    average to drop baseline wander, differentiate, square, and smooth into an
    energy signal in which each QRS is a bump. Peaks above a fraction of the
    99.5th percentile, no closer together than 250 ms, are beats.

    strength is rhythm regularity: the share of RR intervals within 30% of the
    median. Noise produces irregular spacing and scores low; a real ECG, even a
    noisy one, repeats. An earlier autocorrelation version scored a genuine
    81 bpm recording at 0.21 and is why this was rewritten.
    """
    x = np.asarray(adc, dtype=np.float64)
    if x.size < 3 * fs_hz:                       # too short to judge
        return None, 0.0
    x = np.convolve(x, [0.5, 0.5], mode="valid")
    k = max(int(0.5 * fs_hz), 3)
    x = x - np.convolve(x, np.ones(k) / k, mode="same")
    x = x[k:x.size - k] if x.size > 3 * k else x
    if x.size < 2 * fs_hz:
        return None, 0.0

    # A QRS has to stand out from the baseline, in absolute terms and against
    # the noise around it; otherwise "peaks" are just the loudest noise.
    amplitude = float(np.percentile(np.abs(x), 99.5))
    noise = float(np.median(np.abs(x))) + 1e-9
    if amplitude < 20 or amplitude / noise < 3:
        return None, 0.0

    d = np.diff(x)
    energy = np.convolve(d * d, np.ones(max(int(0.12 * fs_hz), 3)), mode="same")
    threshold = 0.35 * float(np.percentile(energy, 99.5))
    if threshold <= 0:
        return None, 0.0

    refractory = int(0.25 * fs_hz)               # no two beats inside 250 ms
    peaks, last = [], -refractory
    for i in range(1, energy.size - 1):
        if (energy[i] > threshold and energy[i] >= energy[i - 1]
                and energy[i] > energy[i + 1] and i - last > refractory):
            peaks.append(i)
            last = i
    if len(peaks) < 3:
        return None, 0.0

    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / fs_hz
    median_rr = float(np.median(rr))
    if not 0.33 <= median_rr <= 1.5:             # outside 40-180 bpm
        return 60.0 / median_rr if median_rr > 0 else None, 0.0
    consistency = float(np.mean(np.abs(rr - median_rr) / median_rr < 0.30))
    return 60.0 / median_rr, consistency


BEAT_MIN_STRENGTH = 0.60   # share of regular RR intervals


def diagnose(adc, lead_off, link_ok=True, long_adc=None):
    """Why is there (or is there not) a usable signal? Returns (valid, message).

    Ordered so the most basic fault is reported first. An ECG input stuck at
    0 V means the AD8232 is unpowered or OUT is not on the ADC pin; that is a
    wiring fault, not an electrode fault, and saying so saves a lot of guessing.
    """
    if not link_ok:
        return False, "SERIAL DISCONNECTED - check the USB cable / port"
    a = np.asarray(adc, dtype=np.float64)
    if a.size < FS_HZ // 2:
        return False, "Waiting for data ..."
    # A pinned input is two different faults that look identical in the samples
    # alone, and naming the wrong one sends someone to check the wiring when
    # the electrodes are the problem. The lead-off pins tell them apart: if the
    # AD8232 is powered and says the electrodes are on, the amplifier is simply
    # railed - which is what happened on 2026-09-23, when a good ECG drifted to
    # the bottom of the ADC range over about thirty seconds.
    if a.max() <= 5:
        if lead_off:
            return False, ("ECG input reads 0 V - check AD8232 power (3.3V and\n"
                           "GND) and the OUT -> VP (GPIO36) wire")
        return False, ("Amplifier output is stuck at the bottom of its range.\n"
                       "Re-seat the electrodes (fresh gel, check the RL pad)\n"
                       "and keep still while the baseline settles")
    if a.min() >= ADC_MAX - 5:
        if lead_off:
            return False, "ECG input stuck at 3.3 V - check the OUT -> VP wire"
        return False, ("Amplifier output is stuck at the top of its range.\n"
                       "Re-seat the electrodes and keep still while the\n"
                       "baseline settles")
    if lead_off:
        return False, ("LEAD OFF - electrodes not on skin, cable unplugged,\n"
                       "or LO+/LO- not wired")
    q = signal_quality(a)
    if q["saturated"]:
        return False, "Signal saturated - poor electrode contact or mains interference"
    if q["flat"]:
        return False, "Signal flat - check electrodes and cable"
    if long_adc is not None:
        bpm, strength = detect_beat(long_adc)
        if strength < BEAT_MIN_STRENGTH:
            return False, ("No heartbeat found - signal present but not an ECG.\n"
                           "Check electrode placement, skin contact and gel")
        return True, f"Signal available - beat {bpm:.0f} bpm (strength {strength:.2f})"
    return True, "Signal available"


# ---------------------------------------------------------------------------
# 3. 10-second window preparation
# ---------------------------------------------------------------------------
@dataclass
class Window:
    values: np.ndarray          # raw ADC counts, float32, length WINDOW_LEN
    t_us: np.ndarray
    first_seq: int
    measured_rate_hz: float
    quality: dict


@dataclass
class WindowBuilder:
    """Collects WINDOW_LEN contiguous, lead-on, gap-free samples.

    Any of these throws the partial window away and starts again:
      - either lead-off output HIGH       (electrode off: not an ECG)
      - a missing sequence number         (a sample was lost on the link)
      - a timestamp jump > 1.5 periods    (the device missed a tick)
    A finished window is also refused if the ADC was saturated, because a
    rail-to-rail signal z-scores into something that looks like data.
    Consecutive windows do not overlap.
    """
    length: int = WINDOW_LEN
    fs_hz: int = FS_HZ
    max_rail_fraction: float = 0.01
    require_beat: bool = True
    _vals: list = field(default_factory=list)
    _ts: list = field(default_factory=list)
    _first_seq: int = -1
    _last: Sample = None
    rejected: collections.Counter = field(default_factory=collections.Counter)
    completed: int = 0

    @property
    def progress(self):
        return len(self._vals)

    def reset(self, reason=None):
        if reason and self._vals:
            self.rejected[reason] += 1
        self._vals, self._ts, self._first_seq = [], [], -1

    def add(self, s):
        period_us = 1e6 / self.fs_hz
        last, self._last = self._last, s
        if s.lead_off:
            self.reset("lead_off")
            return None
        if last is not None and self._vals:
            if s.seq != last.seq + 1:
                self.reset("sequence_gap")
            elif (s.t_us - last.t_us) > 1.5 * period_us:
                self.reset("timing_gap")
        if not self._vals:
            self._first_seq = s.seq
        self._vals.append(s.adc)
        self._ts.append(s.t_us)
        if len(self._vals) < self.length:
            return None

        values = np.asarray(self._vals, dtype=np.float32)
        ts = np.asarray(self._ts, dtype=np.int64)
        span = (ts[-1] - ts[0]) / 1e6
        rate = (self.length - 1) / span if span > 0 else float("nan")
        q = signal_quality(values)
        first = self._first_seq
        self.reset()
        if q["rail_fraction"] > self.max_rail_fraction:
            self.rejected["saturated"] += 1
            return None
        bpm, strength = detect_beat(values, self.fs_hz)
        q["bpm"] = None if bpm is None else round(bpm, 1)
        q["beat_strength"] = round(strength, 3)
        if self.require_beat and strength < BEAT_MIN_STRENGTH:
            self.rejected["no_heartbeat"] += 1
            return None
        self.completed += 1
        return Window(values, ts, first, rate, q)


# ---------------------------------------------------------------------------
# 4. Serial receiver
# ---------------------------------------------------------------------------
def find_esp32_port():
    """First serial port whose USB VID/PID is the ESP32 DevKit's CP210x."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        if (p.vid, p.pid) in ESP32_USB_IDS:
            return p.device
    return None


class SerialReceiver(threading.Thread):
    """Reads the port, parses lines, keeps a bounded buffer, feeds windows.

    Everything the display needs is held in fixed-size deques, so memory does
    not grow however long it runs. A lost port is reported and reopened.
    """

    def __init__(self, port, baud=115200, buffer_s=10, csv_path=None,
                 window_queue=None, raw_log=None):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        n = FS_HZ * buffer_s
        self.adc = collections.deque(maxlen=n)
        self.lead = collections.deque(maxlen=n)          # 1 = lead off
        self.t_dev = collections.deque(maxlen=n)         # device micros()
        self.t_pc = collections.deque(maxlen=n)          # host arrival time
        self.status_lines = collections.deque(maxlen=5)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.windows = WindowBuilder()
        self.window_queue = window_queue
        self.csv_path = csv_path
        self.raw_log = raw_log
        self.connected = False
        self.error = ""
        self.n_samples = self.n_status = self.n_malformed = self.n_seq_gaps = 0
        self.n_text = 0
        self.first_sample_pc = self.last_sample_pc = None
        self.n_lo_plus = self.n_lo_minus = self.n_lead_off = 0
        self._last_seq = None
        self._dt_sum = self._dt_sq = 0.0
        self._dt_n = 0
        self._dt_min = self._dt_max = None
        self._all_adc_stats = [0, 0.0, 0.0, None, None, 0]  # n, sum, sumsq, min, max, rail
        self.t_start = None

    # -- stats -----------------------------------------------------------
    def _account(self, s):
        self.n_samples += 1
        self.n_lo_plus += s.lo_plus
        self.n_lo_minus += s.lo_minus
        self.n_lead_off += s.lead_off
        if self._last_seq is not None:
            if s.seq != self._last_seq + 1:
                self.n_seq_gaps += 1
            elif self.t_dev:
                dt = s.t_us - self.t_dev[-1]
                self._dt_sum += dt
                self._dt_sq += dt * dt
                self._dt_n += 1
                self._dt_min = dt if self._dt_min is None else min(self._dt_min, dt)
                self._dt_max = dt if self._dt_max is None else max(self._dt_max, dt)
        self._last_seq = s.seq
        a = self._all_adc_stats
        a[0] += 1; a[1] += s.adc; a[2] += s.adc * s.adc
        a[3] = s.adc if a[3] is None else min(a[3], s.adc)
        a[4] = s.adc if a[4] is None else max(a[4], s.adc)
        a[5] += (s.adc <= 5 or s.adc >= ADC_MAX - 5)

    def lead_off_now(self, hold_samples=FS_HZ // 2):
        """Lead-off if either LO pin was HIGH at any point in the last 0.5 s.

        A per-sample flag flickers: with floating electrodes the LO outputs
        were seen toggling on alternate samples. Holding for half a second
        gives a stable CONNECTED / LEAD OFF indication.
        """
        with self.lock:
            recent = list(self.lead)[-hold_samples:]
        return (not recent) or any(recent)

    def device_rate_hz(self, n=200):
        with self.lock:
            t = list(self.t_dev)[-n:]
        if len(t) < 2 or t[-1] <= t[0]:
            return float("nan")
        return (len(t) - 1) * 1e6 / (t[-1] - t[0])

    def pc_rate_hz(self):
        with self.lock:
            t = list(self.t_pc)
        if len(t) < 2 or t[-1] <= t[0]:
            return float("nan")
        return (len(t) - 1) / (t[-1] - t[0])

    def snapshot(self):
        with self.lock:
            return np.array(self.adc, dtype=float), np.array(self.lead, dtype=bool)

    def report(self):
        n, s1, s2, mn, mx, rail = self._all_adc_stats
        elapsed = time.perf_counter() - self.t_start if self.t_start else 0.0
        dt_mean = self._dt_sum / self._dt_n if self._dt_n else float("nan")
        dt_sd = (max(self._dt_sq / self._dt_n - dt_mean ** 2, 0.0) ** 0.5
                 if self._dt_n else float("nan"))
        return {
            "port": self.port, "baud": self.baud,
            "elapsed_s": round(elapsed, 2),
            "samples": self.n_samples,
            "status_lines": self.n_status,
            "boot_or_other_text_lines": self.n_text,
            "malformed_lines": self.n_malformed,
            "sequence_gaps": self.n_seq_gaps,
            "device_dt_us": {"mean": round(dt_mean, 3), "sd": round(dt_sd, 3),
                             "min": self._dt_min, "max": self._dt_max},
            "device_rate_hz": round(1e6 / dt_mean, 4) if self._dt_n else None,
            # (n-1) intervals between first and last arrival: host-clock check
            # of the device rate, including any USB buffering at the ends.
            "pc_arrival_rate_hz": (
                round((self.n_samples - 1) / (self.last_sample_pc - self.first_sample_pc), 3)
                if self.n_samples > 1 and self.last_sample_pc > self.first_sample_pc
                else None),
            "pc_arrival_span_s": (round(self.last_sample_pc - self.first_sample_pc, 3)
                                  if self.first_sample_pc is not None else None),
            "lead_off_fraction": round(self.n_lead_off / n, 4) if n else None,
            "lo_plus_high_fraction": round(self.n_lo_plus / n, 4) if n else None,
            "lo_minus_high_fraction": round(self.n_lo_minus / n, 4) if n else None,
            "adc": ({"min": mn, "max": mx, "mean": round(s1 / n, 1),
                     "std": round(max(s2 / n - (s1 / n) ** 2, 0) ** 0.5, 1),
                     "rail_fraction": round(rail / n, 4)} if n else None),
            "windows_completed": self.windows.completed,
            # How many times a partly filled window was thrown away, by reason
            # ("saturated" = a full 1000 samples refused for ADC saturation).
            "windows_discarded": dict(self.windows.rejected),
            "diagnosis": diagnose(list(self.adc)[-2 * FS_HZ:], self.lead_off_now(),
                                  self.connected, long_adc=list(self.adc))[1],
            "last_status_lines": list(self.status_lines),
            "error": self.error,
        }

    # -- main loop -------------------------------------------------------
    def _handle_line(self, raw_line, writer, csv_file, raw_log):
        """One line from the device: count it, buffer it, maybe finish a window."""
        now = time.perf_counter() - self.t_start
        text = raw_line.decode("utf-8", errors="replace")
        if raw_log:
            raw_log.write(f"[{now:8.3f}s] {text.rstrip()}\n")

        parsed = parse_line(text)
        if parsed is None:
            return
        kind, val = parsed
        if kind == "status":
            self.n_status += 1
            self.status_lines.append(val)
            return
        if kind == "malformed":
            self.n_malformed += 1
            return
        if kind == "text":
            self.n_text += 1
            return

        if self.first_sample_pc is None:
            self.first_sample_pc = now
        self.last_sample_pc = now
        val.pc_time = now
        with self.lock:
            self._account(val)
            self.adc.append(val.adc)
            self.lead.append(val.lead_off)
            self.t_dev.append(val.t_us)
            self.t_pc.append(now)

        if writer:
            writer.writerow([f"{now:.4f}", val.seq, val.t_us, val.adc,
                             val.lo_plus, val.lo_minus, int(val.lead_off)])
            if self.n_samples % FS_HZ == 0:
                csv_file.flush()

        window = self.windows.add(val)
        if window is not None and self.window_queue is not None:
            self.window_queue.put(window)

    def run(self):
        import serial
        csv_file = writer = None
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            writer.writerow(["pc_time_s", "seq", "t_us", "ecg_adc",
                             "lo_plus", "lo_minus", "lead_off"])
        raw_log = open(self.raw_log, "w", encoding="utf-8") if self.raw_log else None
        self.t_start = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                try:
                    # Open with DTR and RTS de-asserted. On the ESP32 DevKit
                    # those lines drive EN and GPIO0 through the auto-reset
                    # circuit, so a default open() restarts the board. Each
                    # restart re-samples the strapping pins and loses samples;
                    # the stream should simply be joined where it is.
                    ser = serial.Serial()
                    ser.port, ser.baudrate, ser.timeout = self.port, self.baud, 0.5
                    ser.dtr = False
                    ser.rts = False
                    ser.open()
                    # A large driver-side buffer. Model inference in the same
                    # process can hold this thread off for a second or more,
                    # and the default buffer overflows in that time: 234 of
                    # 1,480 samples were lost that way when the web application
                    # scored its first window (2026-09-23).
                    try:
                        ser.set_buffer_size(rx_size=1 << 16)
                    except (AttributeError, OSError):
                        pass                       # not supported off Windows
                except serial.SerialException as e:
                    self.connected, self.error = False, f"cannot open {self.port}: {e}"
                    time.sleep(1.0)
                    continue

                self.connected, self.error = True, ""
                pending = b""
                try:
                    while not self.stop_event.is_set():
                        # Drain everything waiting in one read. Reading a line
                        # at a time cannot catch up after a pause, because each
                        # call carries its own overhead while samples keep
                        # arriving at 100 per second.
                        chunk = ser.read(max(ser.in_waiting, 1))
                        if not chunk:
                            continue
                        pending += chunk
                        lines = pending.split(b"\n")
                        pending = lines.pop()      # the tail may be incomplete
                        for line in lines:
                            self._handle_line(line, writer, csv_file, raw_log)
                except serial.SerialException as e:
                    self.connected, self.error = False, f"serial error: {e}"
                finally:
                    ser.close()
        finally:
            if csv_file:
                csv_file.close()
            if raw_log:
                raw_log.close()


# ---------------------------------------------------------------------------
# 5. Model inference (same model, same preprocessing as the server)
# ---------------------------------------------------------------------------
class Predictor(threading.Thread):
    """Scores each finished window in the background.

    Loads the served model with model.load_trained_model(), which picks the
    ensemble when present and sets the band-pass flag from that model's own
    config. Thresholds come from the same config, exactly as app.py builds
    them. A window is scored on its raw ADC counts: model.preprocess() z-scores
    it, so the linear ADC-to-volts scale does not matter.
    """

    def __init__(self, window_queue, mode="high_recall", post_url=None):
        super().__init__(daemon=True)
        self.window_queue, self.mode, self.post_url = window_queue, mode, post_url
        self.results = []
        self.latest = None
        self.state = "loading model ..."
        self.model_name = ""
        self.threshold = None

    def run(self):
        import model
        loaded, config = model.load_trained_model()
        thresholds = {
            "default": config["threshold_default"],
            "best_f1": config["threshold_best_f1"],
            "high_recall": config["threshold_high_recall"],
        }
        self.threshold = thresholds[self.mode]
        self.model_name = model.describe(loaded)
        self.state = "waiting for a clean 10-s window"
        while True:
            w = self.window_queue.get()
            if w is None:
                return
            t0 = time.perf_counter()
            batch_shape = model.preprocess(w.values).shape
            probability, was_warmup = model.predict_probability(loaded, w.values)
            ms = (time.perf_counter() - t0) * 1000
            prediction = 1 if probability >= self.threshold else 0
            result = {
                "first_seq": w.first_seq,
                "measured_rate_hz": round(w.measured_rate_hz, 3),
                "input_shape": list(batch_shape),
                "probability": round(probability, 4),
                "threshold": round(self.threshold, 4),
                "mode": self.mode,
                "prediction": prediction,
                "label": "MI PATTERN" if prediction else "NORMAL",
                "inference_ms": round(ms, 1),
                "warmup": was_warmup,
                "quality": w.quality,
            }
            if self.post_url:
                result["server"] = post_window(self.post_url, w.values, self.mode)
            self.results.append(result)
            self.latest = result
            self.state = "scored"


def post_window(base_url, values, mode, device_id="ESP32-001"):
    """Forward a window to the existing /api/predict endpoint (optional).

    Same JSON contract the Wi-Fi firmware and esp32_simulator.py use, so the
    reading is stored in the predictions table with source='device'.
    """
    body = json.dumps({"signal": [float(v) for v in values], "mode": mode,
                       "device_id": device_id}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/api/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:                     # report, never crash the monitor
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 6. Live display
# ---------------------------------------------------------------------------
class LiveDisplay:
    """Scrolling 10-s raw waveform plus a status panel.

    Blitted: only the waveform lines and status text are redrawn each frame,
    not the axes or the window.
    """

    def __init__(self, rx, predictor=None, show_filtered=False, ylim=(0, ADC_MAX)):
        import matplotlib.pyplot as plt
        self.rx, self.predictor, self.show_filtered = rx, predictor, show_filtered
        n = rx.adc.maxlen
        self.t = (np.arange(n) - n) / FS_HZ          # -10 s .. 0 s
        rows = 3 if show_filtered else 2
        ratios = [3, 2, 1.3] if show_filtered else [3, 1.3]
        self.fig, axes = plt.subplots(rows, 1, figsize=(12, 9 if show_filtered else 6.5),
                                      gridspec_kw={"height_ratios": ratios})
        if getattr(self.fig.canvas, "manager", None):
            self.fig.canvas.manager.set_window_title("ECG serial monitor")
        self.ax = axes[0]
        self.ax.set_title("RAW ECG  (AD8232 OUT -> ESP32 GPIO36, 12-bit ADC counts, "
                          f"{FS_HZ} Hz)", loc="left", fontsize=11)
        self.ax.set_xlim(self.t[0], 0)
        self.ax.set_ylim(*ylim)
        self.ax.set_xlabel("time (s, 0 = newest sample)")
        self.ax.set_ylabel("ADC counts")
        self.ax.grid(alpha=0.3)
        # The raw trace is always drawn in full. Lead-off samples are marked
        # with ticks along the top rather than by masking the trace, because
        # with floating electrodes the LO pins flip on alternate samples and a
        # masked line would have no two neighbouring points left to join.
        self._lo_y = ylim[1] - 0.03 * (ylim[1] - ylim[0])
        (self.l_raw,) = self.ax.plot([], [], lw=0.9, color="#1f5fa8", label="raw ADC")
        (self.l_off,) = self.ax.plot([], [], ls="none", marker="|", ms=8,
                                     color="#d9534f", label="lead-off sample (not ECG)")
        self.ax.legend(loc="lower right", fontsize=8)
        # Shown over the trace whenever there is no valid ECG, so that noise
        # from an unplugged sensor or floating electrodes is never mistaken
        # for a heartbeat. The raw trace stays visible, greyed out.
        self.no_signal = self.ax.text(
            0.5, 0.5, "", transform=self.ax.transAxes, ha="center", va="center",
            fontsize=15, fontweight="bold", color="#b04a4a", zorder=5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b04a4a", alpha=0.9))
        self.artists = [self.l_raw, self.l_off, self.no_signal]

        if show_filtered:
            import model
            self._bandpass = model.bandpass
            self.axf = axes[1]
            self.axf.set_title("0.5-40 Hz band-pass - DISPLAY ONLY, not the model input",
                               loc="left", fontsize=10)
            self.axf.set_xlim(self.t[0], 0)
            self.axf.set_ylim(-600, 600)
            self.axf.grid(alpha=0.3)
            (self.l_f,) = self.axf.plot([], [], lw=1.0, color="#444")
            self.artists.append(self.l_f)

        axs = axes[-1]
        axs.axis("off")
        self.status = axs.text(0.0, 1.0, "", va="top", ha="left", family="monospace",
                               fontsize=10, transform=axs.transAxes)
        self.banner = axs.text(0.0, 0.0, DISCLAIMER if predictor else "",
                               va="bottom", ha="left", fontsize=9, color="#8a4b00",
                               transform=axs.transAxes)
        self.artists += [self.status, self.banner]
        self.fig.tight_layout()

    def status_text(self):
        rx = self.rx
        adc, lead = rx.snapshot()
        lead_off = rx.lead_off_now()
        q = signal_quality(adc[-FS_HZ * 2:])
        available, reason = diagnose(adc[-FS_HZ * 2:], lead_off, rx.connected,
                                     long_adc=adc)
        link = "CONNECTED" if rx.connected else f"DISCONNECTED  {rx.error}"
        rep_lo = ""
        if len(lead):
            rep_lo = f"  (lead-off in {100 * lead[-FS_HZ // 2:].mean():.0f}% of last 0.5 s)"
        lines = [
            f"PORT          : {rx.port}  {link}",
            f"ECG STATUS    : {'LEAD OFF' if lead_off else 'CONNECTED'}{rep_lo}",
            f"SIGNAL        : {'AVAILABLE' if available else 'NOT AVAILABLE'}"
            + ("" if available else "  - " + reason.replace(chr(10), " ")),
            f"Samples       : {rx.n_samples}   (malformed {rx.n_malformed}, "
            f"seq gaps {rx.n_seq_gaps}, boot/other text {rx.n_text})",
            f"Sampling rate : {rx.device_rate_hz():.2f} Hz device clock | "
            f"{rx.pc_rate_hz():.2f} Hz PC arrival",
            f"10-s window   : {rx.windows.progress}/{WINDOW_LEN} clean samples  "
            f"(done {rx.windows.completed}, rejected {sum(rx.windows.rejected.values())})",
        ]
        p = self.predictor
        if p is not None:
            if p.latest:
                r = p.latest
                lines.append(f"MI probability: {100 * r['probability']:.1f}%   "
                             f"threshold {r['threshold']} ({r['mode']})   "
                             f"screening: {r['label']}")
            else:
                lines.append(f"Model         : {p.state}")
        return "\n".join(lines)

    def update(self, _frame=None):
        adc, lead = self.rx.snapshot()
        n = len(adc)
        t = self.t[-n:] if n else self.t[:0]
        self.l_raw.set_data(t, adc)
        self.l_off.set_data(t[lead], np.full(int(lead.sum()), self._lo_y))
        valid, reason = diagnose(adc[-FS_HZ * 2:], self.rx.lead_off_now(),
                                 self.rx.connected, long_adc=adc)
        self.l_raw.set_color("#1f5fa8" if valid else "#b0b7bf")
        self.no_signal.set_text("" if valid else "NO VALID ECG\n" + reason)
        self.no_signal.set_visible(not valid)
        if self.show_filtered:
            if n > 30:
                self.l_f.set_data(t, self._bandpass(adc - adc.mean()))
            else:
                self.l_f.set_data([], [])
        self.status.set_text(self.status_text())
        return self.artists

    def run(self, interval_ms=50):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        self._anim = FuncAnimation(self.fig, self.update, interval=interval_ms,
                                   blit=True, cache_frame_data=False)
        plt.show()

    def save(self, path):
        self.update()
        self.fig.savefig(path, dpi=150)

    def close(self, snapshot=None):
        """Stop the animation first, then close: a frame fired after the
        window is gone would try to blit onto a destroyed canvas."""
        import matplotlib.pyplot as plt
        anim = getattr(self, "_anim", None)
        if anim is not None and anim.event_source is not None:
            anim.event_source.stop()
        if snapshot:
            self.save(snapshot)
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# 7. Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--port", help="serial port, e.g. COM7 (default: auto-detect CP210x)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--save", action="store_true",
                    help="write every sample to system/recordings/REC-<time>.csv")
    ap.add_argument("--predict", action="store_true",
                    help="score each clean 10-s window with the trained model")
    ap.add_argument("--mode", default="high_recall",
                    choices=["default", "best_f1", "high_recall"])
    ap.add_argument("--post-url", help="also forward windows to this server's /api/predict")
    ap.add_argument("--no-beat-check", action="store_true",
                    help="score a window even if no heartbeat is detected in it")
    ap.add_argument("--show-filtered", action="store_true",
                    help="add a display-only band-pass panel")
    ap.add_argument("--ylim", type=float, nargs=2, default=(0, ADC_MAX))
    ap.add_argument("--headless", action="store_true", help="no window; for tests")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N seconds (required with --headless)")
    ap.add_argument("--report", help="write a JSON summary here on exit")
    ap.add_argument("--snapshot", help="save the display to this PNG on exit")
    ap.add_argument("--raw-log", help="write every received line, timestamped")
    args = ap.parse_args()

    if args.headless:
        import matplotlib
        matplotlib.use("Agg")
        if not args.duration:
            ap.error("--headless needs --duration")

    port = args.port or find_esp32_port()
    if not port:
        ap.error("no ESP32 (CP210x) serial port found; pass --port")

    csv_path = None
    if args.save:
        # Anonymous recording id: a timestamp, never a name.
        rec_id = f"REC-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        csv_path = os.path.join(RECORDINGS_DIR, rec_id + ".csv")

    wq = queue.Queue() if args.predict else None
    rx = SerialReceiver(port, args.baud, csv_path=csv_path, window_queue=wq,
                        raw_log=args.raw_log)
    rx.windows.require_beat = not args.no_beat_check
    predictor = Predictor(wq, args.mode, args.post_url) if args.predict else None
    rx.start()
    if predictor:
        predictor.start()
    print(f"Reading {port} at {args.baud} baud" + (f", saving to {csv_path}" if csv_path else ""))

    display = None
    if not args.headless or args.snapshot:
        display = LiveDisplay(rx, predictor, args.show_filtered, tuple(args.ylim))

    try:
        if args.headless:
            time.sleep(args.duration)
        else:
            if args.duration:
                timer = display.fig.canvas.new_timer(interval=int(args.duration * 1000))
                timer.single_shot = True
                timer.add_callback(display.close, args.snapshot)
                timer.start()
            display.run()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop_event.set()
        rx.join(timeout=3)
        if display and args.snapshot and args.headless:
            display.save(args.snapshot)
        rep = rx.report()
        rep["csv"] = csv_path
        if predictor:
            if predictor.is_alive():
                wq.put(None)
                predictor.join(timeout=60)
            rep["model"] = predictor.model_name
            rep["predictions"] = predictor.results
            rep["disclaimer"] = DISCLAIMER
        print(json.dumps(rep, indent=2))
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
