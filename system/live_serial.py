"""
Live USB-serial ECG session for the web application
====================================================

    AD8232 -> ESP32 --USB serial--> [ this module ] -> app.py -> 1D CNN

The desktop monitor (`ecg_serial_monitor.py`) already knows how to read the
device, keep a bounded buffer, judge signal quality and assemble ten-second
windows. This module does not repeat any of that: it holds one long-lived
session built from those same classes, so the browser and the desktop monitor
cannot disagree about what counts as a clean window.

One session at a time, because there is one serial port. Starting a session
that is already running is not an error; it reports the running one.

The web layer supplies the scoring function, so this module never imports the
application, the model or the database. That keeps the dependency pointing one
way and lets the tests drive it with a stub.
"""

import threading
import queue
import time

import ecg_serial_monitor as esm


def available_ports():
    """Serial ports the host can see, ESP32 boards first."""
    from serial.tools import list_ports
    out = []
    for p in list_ports.comports():
        out.append({
            "device": p.device,
            "description": p.description or "",
            "is_esp32": (p.vid, p.pid) in esm.ESP32_USB_IDS,
        })
    out.sort(key=lambda d: (not d["is_esp32"], d["device"]))
    return out


class LiveSession:
    """One serial acquisition running behind the web application."""

    def __init__(self, score_window=None, mode="high_recall"):
        # score_window(values, mode) -> dict describing the verdict. Injected by
        # app.py; a session with no scorer still streams the waveform.
        self.score_window = score_window
        self.mode = mode
        self.rx = None
        self.window_queue = None
        self.scorer = None
        self.last_result = None
        self.error = ""
        self.started_at = None
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    @property
    def running(self):
        return self.rx is not None and self.rx.is_alive()

    def start(self, port=None, baud=115200, save_csv=False):
        with self._lock:
            if self.running:
                return {"running": True, "port": self.rx.port,
                        "already_running": True}
            port = port or esm.find_esp32_port()
            if not port:
                self.error = "no ESP32 (CP210x) serial port found"
                return {"running": False, "error": self.error}

            csv_path = None
            if save_csv:
                import datetime
                import os
                rec = f"REC-{datetime.datetime.now():%Y%m%d-%H%M%S}"
                csv_path = os.path.join(esm.RECORDINGS_DIR, rec + ".csv")

            self.window_queue = queue.Queue()
            self.rx = esm.SerialReceiver(port, baud, csv_path=csv_path,
                                         window_queue=self.window_queue)
            self.error = ""
            self.last_result = None
            self.rx.start()
            self.started_at = time.time()
            self.scorer = threading.Thread(target=self._score_loop, daemon=True)
            self.scorer.start()
            return {"running": True, "port": port, "csv": csv_path}

    def stop(self):
        with self._lock:
            if self.rx is not None:
                self.rx.stop_event.set()
                self.rx.join(timeout=3)
            if self.window_queue is not None:
                self.window_queue.put(None)      # release the scorer
            self.rx = None
            self.scorer = None
            return {"running": False}

    # -- scoring ---------------------------------------------------------
    def _score_loop(self):
        """Score each clean window. A failure here must not stop acquisition."""
        q = self.window_queue
        while True:
            window = q.get()
            if window is None:
                return
            if self.score_window is None:
                continue
            try:
                result = dict(self.score_window(window, self.mode))
            except Exception as exc:                  # noqa: BLE001 - reported
                result = {"error": f"scoring failed: {exc}"}
            result["measured_rate_hz"] = round(window.measured_rate_hz, 2)
            result["bpm"] = window.quality.get("bpm")
            result["beat_strength"] = window.quality.get("beat_strength")
            result["at"] = time.strftime("%H:%M:%S")
            self.last_result = result

    # -- what the browser polls -----------------------------------------
    def snapshot(self, trace_points=1000):
        """Current state, small enough to poll several times a second."""
        if not self.running:
            return {"running": False, "error": self.error,
                    "disclaimer": esm.DISCLAIMER}

        rx = self.rx
        adc, lead = rx.snapshot()
        trace = adc[-trace_points:]
        lead_off = rx.lead_off_now()
        valid, reason = esm.diagnose(adc[-esm.FS_HZ * 2:], lead_off,
                                     rx.connected, long_adc=adc)
        bpm, strength = esm.detect_beat(adc)
        return {
            "running": True,
            "port": rx.port,
            "connected": rx.connected,
            "link_error": rx.error,
            "lead_off": bool(lead_off),
            "signal_ok": bool(valid),
            "diagnosis": reason,
            "bpm": None if bpm is None else round(bpm),
            "beat_strength": round(strength, 2),
            "samples": rx.n_samples,
            "malformed": rx.n_malformed,
            "sequence_gaps": rx.n_seq_gaps,
            "rate_device_hz": _clean(rx.device_rate_hz()),
            "rate_host_hz": _clean(rx.pc_rate_hz()),
            "window_progress": rx.windows.progress,
            "window_length": esm.WINDOW_LEN,
            "windows_done": rx.windows.completed,
            "windows_discarded": dict(rx.windows.rejected),
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "trace": [int(v) for v in trace],
            "lead_off_flags": [bool(v) for v in lead[-trace_points:]],
            "last_result": self.last_result,
            "disclaimer": esm.DISCLAIMER,
        }


def _clean(value):
    """NaN is not valid JSON; report it as null instead."""
    try:
        if value != value:          # NaN
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None
