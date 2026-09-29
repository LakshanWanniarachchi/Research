

import collections
import json
import threading
import time

import numpy as np

import ecg_serial_monitor as esm

FS_HZ = esm.FS_HZ
WINDOW_LEN = esm.WINDOW_LEN
ADC_MAX = esm.ADC_MAX

# A device is called offline if nothing has arrived for this long, regardless
# of what its last retained status message said: a device that was unplugged
# without a clean disconnect still claims to be online until the broker
# notices, and the dashboard should not repeat that claim.
SILENCE_TIMEOUT_S = 8.0

# How far the sequence counter may step backwards before it is read as a device
# reboot rather than as a redelivered chunk. A reboot restarts the count at
# zero, and a monitor that mistook that for duplicates would ignore the device
# for ever - seen for real on 2026-09-23 when the publisher was restarted.
RESTART_BACKSTEP = 3


class ChunkError(ValueError):
    """A chunk that cannot be trusted. Reported and counted, never guessed at."""


def parse_chunk(raw, expect_device=None):
    """Validate one `ecg/devices/<id>/data` payload.

    Every field is checked before a single sample is admitted, because a
    malformed chunk that is half-accepted corrupts the signal silently, and a
    corrupted signal is scored by the model just as readily as a real one.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ChunkError("chunk is not valid UTF-8") from exc
    try:
        body = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ChunkError("chunk is not valid JSON") from exc
    if not isinstance(body, dict):
        raise ChunkError("chunk must be a JSON object")

    device_id = body.get("device_id")
    if not isinstance(device_id, str) or not device_id:
        raise ChunkError("missing 'device_id'")
    if expect_device is not None and device_id != expect_device:
        raise ChunkError(f"device_id '{device_id}' does not match topic "
                         f"'{expect_device}'")

    for field in ("seq", "t_us", "n"):
        if not isinstance(body.get(field), int) or isinstance(body.get(field), bool):
            raise ChunkError(f"'{field}' must be an integer")

    fs = body.get("fs", FS_HZ)
    if fs != FS_HZ:
        raise ChunkError(f"chunk says {fs} Hz, this system expects {FS_HZ} Hz")

    adc = body.get("adc")
    if not isinstance(adc, list):
        raise ChunkError("missing 'adc' list")
    if len(adc) != body["n"]:
        raise ChunkError(f"'n' is {body['n']} but 'adc' holds {len(adc)}")
    if not adc:
        raise ChunkError("chunk carries no samples")
    for value in adc:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ChunkError("'adc' must contain only integers")
        if not 0 <= value <= ADC_MAX:
            raise ChunkError(f"ADC value {value} is outside 0..{ADC_MAX}")

    lop, lom = body.get("lop", ""), body.get("lom", "")
    for name, flags in (("lop", lop), ("lom", lom)):
        if not isinstance(flags, str) or len(flags) != len(adc):
            raise ChunkError(f"'{name}' must be one character per sample")
        if set(flags) - {"0", "1"}:
            raise ChunkError(f"'{name}' must contain only '0' and '1'")

    return {"device_id": device_id, "seq": body["seq"], "t_us": body["t_us"],
            "fs": fs, "n": body["n"], "adc": adc, "lop": lop, "lom": lom}


class DeviceStream:
    """The live state of one device: buffer, counters, windows, last verdict."""

    def __init__(self, device_id, buffer_s=10, on_window=None, require_beat=True):
        self.device_id = device_id
        self.on_window = on_window
        size = FS_HZ * buffer_s
        self.adc = collections.deque(maxlen=size)
        self.lead = collections.deque(maxlen=size)
        self.t_dev = collections.deque(maxlen=size)
        self.windows = esm.WindowBuilder(require_beat=require_beat)
        self.lock = threading.Lock()

        self.n_chunks = self.n_samples = 0
        self.n_malformed = self.n_duplicates = 0
        self.n_lost_chunks = 0          # counted from sequence numbers
        self.n_restarts = 0             # the device began counting again
        self.n_lead_off = 0
        self.last_seq = None
        self.last_seen = None
        self.first_seen = None
        self.status = {}                # last retained status message
        self.events = collections.deque(maxlen=20)
        self.last_chunk = None          # the raw contents of the newest chunk
        self.last_result = None
        self.mode = "high_recall"   # operating point, set from the live page
        self.last_error = ""

    # -- ingestion -------------------------------------------------------
    def add_chunk(self, chunk):
        """Admit one validated chunk. Returns the completed window, or None."""
        now = time.time()
        window = None
        with self.lock:
            seq = chunk["seq"]
            if self.last_seq is not None:
                if seq + RESTART_BACKSTEP < self.last_seq:
                    # The counter did not step back by one or two, it fell off a
                    # cliff: the device rebooted and started counting from zero
                    # again. Treating that as a very long run of duplicates
                    # would silently ignore the device for ever, which is how a
                    # monitor stops monitoring without saying so.
                    self.n_restarts += 1
                    self.windows.reset("device_restart")
                    self.events.appendleft(
                        {"at": now, "event": "device_restart", "value": seq})
                    self.last_seq = None
                elif seq <= self.last_seq:
                    # A duplicate or an out-of-order redelivery. Dropping it is
                    # correct: admitting it would put the same second of heart
                    # activity into the signal twice.
                    self.n_duplicates += 1
                    return None



            self.last_seq = seq
            self.n_chunks += 1
            self.first_seen = self.first_seen or now
            self.last_seen = now

            # Kept as received, so the device page can show that data is
            # arriving even when none of it is a usable ECG.
            adc = chunk["adc"]
            self.last_chunk = {
                "seq": seq, "n": chunk["n"], "t_us": chunk["t_us"],
                "received_at": now,
                "adc_head": list(adc[:12]),
                "adc_min": min(adc) if adc else None,
                "adc_max": max(adc) if adc else None,
                "lop": chunk["lop"], "lom": chunk["lom"],
            }

            period_us = 1_000_000 // FS_HZ
            base_index = seq * chunk["n"]
            for i, value in enumerate(chunk["adc"]):
                sample = esm.Sample(
                    seq=base_index + i,
                    t_us=chunk["t_us"] + i * period_us,
                    adc=value,
                    lo_plus=int(chunk["lop"][i]) if chunk["lop"] else 0,
                    lo_minus=int(chunk["lom"][i]) if chunk["lom"] else 0,
                )
                self.n_samples += 1
                self.n_lead_off += sample.lead_off
                self.adc.append(sample.adc)
                self.lead.append(sample.lead_off)
                self.t_dev.append(sample.t_us)
                finished = self.windows.add(sample)
                if finished is not None:
                    window = finished

        if window is not None and self.on_window is not None:
            # Called outside the lock: scoring takes hundreds of milliseconds
            # and must not hold up the next chunk.
            try:
                self.last_result = self.on_window(self.device_id, window)
            except Exception as exc:                  # noqa: BLE001 - reported
                self.last_error = f"scoring failed: {exc}"
                self.last_result = {"error": self.last_error}
        return window

    def note_status(self, raw):
        try:
            body = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        except (ValueError, TypeError):
            body = {"state": str(raw)}
        if isinstance(body, dict):
            self.status = body
            self.status["received_at"] = time.time()

    def note_event(self, raw):
        try:
            body = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        except (ValueError, TypeError):
            body = {"event": str(raw)}
        body["at"] = time.time()
        self.events.appendleft(body)

    # -- reporting -------------------------------------------------------
    @property
    def online(self):
        """Receiving now, rather than 'said it was online at some point'."""
        if self.last_seen is None:
            return False
        return (time.time() - self.last_seen) < SILENCE_TIMEOUT_S

    def measured_rate_hz(self, n=300):
        """Sampling rate from the device's own timestamps, not from the clock
        of whichever thread happened to receive the packets."""
        with self.lock:
            t = list(self.t_dev)[-n:]
        if len(t) < 2 or t[-1] <= t[0]:
            return None
        return round((len(t) - 1) * 1e6 / (t[-1] - t[0]), 2)

    def snapshot(self, trace_points=1000):
        with self.lock:
            adc = np.array(self.adc, dtype=float)
            lead = np.array(self.lead, dtype=bool)
        lead_off = bool(lead[-FS_HZ // 2:].any()) if lead.size else True
        valid, reason = esm.diagnose(adc[-FS_HZ * 2:] if adc.size else adc,
                                     lead_off, True, long_adc=adc if adc.size else None)
        bpm, strength = esm.detect_beat(adc) if adc.size else (None, 0.0)
        online = self.online
        return {
            "device_id": self.device_id,
            "online": online,
            # Each AD8232 lead-off pin on its own, from the last half second
            # of the newest chunk. None while nothing is arriving: a buffer
            # left over from minutes ago says nothing about the electrodes now.
            "electrodes": self._electrode_state() if online else None,
            "last_seen_s_ago": (round(time.time() - self.last_seen, 1)
                                if self.last_seen else None),
            "status": self.status,
            "lead_off": lead_off,
            "signal_ok": bool(valid),
            "diagnosis": reason,
            "bpm": None if bpm is None else round(bpm),
            "beat_strength": round(strength, 2),
            "samples": self.n_samples,
            "chunks": self.n_chunks,
            "lost_chunks": self.n_lost_chunks,
            "duplicate_chunks": self.n_duplicates,
            "device_restarts": self.n_restarts,
            "malformed_chunks": self.n_malformed,
            "rate_hz": self.measured_rate_hz(),
            "window_progress": self.windows.progress,
            "window_length": WINDOW_LEN,
            "windows_done": self.windows.completed,
            "windows_discarded": dict(self.windows.rejected),
            "trace": [int(v) for v in adc[-trace_points:]],
            "lead_off_flags": [bool(v) for v in lead[-trace_points:]],
            "events": list(self.events)[:10],
            "last_chunk": self.last_chunk,
            "mode": self.mode,
            "last_result": self.last_result,
            "disclaimer": esm.DISCLAIMER,
        }

    def _electrode_state(self):
        chunk = self.last_chunk
        if not chunk or not (chunk["lop"] or chunk["lom"]):
            return None
        half = FS_HZ // 2
        return {"lo_plus_off": "1" in (chunk["lop"] or "")[-half:],
                "lo_minus_off": "1" in (chunk["lom"] or "")[-half:]}


class StreamRegistry:
    """Every device the backend has heard from, and the routing of one message."""

    def __init__(self, on_window=None):
        self.on_window = on_window
        self.streams = {}
        self.lock = threading.Lock()
        self.n_messages = 0
        self.n_rejected = 0
        self.last_rejection = ""

    def get(self, device_id, create=True):
        with self.lock:
            stream = self.streams.get(device_id)
            if stream is None and create:
                stream = DeviceStream(device_id, on_window=self.on_window)
                self.streams[device_id] = stream
            return stream

    def device_ids(self):
        with self.lock:
            return sorted(self.streams)

    def handle_message(self, topic, payload):
        """Route one MQTT message. Never raises: one bad device must not stop
        the others, so a rejection is counted and reported, not propagated."""
        self.n_messages += 1
        parts = topic.split("/")
        if len(parts) != 4 or parts[0] != "ecg" or parts[1] != "devices":
            self.n_rejected += 1
            self.last_rejection = f"unexpected topic '{topic}'"
            return None
        device_id, kind = parts[2], parts[3]

        if kind == "data":
            stream = self.get(device_id)
            try:
                chunk = parse_chunk(payload, expect_device=device_id)
            except ChunkError as exc:
                stream.n_malformed += 1
                self.n_rejected += 1
                self.last_rejection = f"{device_id}: {exc}"
                stream.windows.reset("malformed_chunk")
                return None
            return stream.add_chunk(chunk)

        if kind == "status":
            self.get(device_id).note_status(payload)
            return None
        if kind == "event":
            self.get(device_id).note_event(payload)
            return None

        self.n_rejected += 1
        self.last_rejection = f"unhandled topic '{topic}'"
        return None

    def snapshot(self, device_id):
        stream = self.get(device_id, create=False)
        if stream is None:
            return {"device_id": device_id, "online": False, "samples": 0,
                    "trace": [], "lead_off_flags": [],
                    "diagnosis": "No data has arrived from this device yet.",
                    "signal_ok": False, "lead_off": True,
                    "window_length": WINDOW_LEN, "window_progress": 0,
                    "windows_done": 0, "windows_discarded": {},
                    "disclaimer": esm.DISCLAIMER}
        return stream.snapshot()
