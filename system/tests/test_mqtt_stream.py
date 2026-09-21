"""
Tests for the MQTT chunk ingester.

No broker and no device: chunks are built here in the exact shape the firmware
prints them (firmware/ecg_esp32_mqtt), so these tests fail if either side of
that contract changes.

The theme throughout is that a damaged stream must be *refused*, not repaired.
A dropped second of ECG that gets quietly stitched over produces a waveform
that looks plausible and never happened, and the model will score it just as
confidently as a real one.
"""

import json

import numpy as np
import pytest

import mqtt_stream


def chunk(seq=0, n=100, device="ESP32-001", t_us=None, adc=None,
          lop=None, lom=None, fs=100):
    """One chunk in the firmware's format."""
    if adc is None:
        adc = [2048 + (i % 5) for i in range(n)]
    body = {"device_id": device, "seq": seq,
            "t_us": 1_000_000 + seq * n * 10_000 if t_us is None else t_us,
            "fs": fs, "n": n, "adc": adc,
            "lop": lop if lop is not None else "0" * len(adc),
            "lom": lom if lom is not None else "0" * len(adc)}
    return json.dumps(body).encode("utf-8")


def ecg_chunks(n_chunks=10, bpm=72, device="ESP32-001", start_seq=0):
    """Chunks carrying a crude but detectable beat, so windows are accepted."""
    fs, per = 100, 100
    total = n_chunks * per
    rng = np.random.default_rng(5)
    signal = np.full(total, 2100.0) + rng.normal(0, 4, total)
    period = int(fs * 60 / bpm)
    for i in range(0, total, period):
        if i + 3 < total:
            signal[i:i + 3] += [120, 300, 90]
    out = []
    for c in range(n_chunks):
        block = signal[c * per:(c + 1) * per]
        out.append(chunk(seq=start_seq + c, n=per, device=device,
                         adc=[int(v) for v in block]))
    return out


# ---------------------------------------------------------------------------
# parse_chunk
# ---------------------------------------------------------------------------
def test_a_well_formed_chunk_is_accepted():
    parsed = mqtt_stream.parse_chunk(chunk(seq=3), expect_device="ESP32-001")
    assert parsed["seq"] == 3
    assert parsed["n"] == 100 and len(parsed["adc"]) == 100
    assert set(parsed["lop"]) == {"0"}


@pytest.mark.parametrize("payload, message", [
    (b"not json at all", "not valid JSON"),
    (json.dumps([1, 2, 3]).encode(), "must be a JSON object"),
    (json.dumps({"seq": 1}).encode(), "device_id"),
    (chunk(n=100, adc=[2048] * 99), "'n' is 100 but 'adc' holds 99"),
    (chunk(adc=[70000] * 100), "outside 0..4095"),
    (chunk(adc=[1.5] * 100), "only integers"),
    (chunk(fs=250), "expects 100 Hz"),
    (chunk(lop="01"), "one character per sample"),
    (chunk(lop="x" * 100), "only '0' and '1'"),
])
def test_a_damaged_chunk_is_refused_with_a_reason(payload, message):
    with pytest.raises(mqtt_stream.ChunkError) as exc:
        mqtt_stream.parse_chunk(payload, expect_device="ESP32-001")
    assert message in str(exc.value)


def test_a_chunk_claiming_another_device_is_refused():
    """The topic says who published; the body must agree, or one device could
    file readings under another device's id."""
    with pytest.raises(mqtt_stream.ChunkError) as exc:
        mqtt_stream.parse_chunk(chunk(device="ESP32-999"),
                                expect_device="ESP32-001")
    assert "does not match topic" in str(exc.value)


# ---------------------------------------------------------------------------
# DeviceStream
# ---------------------------------------------------------------------------
def test_samples_accumulate_and_a_clean_window_is_produced():
    seen = []
    stream = mqtt_stream.DeviceStream("ESP32-001",
                                      on_window=lambda d, w: seen.append(w) or {"ok": True})
    for raw in ecg_chunks(10):
        stream.add_chunk(mqtt_stream.parse_chunk(raw))
    assert stream.n_samples == 1000
    assert len(seen) == 1 and seen[0].values.shape == (1000,)
    assert stream.windows.completed == 1


def test_a_duplicate_chunk_is_dropped():
    stream = mqtt_stream.DeviceStream("ESP32-001")
    stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=0)))
    stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=0)))   # redelivered
    assert stream.n_duplicates == 1
    assert stream.n_samples == 100          # counted once, not twice


def test_a_missing_chunk_is_counted_and_discards_the_partial_window():
    stream = mqtt_stream.DeviceStream("ESP32-001")
    stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=0)))
    stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=3)))   # 1 and 2 lost
    assert stream.n_lost_chunks == 2
    assert stream.windows.rejected["chunk_gap"] == 1
    assert stream.events[0]["event"] == "chunk_gap"


def test_a_window_spanning_a_gap_is_never_scored():
    """The window that was in progress when the gap happened must not be
    completed by later chunks: the hole would be invisible in the result."""
    scored = []
    stream = mqtt_stream.DeviceStream("ESP32-001",
                                      on_window=lambda d, w: scored.append(w))
    chunks = ecg_chunks(12)
    for raw in chunks[:4]:
        stream.add_chunk(mqtt_stream.parse_chunk(raw))
    for raw in chunks[5:]:                 # chunk 4 never arrives
        stream.add_chunk(mqtt_stream.parse_chunk(raw))
    assert stream.n_lost_chunks == 1
    assert scored == []


def test_lead_off_samples_stop_a_window():
    stream = mqtt_stream.DeviceStream("ESP32-001")
    for i, raw in enumerate(ecg_chunks(10)):
        if i == 5:
            raw = chunk(seq=i, lop="1" * 100)
        stream.add_chunk(mqtt_stream.parse_chunk(raw))
    assert stream.windows.completed == 0
    assert stream.windows.rejected["lead_off"] >= 1


def test_each_electrode_is_reported_and_unknown_when_offline(monkeypatch):
    stream = mqtt_stream.DeviceStream("ESP32-001")
    # Nothing received yet: the electrodes are unknown, not "connected".
    assert stream.snapshot()["electrodes"] is None

    stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=0, lop="1" * 100)))
    snap = stream.snapshot()
    assert snap["electrodes"] == {"lo_plus_off": True, "lo_minus_off": False}
    assert snap["last_chunk"]["seq"] == 0 and len(snap["last_chunk"]["adc_head"]) == 12

    # Once the device goes silent, an old buffer must not vouch for anything.
    monkeypatch.setattr(mqtt_stream, "SILENCE_TIMEOUT_S", -1)
    assert stream.snapshot()["electrodes"] is None


def test_offline_when_nothing_has_arrived():
    stream = mqtt_stream.DeviceStream("ESP32-001")
    assert stream.online is False
    stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=0)))
    assert stream.online is True


# ---------------------------------------------------------------------------
# StreamRegistry routing
# ---------------------------------------------------------------------------
def test_routing_of_data_status_and_event_topics():
    registry = mqtt_stream.StreamRegistry()
    registry.handle_message("ecg/devices/ESP32-001/data", chunk(seq=0))
    registry.handle_message("ecg/devices/ESP32-001/status",
                            b'{"device_id":"ESP32-001","state":"online","rssi":-51}')
    registry.handle_message("ecg/devices/ESP32-001/event",
                            b'{"device_id":"ESP32-001","event":"reconnected"}')

    stream = registry.get("ESP32-001", create=False)
    assert stream.n_samples == 100
    assert stream.status["state"] == "online"
    assert stream.events[0]["event"] == "reconnected"


def test_a_malformed_chunk_is_counted_and_does_not_raise():
    registry = mqtt_stream.StreamRegistry()
    assert registry.handle_message("ecg/devices/ESP32-001/data", b"rubbish") is None
    stream = registry.get("ESP32-001", create=False)
    assert stream.n_malformed == 1
    assert registry.n_rejected == 1
    assert "not valid JSON" in registry.last_rejection


def test_an_unexpected_topic_is_rejected():
    registry = mqtt_stream.StreamRegistry()
    assert registry.handle_message("ecg/devices/ESP32-001", chunk()) is None
    assert registry.handle_message("something/else", chunk()) is None
    assert registry.n_rejected == 2


def test_snapshot_of_an_unknown_device_is_empty_not_an_error():
    registry = mqtt_stream.StreamRegistry()
    snap = registry.snapshot("ESP32-404")
    assert snap["online"] is False and snap["trace"] == []
    assert "not a clinical diagnosis" in snap["disclaimer"]


def test_snapshot_reports_rate_and_beat_from_a_real_looking_stream():
    registry = mqtt_stream.StreamRegistry()
    for raw in ecg_chunks(10, bpm=75):
        registry.handle_message("ecg/devices/ESP32-001/data", raw)
    snap = registry.snapshot("ESP32-001")
    assert snap["samples"] == 1000
    assert snap["rate_hz"] == pytest.approx(100.0, abs=0.5)
    assert 60 <= snap["bpm"] <= 90
    assert snap["lost_chunks"] == 0 and snap["duplicate_chunks"] == 0
    assert len(snap["trace"]) == 1000


def test_a_rebooted_device_is_followed_not_ignored():
    """After a reboot the device counts from zero again.

    Read as duplicates, that would silence the device permanently; the monitor
    would show a frozen trace and never say why. This was a real defect, found
    on 2026-09-23 by restarting the publisher.
    """
    stream = mqtt_stream.DeviceStream("ESP32-001")
    for raw in ecg_chunks(6):
        stream.add_chunk(mqtt_stream.parse_chunk(raw))
    assert stream.n_samples == 600

    for raw in ecg_chunks(6):                  # same seq numbers, from 0
        stream.add_chunk(mqtt_stream.parse_chunk(raw))

    assert stream.n_restarts == 1
    assert stream.n_samples == 1200            # the new samples were admitted
    assert stream.windows.rejected["device_restart"] == 1
    assert stream.events[0]["event"] == "device_restart"


def test_a_genuine_redelivery_is_still_a_duplicate():
    """One chunk arriving twice is not a reboot and must not reset anything."""
    stream = mqtt_stream.DeviceStream("ESP32-001")
    for seq in (0, 1, 2, 2, 1):
        stream.add_chunk(mqtt_stream.parse_chunk(chunk(seq=seq)))
    assert stream.n_duplicates == 2
    assert stream.n_restarts == 0
    assert stream.n_samples == 300
