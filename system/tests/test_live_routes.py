"""
Tests for the live acquisition pages and endpoints.

No serial port and no ESP32 are needed: `live_serial.LiveSession` is driven
with a stub scorer and a stub receiver, which is the reason the session module
never imports the application, the model or the database itself.
"""

import numpy as np
import pytest


def test_live_page_requires_a_session(client):
    resp = client.get("/live")
    assert resp.status_code == 302 and "/login" in resp.headers["Location"]


def test_live_endpoints_require_a_session(client):
    for path in ("/api/live/data", "/api/live/ports"):
        assert client.get(path).status_code == 302
    for path in ("/api/live/start", "/api/live/stop"):
        assert client.post(path).status_code == 302


def test_live_page_renders_for_a_signed_in_user(auth_client):
    resp = auth_client.get("/live")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Live ECG" in body
    # The safety wording is part of the page, not something added by hand later.
    assert "not a clinical diagnosis" in body
    assert "not validated" in body


def test_data_endpoint_reports_a_stopped_session(auth_client):
    data = auth_client.get("/api/live/data").get_json()
    assert data["running"] is False
    assert "not a clinical diagnosis" in data["disclaimer"]


def test_start_without_a_device_fails_cleanly(auth_client, monkeypatch):
    import ecg_serial_monitor as esm
    monkeypatch.setattr(esm, "find_esp32_port", lambda: None)
    resp = auth_client.post("/api/live/start", json={})
    assert resp.status_code == 400
    assert "no ESP32" in resp.get_json()["error"]


def test_a_scored_window_is_stored_against_the_device_holder(
        auth_client, db, monkeypatch):
    """A USB recording is stored exactly as an MQTT one: against the account
    that claimed the device, waveform included. Before 2026-09-24 the serial
    path wrote only a probability row, so USB recordings never reached the
    dashboard or the history page."""
    import app as app_module
    import models

    user = models.get_user_by_name(db, "admin")
    models.claim_device(db, user["id"], "ESP32-USB")
    monkeypatch.setattr(app_module, "LIVE_DEVICE_ID", "ESP32-USB")

    signal = np.random.default_rng(0).normal(2048, 60, 1000)
    result = app_module.score_live_window(signal, "high_recall")

    assert result["source"] == "serial"
    assert result["threshold"] == app_module.THRESHOLDS["high_recall"]
    assert result["label"] in ("NORMAL", "MI PATTERN DETECTED")

    row = db.execute("SELECT source, user_id, device_id, n_samples FROM "
                     "ecg_reports ORDER BY id DESC LIMIT 1").fetchone()
    assert row["source"] == "serial" and row["user_id"] == user["id"]
    assert row["device_id"] == "ESP32-USB" and row["n_samples"] == 1000

    pred = db.execute("SELECT source, truth FROM predictions ORDER BY id DESC "
                      "LIMIT 1").fetchone()
    assert pred["source"] == "device" and pred["truth"] is None


def test_session_snapshot_and_scoring_without_hardware(monkeypatch):
    """LiveSession drives its scorer from the window queue, not from serial."""
    import live_serial
    import ecg_serial_monitor as esm

    scored = []

    def fake_score(window, mode):
        scored.append((len(window.values), mode))
        return {"probability": 0.42, "threshold": 0.2736, "mode": mode,
                "prediction": 1, "label": "MI PATTERN DETECTED",
                "inference_ms": 12.3}

    session = live_serial.LiveSession(score_window=fake_score)
    assert session.snapshot()["running"] is False

    import queue
    session.window_queue = queue.Queue()
    window = esm.Window(values=np.zeros(1000, dtype=np.float32),
                        t_us=np.arange(1000) * 10_000, first_seq=0,
                        measured_rate_hz=100.0,
                        quality={"bpm": 72.0, "beat_strength": 0.9})
    session.window_queue.put(window)
    session.window_queue.put(None)
    session._score_loop()

    assert scored == [(1000, "high_recall")]
    assert session.last_result["probability"] == 0.42
    assert session.last_result["bpm"] == 72.0
    assert session.last_result["measured_rate_hz"] == 100.0


def test_a_failing_scorer_does_not_kill_the_session():
    import live_serial
    import ecg_serial_monitor as esm
    import queue

    def boom(values, mode):
        raise RuntimeError("model unavailable")

    session = live_serial.LiveSession(score_window=boom)
    session.window_queue = queue.Queue()
    session.window_queue.put(esm.Window(np.zeros(1000, dtype=np.float32),
                                        np.arange(1000), 0, 100.0, {}))
    session.window_queue.put(None)
    session._score_loop()
    assert "model unavailable" in session.last_result["error"]
