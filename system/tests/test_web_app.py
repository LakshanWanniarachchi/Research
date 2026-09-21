"""
Tests for the multi-user web application.

The most important test in this file is `test_one_user_cannot_reach_another`.
Everything else is validation and plumbing; that one is the promise the system
makes to the person wearing the electrodes.
"""

import numpy as np
import pytest

import models


def register(client, username, password="a-good-password"):
    """Create an account and leave that client signed in as it."""
    resp = client.post("/register", data={"username": username,
                                          "password": password,
                                          "confirm": password})
    assert resp.status_code in (302, 200), resp.status_code
    return username


def store_report(db, user_id, device_id="ESP32-001", prediction=0,
                 probability=0.12):
    """One stored recording, as the MQTT path would have written it."""
    signal = np.linspace(2000, 2200, 1000, dtype=np.float32)
    return models.save_ecg_report(
        db, user_id=user_id, signal=signal, probability=probability,
        threshold=0.2736, mode="high_recall", prediction=prediction,
        source="mqtt", device_id=device_id, sample_rate=100,
        measured_rate=100.0, bpm=71.0, beat_strength=0.95,
        model_name="1D CNN ensemble", model_version="ensemble-5seed roc_auc=0.9079",
        inference_ms=310.0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_registration_creates_an_account_and_signs_in(client):
    resp = client.post("/register", data={"username": "alice",
                                          "password": "alice-password",
                                          "confirm": "alice-password"})
    assert resp.status_code == 302 and "/profile" in resp.headers["Location"]
    assert client.get("/dashboard").status_code == 200


@pytest.mark.parametrize("form, field", [
    ({"username": "ab", "password": "alice-password", "confirm": "alice-password"}, "username"),
    ({"username": "alice bob", "password": "alice-password", "confirm": "alice-password"}, "username"),
    ({"username": "alice", "password": "short", "confirm": "short"}, "password"),
    ({"username": "alice", "password": "alice-password", "confirm": "different-one"}, "confirm"),
])
def test_bad_registrations_are_refused(client, form, field):
    resp = client.post("/register", data=form)
    assert resp.status_code == 400
    assert client.get("/dashboard").status_code == 302     # still signed out


def test_a_username_cannot_be_taken_twice(client):
    register(client, "alice")
    client.post("/logout")
    resp = client.post("/register", data={"username": "alice",
                                          "password": "another-password",
                                          "confirm": "another-password"})
    assert resp.status_code == 400
    assert "taken" in resp.get_data(as_text=True)


def test_the_password_is_not_stored_in_clear(client, db):
    register(client, "alice", "alice-password")
    row = db.execute("SELECT password_hash FROM users WHERE username = 'alice'").fetchone()
    assert "alice-password" not in row["password_hash"]
    assert len(row["password_hash"]) > 40


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def test_profile_is_saved_and_shown(client, db):
    register(client, "alice")
    resp = client.post("/profile", data={"full_name": "Alice Example",
                                         "dob": "1998-04-12",
                                         "height_cm": "172.5",
                                         "weight_kg": "64"})
    assert resp.status_code == 200
    row = db.execute("SELECT full_name, dob, height_cm, weight_kg FROM users "
                     "WHERE username = 'alice'").fetchone()
    assert row["full_name"] == "Alice Example"
    assert row["dob"] == "1998-04-12"
    assert row["height_cm"] == 172.5 and row["weight_kg"] == 64.0
    assert "Alice Example" in client.get("/dashboard").get_data(as_text=True)


@pytest.mark.parametrize("field, value", [
    ("dob", "2099-01-01"),      # in the future
    ("dob", "not-a-date"),
    ("height_cm", "17"),        # metres typed as centimetres
    ("height_cm", "900"),
    ("weight_kg", "3"),
    ("weight_kg", "abc"),
])
def test_impossible_profile_values_are_refused(client, field, value):
    register(client, "alice")
    resp = client.post("/profile", data={field: value})
    assert resp.status_code == 400


def test_an_empty_profile_field_is_allowed(client):
    """Someone may know their height today and their weight next week."""
    register(client, "alice")
    assert client.post("/profile", data={"height_cm": "170"}).status_code == 200


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------
def test_claiming_a_device_links_it_to_the_account(client, db):
    register(client, "alice")
    client.post("/devices", data={"action": "claim", "device_id": "ESP32-001",
                                  "label": "bench"})
    row = db.execute("SELECT d.device_id, u.username FROM devices d "
                     "JOIN users u ON u.id = d.user_id").fetchone()
    assert row["device_id"] == "ESP32-001" and row["username"] == "alice"


def test_a_device_cannot_be_claimed_by_two_accounts(client, db):
    register(client, "alice")
    client.post("/devices", data={"action": "claim", "device_id": "ESP32-001"})
    client.post("/logout")

    register(client, "bob")
    client.post("/devices", data={"action": "claim", "device_id": "ESP32-001"},
                follow_redirects=True)
    owners = db.execute("SELECT u.username FROM devices d JOIN users u "
                        "ON u.id = d.user_id WHERE d.device_id = 'ESP32-001'").fetchall()
    assert [r["username"] for r in owners] == ["alice"]


def test_a_device_can_be_released_by_its_owner_only(client, db):
    register(client, "alice")
    client.post("/devices", data={"action": "claim", "device_id": "ESP32-001"})
    client.post("/logout")

    register(client, "bob")
    client.post("/devices", data={"action": "release", "device_id": "ESP32-001"})
    assert db.execute("SELECT COUNT(*) AS n FROM devices").fetchone()["n"] == 1

    client.post("/logout")
    client.post("/login", data={"username": "alice", "password": "a-good-password"})
    client.post("/devices", data={"action": "release", "device_id": "ESP32-001"})
    assert db.execute("SELECT COUNT(*) AS n FROM devices").fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# Reports, history and isolation
# ---------------------------------------------------------------------------
def test_a_stored_report_appears_in_history_and_opens(client, db):
    register(client, "alice")
    user = models.get_user_by_name(db, "alice")
    report_id = store_report(db, user["id"], prediction=1, probability=0.83)

    history = client.get("/ecg/history").get_data(as_text=True)
    # The history table shows the probability as a percentage, which is how a
    # reader compares it against a threshold quoted the same way.
    assert "MI pattern" in history and "83.0%" in history

    page = client.get(f"/ecg/{report_id}")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "83.0%" in body                       # the probability
    assert "ensemble-5seed roc_auc=0.9079" in body   # the model that produced it
    assert "not a clinical diagnosis" in body
    assert "2000" in body                        # the waveform is embedded


def test_history_filters_by_result_and_counts(client, db):
    register(client, "alice")
    user = models.get_user_by_name(db, "alice")
    store_report(db, user["id"], prediction=0, probability=0.10)
    store_report(db, user["id"], prediction=1, probability=0.71)
    store_report(db, user["id"], prediction=1, probability=0.66)

    assert "3 recordings" in client.get("/ecg/history").get_data(as_text=True)
    assert "2 recordings" in client.get("/ecg/history?result=mi").get_data(as_text=True)
    assert "1 recording" in client.get("/ecg/history?result=normal").get_data(as_text=True)


def test_one_user_cannot_reach_another(client, db):
    """The promise: a report belongs to one account and to no other.

    Bob is signed in and knows Alice's report id. Every route that could leak
    it must refuse: the report page, the history list, and the live feed of her
    device. 404 rather than 403, because Bob has no business learning that the
    id exists at all.
    """
    register(client, "alice")
    alice = models.get_user_by_name(db, "alice")
    alice_report = store_report(db, alice["id"], device_id="ESP32-ALICE")
    models.claim_device(db, alice["id"], "ESP32-ALICE")
    client.post("/logout")

    register(client, "bob")
    assert client.get(f"/ecg/{alice_report}").status_code == 404
    assert client.get("/ecg/api/live?device=ESP32-ALICE").status_code == 404

    history = client.get("/ecg/history").get_data(as_text=True)
    assert "ESP32-ALICE" not in history
    assert "0 recordings" in history
    assert "ESP32-ALICE" not in client.get("/dashboard").get_data(as_text=True)

    bob = models.get_user_by_name(db, "bob")
    assert models.get_ecg_report(db, bob["id"], alice_report) is None
    assert models.get_ecg_report(db, alice["id"], alice_report) is not None


def test_every_ecg_page_needs_a_session(client):
    for path in ("/ecg/live", "/ecg/device", "/ecg/history", "/ecg/1",
                 "/ecg/api/live", "/profile", "/devices"):
        resp = client.get(path)
        assert resp.status_code == 302 and "/login" in resp.headers["Location"]


def test_the_device_stream_page_renders_for_its_owner(client):
    register(client, "alice")
    client.post("/devices", data={"action": "claim", "device_id": "ESP32-001"})
    body = client.get("/ecg/device").get_data(as_text=True)
    assert "Device Stream" in body and "ESP32-001" in body


def test_a_stored_waveform_survives_the_round_trip(db, client):
    """The report page must redraw what was scored, not an approximation."""
    register(client, "alice")
    user = models.get_user_by_name(db, "alice")
    signal = np.sin(np.linspace(0, 20, 1000)).astype(np.float32) * 300 + 2048
    report_id = models.save_ecg_report(
        db, user_id=user["id"], signal=signal, probability=0.4, threshold=0.2736,
        mode="high_recall", prediction=1, source="mqtt", device_id="ESP32-001",
        sample_rate=100)

    row = models.get_ecg_report(db, user["id"], report_id)
    restored = models.blob_to_signal(row["signal"])
    assert restored.shape == (1000,)
    assert np.allclose(restored, signal, atol=1e-4)
    assert row["duration_s"] == 10.0 and row["lead"] == "I"


def test_the_operating_point_can_be_changed_and_is_remembered(client, db, monkeypatch):
    """The live page picks the threshold the next window is judged at."""
    import mqtt_bridge
    register(client, "alice")
    user = models.get_user_by_name(db, "alice")
    models.claim_device(db, user["id"], "ESP32-001")
    mqtt_bridge.STREAMS.get("ESP32-001")           # the device has been heard from

    d = client.get("/ecg/api/live?device=ESP32-001&mode=default").get_json()
    assert d["mode"] == "default"
    assert d["thresholds"]["default"] == 0.5

    # An unknown point is ignored rather than accepted and used later.
    d = client.get("/ecg/api/live?device=ESP32-001&mode=nonsense").get_json()
    assert d["mode"] == "default"
