"""The MQTT bridge.

No broker runs during these tests. `handle_reading` was written to take bytes
and return a dict precisely so that every malformed-payload path could be
exercised directly, which standing up a broker and publishing to it would make
slow and awkward.
"""

import json

import numpy as np
import pytest

import mqtt_bridge
from mqtt_bridge import PayloadError

THRESHOLDS = {"default": 0.5, "best_f1": 0.4433, "high_recall": 0.2736}


def payload(device_id="ESP32-TEST", n=1000, mode="high_recall", token=None,
            signal=None):
    body = {"device_id": device_id, "mode": mode,
            "signal": signal if signal is not None else [0.0] * n}
    if token is not None:
        body["token"] = token
    return json.dumps(body).encode("utf-8")


class TestTopicParsing:
    def test_device_id_is_taken_from_the_topic(self):
        assert mqtt_bridge.device_id_from_topic("ecg/ESP32-9/reading") == "ESP32-9"

    @pytest.mark.parametrize("topic", [
        "ecg/ESP32-9", "ecg/ESP32-9/result", "other/ESP32-9/reading",
        "ecg//reading/extra", "",
    ])
    def test_unexpected_topics_are_rejected(self, topic):
        with pytest.raises(PayloadError):
            mqtt_bridge.device_id_from_topic(topic)


class TestPayloadValidation:
    """A device on a public broker can publish anything, including nonsense."""

    def test_a_well_formed_payload_parses(self):
        got = mqtt_bridge.parse_payload(payload(), input_length=1000)
        assert got["device_id"] == "ESP32-TEST"
        assert len(got["signal"]) == 1000

    def test_non_json_is_rejected(self):
        with pytest.raises(PayloadError, match="JSON"):
            mqtt_bridge.parse_payload(b"not json at all", input_length=1000)

    def test_a_json_array_is_rejected(self):
        with pytest.raises(PayloadError, match="JSON object"):
            mqtt_bridge.parse_payload(b"[1, 2, 3]", input_length=1000)

    def test_missing_signal_is_rejected(self):
        raw = json.dumps({"device_id": "X"}).encode()
        with pytest.raises(PayloadError, match="signal"):
            mqtt_bridge.parse_payload(raw, input_length=1000)

    def test_wrong_length_is_rejected(self):
        with pytest.raises(PayloadError, match="exactly 1000"):
            mqtt_bridge.parse_payload(payload(n=999), input_length=1000)

    def test_non_numeric_samples_are_rejected(self):
        bad = ["a"] * 1000
        with pytest.raises(PayloadError, match="only numbers"):
            mqtt_bridge.parse_payload(payload(signal=bad), input_length=1000)

    def test_missing_device_id_is_rejected(self):
        raw = json.dumps({"signal": [0.0] * 1000}).encode()
        with pytest.raises(PayloadError, match="device_id"):
            mqtt_bridge.parse_payload(raw, input_length=1000)


class TestHandleReading:
    def test_a_reading_is_scored_and_stored(self, db, seeded_patient, stub_model):
        import models
        result = mqtt_bridge.handle_reading(
            "ecg/ESP32-TEST/reading", payload(),
            cnn=[stub_model], thresholds=THRESHOLDS, require_token=False)

        assert result["prediction"] == 1          # stub returns 0.9
        assert result["patient_name"] == "Test Patient"
        assert models.prediction_stats(db)["total"] == 1

    def test_topic_and_body_must_name_the_same_device(self, db, stub_model):
        """Otherwise one device could file readings against another's record."""
        with pytest.raises(PayloadError, match="does not match topic"):
            mqtt_bridge.handle_reading(
                "ecg/ESP32-OTHER/reading", payload(device_id="ESP32-TEST"),
                cnn=[stub_model], thresholds=THRESHOLDS, require_token=False)

    def test_unknown_mode_is_rejected(self, db, stub_model):
        with pytest.raises(PayloadError, match="mode must be"):
            mqtt_bridge.handle_reading(
                "ecg/ESP32-TEST/reading", payload(mode="nonsense"),
                cnn=[stub_model], thresholds=THRESHOLDS, require_token=False)

    def test_an_unknown_device_is_stored_unattributed(self, db, stub_model):
        import models
        result = mqtt_bridge.handle_reading(
            "ecg/NOT-ENROLLED/reading", payload(device_id="NOT-ENROLLED"),
            cnn=[stub_model], thresholds=THRESHOLDS, require_token=False)

        assert result["patient_id"] is None
        assert "not registered" in result["warning"]
        # Discarding the reading would be the worse failure.
        assert models.prediction_stats(db)["total"] == 1

    def test_a_device_reading_never_claims_a_ground_truth(self, db, seeded_patient,
                                                          stub_model):
        """A real patient's diagnosis is unknown at the moment of measurement."""
        mqtt_bridge.handle_reading(
            "ecg/ESP32-TEST/reading", payload(),
            cnn=[stub_model], thresholds=THRESHOLDS, require_token=False)
        row = db.execute("SELECT truth, correct, source FROM predictions").fetchone()
        assert row["truth"] is None
        assert row["correct"] is None
        assert row["source"] == "device"


class TestDeviceToken:
    def test_a_correct_token_is_accepted(self, db, seeded_patient, stub_model):
        result = mqtt_bridge.handle_reading(
            "ecg/ESP32-TEST/reading", payload(token="secret-token-abc"),
            cnn=[stub_model], thresholds=THRESHOLDS, require_token=True)
        assert result["patient_name"] == "Test Patient"

    def test_a_wrong_token_is_refused(self, db, seeded_patient, stub_model):
        with pytest.raises(PayloadError, match="not recognised"):
            mqtt_bridge.handle_reading(
                "ecg/ESP32-TEST/reading", payload(token="wrong"),
                cnn=[stub_model], thresholds=THRESHOLDS, require_token=True)

    def test_a_missing_token_is_refused(self, db, seeded_patient, stub_model):
        with pytest.raises(PayloadError, match="not recognised"):
            mqtt_bridge.handle_reading(
                "ecg/ESP32-TEST/reading", payload(),
                cnn=[stub_model], thresholds=THRESHOLDS, require_token=True)

    def test_a_refused_reading_is_not_stored(self, db, seeded_patient, stub_model):
        import models
        with pytest.raises(PayloadError):
            mqtt_bridge.handle_reading(
                "ecg/ESP32-TEST/reading", payload(token="wrong"),
                cnn=[stub_model], thresholds=THRESHOLDS, require_token=True)
        assert models.prediction_stats(db)["total"] == 0


class TestClientConstruction:
    def test_the_client_declares_the_callback_api_version(self):
        """paho-mqtt 2.x raises unless the version is declared explicitly."""
        client = mqtt_bridge.build_client()
        assert client is not None

    def test_the_bridge_is_off_unless_enabled(self, stub_model):
        """An unreachable optional transport must not stop the app starting."""
        assert mqtt_bridge.MQTT_ENABLED is False
        assert mqtt_bridge.start([stub_model], THRESHOLDS) is None
