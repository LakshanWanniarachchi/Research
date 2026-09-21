"""Preprocessing, thresholds, storage and the HTTP endpoints.

These cover the invariants the thesis relies on: that training and serving
preprocess identically, that a model is never served under another model's
settings, that inspecting a result cannot change the record, and that start-up
cost is kept out of the reported inference speed.
"""

import numpy as np
import pytest

import model as mdl
import models


class TestPreprocessing:
    def test_z_score_gives_zero_mean_and_unit_deviation(self):
        mdl.set_bandpass(False)
        out = mdl.clean(np.random.default_rng(0).normal(5, 3, 1000))
        assert abs(out.mean()) < 1e-5
        assert abs(out.std() - 1.0) < 1e-4

    def test_a_flat_signal_does_not_divide_by_zero(self):
        mdl.set_bandpass(False)
        out = mdl.clean(np.zeros(1000, dtype=np.float32))
        assert np.all(np.isfinite(out))

    def test_single_and_batch_paths_agree(self):
        """Training uses the batch path and serving uses the single path.

        If they diverged the model would be trained on one thing and served
        another, and nothing would fail visibly.
        """
        sig = np.random.default_rng(1).normal(0, 1, 1000).astype(np.float32)
        for enabled in (False, True):
            mdl.set_bandpass(enabled)
            one = mdl.preprocess(sig).ravel()
            many = mdl.preprocess_many(np.stack([sig]))[0].ravel()
            assert np.allclose(one, many, atol=1e-5)
        mdl.set_bandpass(False)

    def test_preprocess_produces_the_shape_the_network_expects(self):
        mdl.set_bandpass(False)
        assert mdl.preprocess(np.zeros(1000)).shape == (1, 1000, 1)


class TestBandpass:
    def test_baseline_wander_is_removed(self):
        t = np.arange(1000) / 100.0
        beat = np.sin(2 * np.pi * 1.2 * t)
        drift = 3.0 * np.sin(2 * np.pi * 0.15 * t)     # below the 0.5 Hz edge
        out = mdl.bandpass((beat + drift).astype(np.float32))

        def power(x, lo, hi):
            spec = np.abs(np.fft.rfft(x)) ** 2
            freq = np.fft.rfftfreq(len(x), 1 / 100.0)
            return spec[(freq >= lo) & (freq < hi)].sum()

        before = power(beat + drift, 0, 0.5)
        after = power(out, 0, 0.5)
        assert after / before < 0.01           # at least 99% of drift removed

    def test_the_passband_survives(self):
        t = np.arange(1000) / 100.0
        beat = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)
        out = mdl.bandpass(beat)
        # Correlation is the honest check: amplitude may shift slightly, shape
        # must not.
        assert np.corrcoef(beat, out)[0, 1] > 0.99

    def test_filtering_is_off_unless_the_config_says_otherwise(self):
        """Models trained without the filter must not be served with it."""
        mdl._apply_preprocessing_config({})
        assert mdl.bandpass_enabled() is False

    def test_filtering_follows_the_loaded_model(self):
        mdl._apply_preprocessing_config(
            {"bandpass": {"enabled": True, "low_hz": 0.5, "high_hz": 40.0}})
        assert mdl.bandpass_enabled() is True
        mdl._apply_preprocessing_config({})       # leave it off for other tests


class TestPredictionRecord:
    def test_warm_up_is_recorded_separately_from_inference_speed(self, db):
        """A 2.4 s graph-tracing call averaged in with 0.3 s calls describes
        neither quantity."""
        for ms, warm in [(2400.0, True), (300.0, False), (310.0, False)]:
            models.save_prediction(db, source="device", probability=0.5,
                                   threshold=0.5, mode="default", prediction=1,
                                   inference_ms=ms, is_warmup=warm)
        stats = models.prediction_stats(db)
        assert stats["n_warm_calls"] == 2
        assert stats["n_warmup_calls"] == 1
        assert 300 <= stats["avg_inference_ms"] <= 310
        assert stats["warmup_ms"] == 2400.0

    def test_accuracy_counts_only_rows_with_a_known_truth(self, db):
        models.save_prediction(db, source="simulation", probability=0.9,
                               threshold=0.5, mode="default", prediction=1,
                               inference_ms=1.0, recording_id=0, truth=1)
        models.save_prediction(db, source="device", probability=0.9,
                               threshold=0.5, mode="default", prediction=1,
                               inference_ms=1.0)          # truth unknown
        stats = models.prediction_stats(db)
        assert stats["total"] == 2
        assert stats["n_with_known_truth"] == 1
        assert stats["accuracy_so_far"] == 1.0

    def test_a_signal_survives_the_round_trip_to_the_database(self, db):
        sig = np.random.default_rng(2).normal(0, 1, 1000).astype(np.float32)
        db.execute("INSERT INTO recordings (id, signal, n_samples, label, fold) "
                   "VALUES (?, ?, ?, ?, ?)",
                   (99, models.signal_to_blob(sig), 1000, 1, 10))
        db.commit()
        back = models.blob_to_signal(models.get_recording(db, 99)["signal"])
        assert np.array_equal(sig, back)


class TestDeviceEndpoint:
    def test_a_valid_window_is_scored(self, client):
        resp = client.post("/api/predict",
                           json={"signal": [0.1] * 1000, "mode": "high_recall"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["prediction"] == 1
        assert body["threshold"] == 0.2736

    @pytest.mark.parametrize("signal,expected", [
        ([0.1] * 999, "exactly"),
        ([0.1] * 1001, "exactly"),
        ("not a list", "signal"),
    ])
    def test_a_malformed_window_is_refused(self, client, signal, expected):
        resp = client.post("/api/predict", json={"signal": signal})
        assert resp.status_code == 400
        assert expected in resp.get_json()["error"]

    def test_an_unknown_operating_point_is_refused(self, client):
        resp = client.post("/api/predict",
                           json={"signal": [0.1] * 1000, "mode": "nonsense"})
        assert resp.status_code == 400

    def test_the_incoming_signal_is_not_retained(self, client, db):
        """A known limitation, asserted so that it is a decision, not a drift.

        Device readings cannot be replayed or re-scored later because the
        signal is discarded. If this ever changes, this test should fail and be
        updated deliberately.
        """
        client.post("/api/predict", json={"signal": [0.1] * 1000})
        row = db.execute("SELECT recording_id FROM predictions").fetchone()
        assert row["recording_id"] is None


class TestOperatingPoints:
    def test_rescore_does_not_write_a_row(self, auth_client, db):
        """Looking at a result at another operating point must not create one.

        This previously logged a new device reading on every click, inflating
        the record count and discarding the known ground truth.
        """
        auth_client.get("/api/record?type=mi&mode=high_recall")
        before = models.prediction_stats(db)["total"]

        resp = auth_client.post("/api/rescore",
                                json={"signal": [0.1] * 1000, "mode": "best_f1"})
        assert resp.status_code == 200
        assert models.prediction_stats(db)["total"] == before

    def test_replay_draws_only_from_the_unseen_test_fold(self, auth_client, db):
        for _ in range(4):
            assert auth_client.get("/api/record").status_code == 200
        rows = db.execute(
            "SELECT DISTINCT r.fold FROM predictions p "
            "JOIN recordings r ON r.id = p.recording_id").fetchall()
        assert [r["fold"] for r in rows] == [10]

    def test_the_served_thresholds_come_from_the_loaded_model(self, client):
        """The defect this guards against served an ensemble at another
        model's cut-offs, and nothing failed visibly."""
        body = client.get("/api/health").get_json()
        assert body["thresholds"]["high_recall"] == 0.2736
        assert body["test_roc_auc"] == 0.9079
