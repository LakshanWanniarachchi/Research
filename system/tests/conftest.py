"""
Shared test fixtures.

Two things have to be true before any test runs.

**The real database must not be touched.** `ecg.db` holds 14,538 recordings and
the prediction history that the thesis reports. Every test therefore runs
against a temporary database created for that test, by repointing
`models.DB_PATH` before anything imports the application.

**The real model must not be loaded.** Loading five Keras models takes tens of
seconds and needs the .keras files present, which would make the suite slow and
would couple it to trained artefacts that are regenerated. A stub model with a
controllable output is substituted instead, so the tests exercise the
application's logic rather than TensorFlow's. The two tests that genuinely need
real preprocessing call it directly.
"""

import os
import sys

import numpy as np
import pytest

# The system modules live one directory up.
HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = os.path.dirname(HERE)
sys.path.insert(0, SYSTEM)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# The developer's .env turns the MQTT transport on, and config.py loads it into
# the environment when app.py is imported. A test run must not open a socket to
# a real broker: the suite has to pass on a machine with no broker at all, and
# a test that quietly depends on one is a test that fails for the next person.
# Set before any project module is imported, because these are read at import.
os.environ["MQTT_ENABLED"] = "0"
# A known password, so tests can sign in. auth.ensure_first_user reads this.
os.environ["ECG_ADMIN_PASSWORD"] = "test-password-123"
os.environ["ECG_SECRET_KEY"] = "test-secret-key-not-for-production"


class StubModel:
    """Stands in for a Keras model.

    Returns a probability that the test controls, so that threshold behaviour
    can be checked at exact boundaries rather than at whatever a real network
    happens to output.
    """

    def __init__(self, probability=0.9):
        self.probability = probability
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        return np.array([[self.probability]], dtype=np.float32)


STUB_CONFIG = {
    "task": "test",
    "sample_rate_hz": 100,
    "input_length": 1000,
    "preprocessing": "per-recording z-score",
    "threshold_default": 0.5,
    "threshold_best_f1": 0.4433,
    "threshold_high_recall": 0.2736,
    "test_roc_auc": 0.9079,
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A temporary database with the schema and a little seeded data."""
    import models

    path = str(tmp_path / "test.db")
    monkeypatch.setattr(models, "DB_PATH", path)

    conn = models.connect()
    models.create_tables(conn)

    # A handful of recordings in the test fold so replay has something to draw.
    rng = np.random.default_rng(0)
    for i in range(6):
        signal = rng.normal(0, 1, 1000).astype(np.float32)
        conn.execute(
            "INSERT INTO recordings (id, signal, n_samples, label, fold) "
            "VALUES (?, ?, ?, ?, ?)",
            (i, models.signal_to_blob(signal), 1000, i % 2, 10),
        )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def seeded_patient(db):
    """One enrolled patient holding a device and a token."""
    import models

    pid = models.create_patient(db, name="Test Patient", age=61, sex="M",
                                device_id="ESP32-TEST")
    models.set_device_token(db, pid, "secret-token-abc")
    return models.get_patient(db, pid)


@pytest.fixture
def stub_model():
    """The stub the application will serve."""
    return StubModel(probability=0.9)


@pytest.fixture
def client(db, stub_model, monkeypatch):
    """A Flask test client with the model stubbed and the temp database live."""
    import model as mdl

    monkeypatch.setattr(mdl, "load_trained_model",
                        lambda *a, **k: ([stub_model], dict(STUB_CONFIG)))
    # A fresh process flag, so warm-up behaviour is deterministic per test.
    monkeypatch.setattr(mdl, "_warmed_up", False)
    monkeypatch.setattr(mdl, "_bandpass_enabled", False)

    # Import late: app.py loads the model at import time, so the patch above
    # has to be in place first. Drop any cached copy from a previous test.
    for name in ("app",):
        sys.modules.pop(name, None)
    import app as app_module

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        c.application_module = app_module
        yield c


@pytest.fixture
def auth_client(client):
    """A test client that has already signed in as admin."""
    resp = client.post("/login", data={"username": "admin",
                                       "password": "test-password-123"})
    assert resp.status_code in (302, 200), "fixture could not sign in"
    return client
