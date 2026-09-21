"""
STEP 3  -  The Flask web application
====================================

Run this AFTER 1_extract_data.py and 2_train_model.py:

    python app.py

then open  http://127.0.0.1:5000  in a browser.

Where this sits in the whole system
-----------------------------------
    AD8232 sensor -> ESP32 -> [ Flask server -> 1D CNN ] -> web dashboard
                                    ^^^^^^ this file ^^^^^^

The ESP32 hardware is not built yet, so the app runs in SIMULATION MODE: it
replays a real recording out of the SQLite database and pushes it through
exactly the same code as a live signal would use. When the hardware is ready
it will POST to /api/predict and nothing else has to change.

Every decision the app makes - simulated or real - is written into the
`predictions` table, so there is a permanent record of what the system did
and how long it took.

IMPORTANT: simulation only ever uses FOLD 10, the test fold the model never
saw while training. Demoing on training data would look impressive and mean
nothing.
"""

import json
import os
import secrets
import sqlite3
import sys
import time
from datetime import timedelta

# Started as `python app.py`, this module is named __main__, and a later
# `import app` (web_ecg.py does one inside the live endpoint) would build a
# second copy: five models loaded again and a second MQTT client under the
# same client id. The broker then disconnects each in turn - "session taken
# over" every second - and no ECG chunk is ever delivered. Registering this
# module as `app` makes every import return the one already running.
if __name__ == "__main__":
    sys.modules.setdefault("app", sys.modules["__main__"])

from flask import Flask, jsonify, render_template, request

import auth
import config       # .env -> environment, before anything reads it
import live_serial  # the USB-serial acquisition session
import models     # our database file
import model      # our CNN file
import mqtt_bridge
import web_account
import web_ecg

app = Flask(__name__)

# Sessions are signed with this key. A fixed default would mean anyone with the
# source could forge a session cookie, so it comes from the environment and is
# generated per-process if unset. Generating it means sessions do not survive a
# restart, which is correct for a prototype and wrong for production; the
# deployment note in the README says to set it explicitly.
app.secret_key = config.SECRET_KEY or secrets.token_hex(32)
if not config.SECRET_KEY:
    print("  note: ECG_SECRET_KEY not set, using a per-process key "
          "(sessions end at restart)")

app.permanent_session_lifetime = timedelta(minutes=auth.SESSION_MINUTES)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,   # JavaScript cannot read the session cookie
    SESSION_COOKIE_SAMESITE="Lax",  # not sent on cross-site POSTs
)
app.register_blueprint(auth.auth_bp)
app.register_blueprint(web_account.account_bp)
app.register_blueprint(web_ecg.ecg_bp)

# Make sure the patients table (and the patient_id column on an older
# predictions table) exist before the first request arrives.
_conn = models.connect()
models.create_tables(_conn)
_conn.close()

# ---------------------------------------------------------------------------
# 1. Load the model once, when the server starts
# ---------------------------------------------------------------------------
# Loading takes a few seconds, so we do it here rather than on every request.
print("Loading the trained model ...")
# The settings come back WITH the models, not from a fixed path. Opening
# model_config.json here instead would name the single model's file even when
# the ensemble is what got loaded, and the app would then decide every verdict
# at a threshold tuned for a different network.
CNN, CONFIG = model.load_trained_model()

THRESHOLDS = {
    "default": CONFIG["threshold_default"],
    "best_f1": CONFIG["threshold_best_f1"],
    "high_recall": CONFIG["threshold_high_recall"],
}
INPUT_LENGTH = CONFIG["input_length"]
SAMPLE_RATE = CONFIG["sample_rate_hz"]

# The fold used for simulation. 10 = the unseen test fold.
TEST_FOLD = 10

# Stored with every report, so a result can always be traced to the model that
# produced it. Three trained runs exist with different AUCs; this names the one
# actually being served rather than the best one.
MODEL_VERSION = (f"{'ensemble-5seed' if len(CNN) > 1 else 'single'}"
                 f" roc_auc={CONFIG.get('test_roc_auc')}")

print(f"Serving: {model.describe(CNN)}")
print(f"Thresholds: {THRESHOLDS}")

# A fresh database has no accounts and could not be signed into. Create one and
# print the generated password exactly once.
_conn = models.connect()
_seeded = auth.ensure_first_user(_conn)
_conn.close()
if _seeded:
    print("\n  Created the first account:  username 'admin'")
    print(f"  Password: {_seeded}")
    print("  This is shown once. Sign in and change it." + "\n")



# ---------------------------------------------------------------------------
# 2. The one place a decision is made
# ---------------------------------------------------------------------------
def decide(signal, mode):
    """Run the model on one signal and turn the answer into a verdict.

    Returns the verdict, how long the model took, and whether this was the
    process's first (warm-up) inference - because measuring that speed is one
    of the things this research set out to do, and a 2.4-second graph-tracing
    call would otherwise be averaged in with 0.3-second real ones.
    """
    start = time.perf_counter()
    probability, was_warmup = model.predict_probability(CNN, signal)
    inference_ms = (time.perf_counter() - start) * 1000

    threshold = THRESHOLDS[mode]
    prediction = 1 if probability >= threshold else 0

    verdict = {
        "probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "mode": mode,
        "prediction": prediction,
        "label": "MI PATTERN DETECTED" if prediction else "NORMAL",
        "inference_ms": round(inference_ms, 1),
    }
    return verdict, inference_ms, was_warmup


# ---------------------------------------------------------------------------
# 3. The pages and endpoints
# ---------------------------------------------------------------------------
@app.route("/")
@auth.login_required
def index():
    """The dashboard page."""
    conn = models.connect()
    n_test = sum(models.count_by_label(conn, fold=TEST_FOLD).values())
    n_total = models.count_recordings(conn)
    conn.close()

    return render_template(
        "index.html",
        user=auth.current_user(),
        thresholds=THRESHOLDS,
        n_test=n_test,
        n_total=n_total,
        sample_rate=SAMPLE_RATE,
        roc_auc=CONFIG.get("test_roc_auc"),
    )


@app.route("/api/record")
@auth.login_required
def api_record():
    """SIMULATION: replay one real recording from the unseen test fold.

    Query options:
        type       = any | mi | normal
        mode       = default | best_f1 | high_recall
        patient_id = optional; file this reading against a patient
    """
    want = request.args.get("type", "any")
    mode = request.args.get("mode", "high_recall")
    patient_id = request.args.get("patient_id", type=int)

    if mode not in THRESHOLDS:
        return jsonify({"error": f"mode must be one of {list(THRESHOLDS)}"}), 400

    # Turn the word into the label number the database stores.
    label = {"mi": 1, "normal": 0}.get(want)      # None means "any"

    conn = models.connect()

    if patient_id is not None and models.get_patient(conn, patient_id) is None:
        conn.close()
        return jsonify({"error": f"no patient with id {patient_id}"}), 404

    row = models.random_recording(conn, fold=TEST_FOLD, label=label)
    if row is None:
        conn.close()
        return jsonify({"error": "no matching recording in the database"}), 404

    signal = models.blob_to_signal(row["signal"])
    truth = row["label"]

    result, inference_ms, was_warmup = decide(signal, mode)

    # Write this decision into the database.
    models.save_prediction(
        conn,
        source="simulation",
        probability=result["probability"],
        threshold=result["threshold"],
        mode=mode,
        prediction=result["prediction"],
        inference_ms=inference_ms,
        recording_id=row["id"],
        truth=truth,
        patient_id=patient_id,
        is_warmup=was_warmup,
    )
    conn.close()

    # Add the things only the simulation knows.
    result["recording_id"] = row["id"]
    result["signal"] = [round(float(v), 4) for v in signal]
    result["sample_rate"] = SAMPLE_RATE
    result["truth"] = truth
    result["truth_label"] = "MI" if truth == 1 else "Normal"
    result["correct"] = bool(result["prediction"] == truth)
    result["patient_id"] = patient_id

    return jsonify(result)


@app.route("/api/rescore", methods=["POST"])
@auth.login_required
def api_rescore():
    """Re-run the model on a signal ALREADY on screen, without recording it.

    Why this exists, separately from /api/predict:

    The dashboard lets you flip between the three operating points to see how
    the verdict changes on the trace you are looking at. It used to do that by
    POSTing the trace to /api/predict - which logged each flip as a brand new
    reading from a DEVICE, with truth = NULL. Three clicks on one simulated
    recording therefore wrote three fake 'device' rows and threw away the
    ground truth we actually had.

    That quietly corrupted two of the numbers the research reports: the
    simulation/device split, and accuracy_so_far (which only averages rows
    whose truth is known). This endpoint scores without writing, so looking at
    a result can no longer change the results.
    """
    body = request.get_json(silent=True) or {}
    signal = body.get("signal")
    mode = body.get("mode", "high_recall")

    if not isinstance(signal, list) or len(signal) != INPUT_LENGTH:
        return jsonify({"error": f"signal must be a list of {INPUT_LENGTH} numbers"}), 400
    if mode not in THRESHOLDS:
        return jsonify({"error": f"mode must be one of {list(THRESHOLDS)}"}), 400

    try:
        result, _, _ = decide(signal, mode)
    except (ValueError, TypeError):
        return jsonify({"error": "the signal must contain only numbers"}), 400

    # Nothing is saved. That is the entire point.
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """LIVE: the endpoint the ESP32 POSTs to.

    Send JSON:  {"signal": [ ...1000 numbers... ],
                 "mode": "high_recall",
                 "device_id": "ESP32-001"}

    Send the RAW sensor values - the server does the cleaning itself, using
    the same code the model was trained with.

    device_id is optional but strongly recommended: it is how the reading gets
    attached to a patient. The device does not need to know any patient number,
    only its own name, which means a device can be handed to a different
    patient by editing one database row instead of reflashing the firmware.
    """
    body = request.get_json(silent=True) or {}
    signal = body.get("signal")
    mode = body.get("mode", "high_recall")
    device_id = body.get("device_id")

    if not isinstance(signal, list):
        return jsonify({
            "error": "send JSON shaped like {\"signal\": [ ...numbers... ]}"
        }), 400

    if len(signal) != INPUT_LENGTH:
        seconds = INPUT_LENGTH // SAMPLE_RATE
        return jsonify({
            "error": f"signal must be exactly {INPUT_LENGTH} numbers "
                     f"({seconds} seconds at {SAMPLE_RATE} Hz), "
                     f"but {len(signal)} were sent"
        }), 400

    if mode not in THRESHOLDS:
        return jsonify({"error": f"mode must be one of {list(THRESHOLDS)}"}), 400

    try:
        result, inference_ms, was_warmup = decide(signal, mode)
    except (ValueError, TypeError):
        return jsonify({"error": "the signal must contain only numbers"}), 400

    # Log it. There is no `truth` here: a real patient's diagnosis is unknown,
    # which is the entire point of running the model.
    conn = models.connect()

    # Work out whose reading this is. An unknown device_id is NOT an error -
    # the reading is still valid and still worth storing, it just is not
    # attributed to anyone. Throwing it away would be the worse failure.
    # How a device proves it is the device it claims to be. When token checking
    # is on, an unknown id and a wrong token are refused identically, so a
    # caller cannot discover valid device ids by watching the error change.
    if device_id and mqtt_bridge.REQUIRE_DEVICE_TOKEN:
        patient = models.get_patient_by_device_token(
            conn, device_id, body.get("token"))
        if patient is None:
            conn.close()
            return jsonify({"error": "device id or token not recognised"}), 401
    elif device_id:
        patient = models.get_patient_by_device(conn, device_id)
    else:
        patient = None
    patient_id = patient["id"] if patient else None

    models.save_prediction(
        conn,
        source="device",
        probability=result["probability"],
        threshold=result["threshold"],
        mode=mode,
        prediction=result["prediction"],
        inference_ms=inference_ms,
        patient_id=patient_id,
        is_warmup=was_warmup,
    )
    conn.close()

    result["device_id"] = device_id
    result["patient_id"] = patient_id
    result["patient_name"] = patient["name"] if patient else None
    if device_id and patient is None:
        result["warning"] = (f"device '{device_id}' is not registered to any "
                             f"patient; the reading was stored unattributed")

    return jsonify(result)


# ---------------------------------------------------------------------------
# 3b. Live acquisition over USB serial
# ---------------------------------------------------------------------------
# The device that streams over USB is the same device the /api/predict endpoint
# was written for, so a live window takes the same decide() call and is stored
# the same way. What differs is only how the samples arrive.
LIVE_DEVICE_ID = os.environ.get("ECG_LIVE_DEVICE_ID", "ESP32-001")


def score_and_store(device_id, window, mode, source):
    """Score one ten-second window and file it. The only scorer in the system.

    USB and MQTT differ in how the samples arrive and in nothing else, so they
    share this: a recording must not depend on its transport for what it is
    worth. `window` is a Window from the ingester, or a bare list of samples
    when the caller has nothing else to give.
    """
    values = [float(v) for v in getattr(window, "values", window)]
    quality = getattr(window, "quality", {}) or {}
    measured_rate = getattr(window, "measured_rate_hz", None)

    if mode not in THRESHOLDS:
        mode = "high_recall"
    result, inference_ms, was_warmup = decide(values, mode)

    conn = models.connect()
    try:
        device = models.get_device(conn, device_id)
        models.touch_device(conn, device_id, status="receiving")

        # The research record, unchanged: one row per decision.
        models.save_prediction(
            conn, source="device",
            probability=result["probability"], threshold=result["threshold"],
            mode=mode, prediction=result["prediction"],
            inference_ms=inference_ms, patient_id=None, is_warmup=was_warmup,
        )

        result["device_id"] = device_id
        result["source"] = source
        result["bpm"] = quality.get("bpm")
        result["beat_strength"] = quality.get("beat_strength")
        result["measured_rate_hz"] = round(measured_rate, 2) if measured_rate else None
        result["at"] = time.strftime("%H:%M:%S")

        if device is None:
            result["warning"] = (f"device '{device_id}' is not linked to any "
                                 f"account, so no report was stored")
            return result

        # The user-facing record: the waveform too, so the report can be
        # reopened and checked.
        result["report_id"] = models.save_ecg_report(
            conn, user_id=device["user_id"], signal=values,
            probability=result["probability"], threshold=result["threshold"],
            mode=mode, prediction=result["prediction"], source=source,
            device_id=device_id, sample_rate=SAMPLE_RATE,
            measured_rate=measured_rate, bpm=quality.get("bpm"),
            beat_strength=quality.get("beat_strength"),
            quality_json=json.dumps(quality),
            model_name=model.describe(CNN),
            model_version=MODEL_VERSION, inference_ms=inference_ms,
        )
    finally:
        conn.close()
    return result


def score_live_window(window, mode="high_recall"):
    """A window that arrived over USB serial."""
    return score_and_store(LIVE_DEVICE_ID, window, mode, source="serial")


LIVE = live_serial.LiveSession(score_window=score_live_window)


# ---------------------------------------------------------------------------
# 3c. Live acquisition over Wi-Fi and MQTT
# ---------------------------------------------------------------------------
# The same window, arriving by a different road. The samples are reassembled
# from one-second chunks by mqtt_stream, and land here once a full ten seconds
# has passed every check. Scoring goes through the same decide() as every other
# path, so a reading is never judged by a second implementation.
def score_mqtt_window(device_id, window, mode=None):
    """A window that arrived over Wi-Fi and MQTT.

    The operating point comes from whatever the live page last selected for
    this device; it moves the threshold, never the probability.
    """
    if mode is None:
        stream = mqtt_bridge.STREAMS.get(device_id, create=False)
        mode = getattr(stream, "mode", "high_recall")
    return score_and_store(device_id, window, mode, source="mqtt")


# Optional transport, started here rather than with the rest of the start-up
# because it is handed score_mqtt_window and that function has to exist first.
# The application runs with or without a broker.
MQTT_CLIENT = mqtt_bridge.start(CNN, THRESHOLDS, INPUT_LENGTH,
                                on_window=score_mqtt_window)
mqtt_bridge.CLIENT = MQTT_CLIENT


@app.route("/live")
@auth.login_required
def live_page():
    """The live ECG page: waveform, signal status, and the latest screening."""
    return render_template("live.html", user=auth.current_user(),
                           device_id=LIVE_DEVICE_ID,
                           thresholds=THRESHOLDS)


@app.route("/api/live/ports")
@auth.login_required
def api_live_ports():
    try:
        return jsonify({"ports": live_serial.available_ports()})
    except Exception as exc:                      # pyserial missing, etc.
        return jsonify({"ports": [], "error": str(exc)})


@app.route("/api/live/start", methods=["POST"])
@auth.login_required
def api_live_start():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "high_recall")
    if mode not in THRESHOLDS:
        return jsonify({"error": f"mode must be one of {list(THRESHOLDS)}"}), 400
    LIVE.mode = mode
    state = LIVE.start(port=body.get("port") or None,
                       save_csv=bool(body.get("save")))
    state["mode"] = mode
    status = 200 if state.get("running") else 400
    return jsonify(state), status


@app.route("/api/live/stop", methods=["POST"])
@auth.login_required
def api_live_stop():
    return jsonify(LIVE.stop())


@app.route("/api/live/data")
@auth.login_required
def api_live_data():
    """Polled by the live page. Small enough to call several times a second."""
    return jsonify(LIVE.snapshot())


# ---------------------------------------------------------------------------
# 4. Patients
# ---------------------------------------------------------------------------
@app.route("/api/patients", methods=["GET", "POST"])
@auth.login_required
def api_patients():
    """GET  - list every patient with their reading counts.
    POST - enrol a new one: {"name": ..., "age": ..., "sex": ..., "device_id": ...}
    """
    conn = models.connect()

    if request.method == "GET":
        rows = [dict(r) for r in models.list_patients(conn)]
        conn.close()
        return jsonify({"patients": rows})

    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        conn.close()
        return jsonify({"error": "a patient needs a name"}), 400

    try:
        patient_id = models.create_patient(
            conn,
            name=name,
            age=body.get("age"),
            sex=body.get("sex"),
            device_id=(body.get("device_id") or "").strip() or None,
            notes=body.get("notes"),
        )
    except sqlite3.IntegrityError:
        # device_id is UNIQUE - this is the "one ESP32, one chest" rule.
        conn.close()
        return jsonify({
            "error": f"device '{body.get('device_id')}' is already assigned "
                     f"to another patient"
        }), 409

    row = dict(models.get_patient(conn, patient_id))
    conn.close()
    return jsonify(row), 201


@app.route("/patient/<int:patient_id>")
@auth.login_required
def patient_page(patient_id):
    """One patient's page: who they are, and every reading taken from them."""
    conn = models.connect()
    patient = models.get_patient(conn, patient_id)
    if patient is None:
        conn.close()
        return render_template("patient.html", patient=None,
                               user=auth.current_user(),
                               thresholds=THRESHOLDS), 404

    rows = models.patient_predictions(conn, patient_id, limit=50)
    stats = models.patient_stats(conn, patient_id)
    conn.close()

    return render_template(
        "patient.html",
        user=auth.current_user(),
        patient=dict(patient),
        predictions=[dict(r) for r in rows],
        stats=stats,
        thresholds=THRESHOLDS,
        sample_rate=SAMPLE_RATE,
    )


# ---------------------------------------------------------------------------
# 5. History and health
# ---------------------------------------------------------------------------
@app.route("/api/history")
@auth.login_required
def api_history():
    """The most recent decisions, straight out of the database."""
    limit = request.args.get("limit", 20, type=int)
    limit = max(1, min(limit, 100))          # keep it sensible

    conn = models.connect()
    rows = models.recent_predictions(conn, limit=limit)
    stats = models.prediction_stats(conn)
    conn.close()

    return jsonify({
        "stats": stats,
        # sqlite3.Row behaves like a dict but is not one, so convert it.
        "predictions": [dict(row) for row in rows],
    })


@app.route("/api/health")
def api_health():
    """A quick check that the server, model and database are all alive."""
    conn = models.connect()
    n_total = models.count_recordings(conn)
    n_test = sum(models.count_by_label(conn, fold=TEST_FOLD).values())
    stats = models.prediction_stats(conn)
    conn.close()

    return jsonify({
        "status": "ok",
        # Asked of the loaded model rather than hard-coded, so this cannot
        # claim "single model" while an ensemble is actually serving.
        "model": model.describe(CNN),
        "test_roc_auc": CONFIG.get("test_roc_auc"),
        "thresholds": THRESHOLDS,
        "input_length": INPUT_LENGTH,
        "sample_rate_hz": SAMPLE_RATE,
        "recordings_in_database": n_total,
        "recordings_in_test_fold": n_test,
        "predictions_made": stats["total"],
    })


@app.route("/dashboard")
@auth.login_required
def dashboard():
    """The signed-in user's home page.

    Their own recordings, their own devices, and the state of the system that
    produced them. Everything on this page is filtered by the signed-in user:
    the totals, the recent list and the device states all come from queries
    that name them, so nothing here can show another account's heart.
    """
    user = auth.current_user()

    conn = models.connect()
    try:
        stats = models.ecg_report_stats(conn, user["id"])
        recent = [dict(r) for r in models.list_ecg_reports(conn, user["id"], limit=8)]
        devices = [dict(r) for r in models.list_devices(conn, user["id"])]
        n_test = sum(models.count_by_label(conn, fold=TEST_FOLD).values())
    finally:
        conn.close()

    # Live device state comes from the ingester rather than the database: the
    # question "is it sending now?" cannot be answered by a stored timestamp.
    for device in devices:
        stream = mqtt_bridge.STREAMS.get(device["device_id"], create=False)
        device["online"] = bool(stream and stream.online)
        device["rate_hz"] = stream.measured_rate_hz() if stream else None

    return render_template(
        "dashboard.html",
        user=user,
        age=web_account.profile_age(user["dob"]),
        stats=stats,
        recent=recent,
        devices=devices,
        n_test=n_test,
        model_name=model.describe(CNN),
        model_version=MODEL_VERSION,
        thresholds=THRESHOLDS,
        bandpass=model.bandpass_enabled(),
        roc_auc=CONFIG.get("test_roc_auc"),
        mqtt_enabled=mqtt_bridge.MQTT_ENABLED,
        mqtt_connected=bool(MQTT_CLIENT and MQTT_CLIENT.is_connected()),
        mqtt_host=mqtt_bridge.MQTT_HOST,
        mqtt_port=mqtt_bridge.MQTT_PORT,
        mqtt_messages=mqtt_bridge.STREAMS.n_messages,
        mqtt_rejected=mqtt_bridge.STREAMS.n_rejected,
        require_token=mqtt_bridge.REQUIRE_DEVICE_TOKEN,
    )


@app.route("/api/status")
@auth.login_required
def api_status():
    """Machine-readable version of the dashboard header."""
    conn = models.connect()
    stats = models.prediction_stats(conn)
    conn.close()
    return jsonify({
        "model": model.describe(CNN),
        "bandpass_enabled": model.bandpass_enabled(),
        "thresholds": THRESHOLDS,
        "test_roc_auc": CONFIG.get("test_roc_auc"),
        "mqtt": {
            "enabled": mqtt_bridge.MQTT_ENABLED,
            "connected": MQTT_CLIENT is not None,
            "host": mqtt_bridge.MQTT_HOST,
            "port": mqtt_bridge.MQTT_PORT,
            "topic": mqtt_bridge.TOPIC_READING,
            "device_token_required": mqtt_bridge.REQUIRE_DEVICE_TOKEN,
        },
        "predictions": stats,
    })


if __name__ == "__main__":
    print("\n  Open  http://127.0.0.1:5000  in your browser")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
