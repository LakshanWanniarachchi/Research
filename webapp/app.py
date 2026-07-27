"""
Flask test app  -  "simulation mode" for the heart attack detector
==================================================================

This is the server layer from the thesis system diagram:

    AD8232 -> ESP32 -> [ Flask server -> 1D CNN ] -> web dashboard

The hardware does not exist yet, so this app runs in SIMULATION MODE: instead of
reading a live signal from the ESP32, it replays a real ECG recording taken from
the PTB-XL dataset and sends it through exactly the same code path the live
signal will use later.

Two endpoints matter:

  GET  /api/record   picks a random recording from the dataset (simulation)
  POST /api/predict  takes a raw 1000-point signal  <- this is the one the
                     ESP32 will call once the hardware is ready

Both funnel into the same predict() function, so whatever the demo shows is
genuinely what the device will get.

IMPORTANT: only recordings from FOLD 10 are used. That is the test fold, which
the model never saw while training. Demoing on training data would show
falsely confident results.

Run:
    pip install flask
    python app.py
    then open http://127.0.0.1:5000
"""

import os
import ast
import json
import random

import numpy as np
from flask import Flask, jsonify, render_template, request

# TensorFlow prints a lot of start-up noise; quieten it before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

MODEL_PATH = os.path.join(PROJECT, "mi_cnn_model.keras")
CONFIG_PATH = os.path.join(PROJECT, "model_config.json")
DATA_PATH = os.path.join(PROJECT, "data", "ptbxl_leadI.npz")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Load the model and its settings (once, at start-up)
# ---------------------------------------------------------------------------
# If the 5-model ensemble folder has been downloaded from Drive, use it.
# Otherwise fall back to the single baseline model. Nothing else in this file
# has to change, because predict() just averages over whatever got loaded.
ENSEMBLE_DIR = os.path.join(PROJECT, "ensemble")
ENSEMBLE_CFG = os.path.join(ENSEMBLE_DIR, "ensemble_config.json")

if os.path.exists(ENSEMBLE_CFG):
    CONFIG = json.load(open(ENSEMBLE_CFG))
    print(f"Loading ensemble of {len(CONFIG['members'])} models ...")
    MODELS = [tf.keras.models.load_model(os.path.join(ENSEMBLE_DIR, m))
              for m in CONFIG["members"]]
    MODEL_NAME = f"ensemble of {len(MODELS)} models"
else:
    CONFIG = json.load(open(CONFIG_PATH))
    print("Loading single model ...")
    MODELS = [tf.keras.models.load_model(MODEL_PATH)]
    MODEL_NAME = "single model (no ensemble found)"

# The three operating points worked out in the notebook. The default 0.5 is
# kept only so the dashboard can show why it is a bad choice for this job.
THRESHOLDS = {
    "default": CONFIG["threshold_default"],
    "best_f1": CONFIG["threshold_best_f1"],
    "high_recall": CONFIG["threshold_high_recall"],
}
print(f"Loaded {MODEL_NAME}. Thresholds: {THRESHOLDS}")

# ---------------------------------------------------------------------------
# 2. Load the recordings used for the simulation
# ---------------------------------------------------------------------------
_d = np.load(DATA_PATH)
X_ALL, Y_ALL, FOLDS = _d["X"], _d["y"], _d["folds"]

# Test fold only - see the note at the top of this file.
TEST_IDX = np.where(FOLDS == 10)[0]


# ---------------------------------------------------------------------------
# 3. Attach the original PTB-XL details (age, sex, doctor's report, MI type)
# ---------------------------------------------------------------------------
# Nice to have, not essential: if the raw dataset folder is missing the app
# still runs, it just shows fewer details.
def load_metadata():
    """Rebuild the link from row number -> original PTB-XL record."""
    dataset_name = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    for base in (os.path.join(os.path.dirname(PROJECT), dataset_name),
                 os.path.join(os.path.dirname(PROJECT), dataset_name, dataset_name)):
        if os.path.exists(os.path.join(base, "ptbxl_database.csv")):
            break
    else:
        print("PTB-XL folder not found - running without patient details.")
        return None

    import pandas as pd
    db = pd.read_csv(os.path.join(base, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(base, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]
    to_class = scp.diagnostic_class.to_dict()
    to_sub = scp.diagnostic_subclass.to_dict()

    # Repeat step 1's filter in the SAME order, so row i here is row i in X.
    meta = []
    for ecg_id, row in db.iterrows():
        classes = {to_class[c] for c in row.scp_codes if c in to_class}
        if classes == {"NORM"}:
            label = 0
        elif "MI" in classes:
            label = 1
        else:
            continue
        subs = sorted({to_sub[c] for c in row.scp_codes
                       if c in to_sub and to_class.get(c) == "MI"})
        meta.append({
            "ecg_id": int(ecg_id),
            "label": label,
            "age": None if np.isnan(row.age) else int(row.age),
            "sex": "male" if row.sex == 0 else "female",
            "subclass": ", ".join(subs) if subs else None,
            "report": str(row.report)[:200] if isinstance(row.report, str) else None,
        })

    # Safety check: the rebuilt labels must match the saved ones exactly,
    # otherwise the details would be attached to the wrong recordings.
    if len(meta) != len(Y_ALL) or any(m["label"] != int(y) for m, y in zip(meta, Y_ALL)):
        print("Metadata did not line up with the saved data - skipping details.")
        return None
    print(f"Patient details loaded and verified for {len(meta)} recordings.")
    return meta


META = load_metadata()


# ---------------------------------------------------------------------------
# 4. The prediction path - used by BOTH the simulation and the real device
# ---------------------------------------------------------------------------
def preprocess(signal):
    """Exactly the cleaning used in training. If this ever stops matching the
    notebook, the live predictions silently become wrong."""
    sig = np.asarray(signal, dtype=np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)      # per-recording z-score
    return sig.reshape(1, -1, 1)                        # (batch, time, channel)


def predict(signal):
    """Return the probability of MI for one 1000-point Lead I signal.

    With the ensemble loaded this averages the 5 members, which is exactly what
    the notebook did when it measured the ensemble's score. With a single model
    it is just that model's output.
    """
    batch = preprocess(signal)
    probs = [float(m.predict(batch, verbose=0).ravel()[0]) for m in MODELS]
    return float(np.mean(probs))


def verdict(prob, mode="high_recall"):
    """Turn the probability into a decision at the chosen operating point."""
    thr = THRESHOLDS[mode]
    return {
        "probability": round(prob, 4),
        "threshold": round(thr, 4),
        "mode": mode,
        "prediction": 1 if prob >= thr else 0,
        "label": "MI PATTERN DETECTED" if prob >= thr else "NORMAL",
    }


# ---------------------------------------------------------------------------
# 5. Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           thresholds=THRESHOLDS,
                           n_test=len(TEST_IDX),
                           model_name=MODEL_NAME,
                           has_meta=META is not None)


@app.route("/api/record")
def api_record():
    """SIMULATION: replay a real recording from the unseen test fold."""
    want = request.args.get("type", "any")            # any | mi | normal
    mode = request.args.get("mode", "high_recall")

    pool = TEST_IDX
    if want == "mi":
        pool = TEST_IDX[Y_ALL[TEST_IDX] == 1]
    elif want == "normal":
        pool = TEST_IDX[Y_ALL[TEST_IDX] == 0]

    i = int(random.choice(pool))
    signal = X_ALL[i]
    prob = predict(signal)

    out = verdict(prob, mode)
    out["signal"] = [round(float(v), 4) for v in signal]
    out["truth"] = int(Y_ALL[i])
    out["truth_label"] = "MI" if Y_ALL[i] == 1 else "Normal"
    out["correct"] = bool(out["prediction"] == out["truth"])
    out["sample_rate"] = CONFIG["sample_rate_hz"]

    if META is not None:
        m = META[i]
        out["ecg_id"] = m["ecg_id"]
        out["age"] = m["age"]
        out["sex"] = m["sex"]
        out["subclass"] = m["subclass"]
        out["report"] = m["report"]

    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """LIVE: the endpoint the ESP32 will POST to.

    Expects JSON:  {"signal": [ ...1000 numbers... ]}
    """
    body = request.get_json(silent=True) or {}
    signal = body.get("signal")
    mode = body.get("mode", "high_recall")

    if not isinstance(signal, list):
        return jsonify({"error": "send JSON like {'signal': [...1000 numbers...]}"}), 400

    need = CONFIG["input_length"]
    if len(signal) != need:
        return jsonify({
            "error": f"signal must be exactly {need} samples "
                     f"({need // CONFIG['sample_rate_hz']} seconds at "
                     f"{CONFIG['sample_rate_hz']} Hz), got {len(signal)}"
        }), 400
    if mode not in THRESHOLDS:
        return jsonify({"error": f"mode must be one of {list(THRESHOLDS)}"}), 400

    try:
        return jsonify(verdict(predict(signal), mode))
    except (ValueError, TypeError):
        return jsonify({"error": "signal must contain only numbers"}), 400


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "model": MODEL_NAME,
                    "n_models": len(MODELS),
                    "thresholds": THRESHOLDS, "test_records": len(TEST_IDX)})


if __name__ == "__main__":
    print("\nOpen  http://127.0.0.1:5000  in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
