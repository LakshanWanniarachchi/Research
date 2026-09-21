"""
STEP 2  -  Train the 1D CNN and choose the decision threshold
============================================================

Run this AFTER 1_extract_data.py. It takes roughly 10-20 minutes on a
normal laptop (no graphics card needed).

What it does:
  1. Reads all the recordings back OUT of the SQLite database.
  2. Splits them using the official PTB-XL fold numbers.
  3. Trains the CNN from model.py.
  4. Chooses the decision threshold using the VALIDATION fold only.
  5. Tests once on the TEST fold and prints the scores.
  6. Saves model.keras and model_config.json for the web app to use.

The two rules that keep this honest
-----------------------------------
RULE 1 - the same patient must never be in two groups.
    PTB-XL already numbered every recording 1 to 10 with this in mind, so we
    just use those numbers:
        folds 1-8  -> training    (the model learns from these)
        fold  9    -> validation  (used to decide when to stop, and to pick
                                   the threshold)
        fold  10   -> test        (touched exactly once, at the very end)

RULE 2 - never choose anything by looking at the test fold.
    The threshold is picked on fold 9 and then applied UNCHANGED to fold 10.
    If we tried thresholds on fold 10 and kept the best one, the score would
    be flattering and meaningless.
"""

import os
import json

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

import models      # our database file
import model       # our CNN file

# Fix the random numbers so re-running gives the same answer.
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]

# Apply the 0.5-40 Hz band-pass filter described in the methodology.
#
# Set this to False only to reproduce the older z-score-only models. Whatever it
# is set to is written into model_config.json, and the web app reads it back and
# preprocesses live signals the same way - so training and serving cannot end up
# disagreeing about whether the filter was used.
USE_BANDPASS = True

# ---------------------------------------------------------------------------
# 1. Load the recordings from the database
# ---------------------------------------------------------------------------
conn = models.connect()

if models.count_recordings(conn) == 0:
    raise SystemExit("The database is empty. Run:  python 1_extract_data.py")

print("Loading recordings from the database ...")
X_train_raw, y_train = models.load_fold(conn, TRAIN_FOLDS)
X_val_raw,   y_val   = models.load_fold(conn, VAL_FOLDS)
X_test_raw,  y_test  = models.load_fold(conn, TEST_FOLDS)
conn.close()

print(f"  training  : {len(y_train):5d} recordings ({int(y_train.sum())} MI)")
print(f"  validation: {len(y_val):5d} recordings ({int(y_val.sum())} MI)")
print(f"  test      : {len(y_test):5d} recordings ({int(y_test.sum())} MI)")

# ---------------------------------------------------------------------------
# 2. Clean the signals (exactly the same way the web app will)
# ---------------------------------------------------------------------------
# Set the filter BEFORE preprocessing anything, and record the same setting in
# model_config.json further down, so the app reproduces it at serving time.
model.set_bandpass(USE_BANDPASS)
if USE_BANDPASS:
    print(f"\nPreprocessing: band-pass {model.BANDPASS_LOW_HZ}-"
          f"{model.BANDPASS_HIGH_HZ} Hz (order {model.BANDPASS_ORDER}, "
          f"zero-phase), then per-recording z-score")
else:
    print("\nPreprocessing: per-recording z-score only (no band-pass)")

X_train = model.preprocess_many(X_train_raw)
X_val = model.preprocess_many(X_val_raw)
X_test = model.preprocess_many(X_test_raw)

# ---------------------------------------------------------------------------
# 3. Class weights
# ---------------------------------------------------------------------------
# There are far more Normal recordings than MI ones. Without this, the model
# could score well simply by answering "Normal" every single time. Class
# weights make each MI mistake cost more, so that shortcut stops working.
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print(f"\nClass weights: Normal {class_weight[0]:.3f}, MI {class_weight[1]:.3f}")

# ---------------------------------------------------------------------------
# 4. Build and train
# ---------------------------------------------------------------------------
cnn = model.build_model(X_train.shape[1])
cnn.summary()

# EarlyStopping watches the validation loss. When it stops improving for 5
# epochs in a row, training stops and the best version is put back. This
# saves time and stops the model over-fitting (memorising the training set).
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
]

print("\nTraining ... (this is the slow part)")
history = cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2,          # one line per epoch, no moving progress bar
)

# ---------------------------------------------------------------------------
# 5. Choose the threshold  -  ON THE VALIDATION FOLD ONLY
# ---------------------------------------------------------------------------
# The model outputs a probability between 0 and 1. To turn that into a yes/no
# answer we need a cut-off. The obvious choice, 0.5, is the WRONG one for
# medicine: missing a real heart attack is far worse than a false alarm, and
# 0.5 treats both mistakes as equally bad.
#
# So we try many cut-offs on the validation fold and keep two of them.
val_probs = cnn.predict(X_val, verbose=0).ravel()

candidates = np.arange(0.01, 1.00, 0.01)


def scores_at(threshold, probs, truth):
    """Work out recall and F1 if we used this cut-off."""
    preds = (probs >= threshold).astype(int)
    return (recall_score(truth, preds, zero_division=0),
            f1_score(truth, preds, zero_division=0))


# (a) best-F1: the most balanced trade-off between false alarms and misses.
best_f1_threshold = max(candidates, key=lambda t: scores_at(t, val_probs, y_val)[1])

# (b) high-recall: catch at least 90% of the real heart attacks.
#     A LOWER cut-off shouts "MI" more readily, so it misses fewer real cases
#     but raises more false alarms. Among all the cut-offs that reach the
#     target we keep the HIGHEST one, because that meets the safety goal with
#     the fewest false alarms. This is the setting a screening tool should use.
TARGET_RECALL = 0.90
reaching_target = [t for t in candidates
                   if scores_at(t, val_probs, y_val)[0] >= TARGET_RECALL]
high_recall_threshold = max(reaching_target) if reaching_target else best_f1_threshold

THRESHOLDS = {
    "default": 0.5,
    "best_f1": round(float(best_f1_threshold), 4),
    "high_recall": round(float(high_recall_threshold), 4),
}

print("\nThresholds chosen on the VALIDATION fold (fold 9):")
for name, t in THRESHOLDS.items():
    r, f = scores_at(t, val_probs, y_val)
    print(f"  {name:12s} = {t:.4f}   (validation recall {r:.4f}, F1 {f:.4f})")

# ---------------------------------------------------------------------------
# 6. The real exam: the test fold, used once
# ---------------------------------------------------------------------------
test_probs = cnn.predict(X_test, verbose=0).ravel()

# These two describe the model at EVERY threshold, so they do not depend on
# the cut-off we picked at all.
roc_auc = roc_auc_score(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)

lines = []
lines.append("=" * 62)
lines.append("TEST-SET RESULTS  (fold 10 - never used for training or tuning)")
lines.append("=" * 62)
lines.append(f"ROC-AUC : {roc_auc:.4f}   (how well the two classes separate)")
lines.append(f"PR-AUC  : {pr_auc:.4f}   (the fairer headline on uneven data)")

for name, threshold in THRESHOLDS.items():
    preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    lines.append("")
    lines.append(f"--- operating point: {name}  (threshold {threshold:.4f}) ---")
    lines.append(f"  Accuracy   : {accuracy_score(y_test, preds):.4f}")
    lines.append(f"  Precision  : {precision_score(y_test, preds, zero_division=0):.4f}"
                 "   (of the MI alarms, how many were right)")
    lines.append(f"  Recall     : {recall_score(y_test, preds, zero_division=0):.4f}"
                 "   (of the real MIs, how many we caught)")
    lines.append(f"  Specificity: {tn / (tn + fp):.4f}"
                 "   (of the healthy, how many we left alone)")
    lines.append(f"  F1-score   : {f1_score(y_test, preds, zero_division=0):.4f}")
    lines.append("")
    lines.append("  Confusion matrix (rows = truth, columns = what we said):")
    lines.append("                   said Normal   said MI")
    lines.append(f"    truly Normal      {tn:6d}    {fp:6d}")
    lines.append(f"    truly MI          {fn:6d}    {tp:6d}")
    lines.append(f"  MISSED HEART ATTACKS: {fn}   <- the most costly mistake")

report = "\n".join(lines)
print("\n" + report)

with open(os.path.join(HERE, "test_results.txt"), "w", encoding="utf-8") as f:
    f.write(report + "\n")

# ---------------------------------------------------------------------------
# 7. Save the model and its settings
# ---------------------------------------------------------------------------
cnn.save(model.MODEL_PATH)

config = {
    "task": "PTB-XL Lead I, Normal (0) vs Myocardial Infarction (1)",
    "sample_rate_hz": model.SAMPLE_RATE_HZ,
    "input_length": int(X_train.shape[1]),
    "preprocessing": (
        f"band-pass {model.BANDPASS_LOW_HZ}-{model.BANDPASS_HIGH_HZ} Hz "
        f"(Butterworth order {model.BANDPASS_ORDER}, zero-phase), then "
        f"per-recording z-score: (x - mean) / (std + 1e-8)"
        if USE_BANDPASS else
        "per-recording z-score: (x - mean) / (std + 1e-8)"
    ),
    # Read back by model.load_trained_model() so the app filters live signals
    # exactly as this run filtered its training data.
    "bandpass": {
        "enabled": bool(USE_BANDPASS),
        "low_hz": model.BANDPASS_LOW_HZ,
        "high_hz": model.BANDPASS_HIGH_HZ,
        "order": model.BANDPASS_ORDER,
        "zero_phase": True,
    },
    "threshold_default": THRESHOLDS["default"],
    "threshold_best_f1": THRESHOLDS["best_f1"],
    "threshold_high_recall": THRESHOLDS["high_recall"],
    "thresholds_chosen_on": "validation fold 9 only",
    "split": {"train": "folds 1-8", "val": "fold 9", "test": "fold 10"},
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "epochs_run": int(len(history.history["loss"])),
    "test_roc_auc": round(float(roc_auc), 4),
    "test_pr_auc": round(float(pr_auc), 4),
    "seed": 42,
}

with open(model.CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# ---------------------------------------------------------------------------
# 8. Save the per-epoch training history
# ---------------------------------------------------------------------------
# Keras keeps loss and accuracy for every epoch, for both training and
# validation, but throws it away when the script exits. Earlier runs saved only
# a rendered training-curves PNG, which meant the curves could not be re-plotted
# at a different size, the train/validation gap could not be quoted as a number,
# and the epoch at which early stopping restored the best weights could not be
# recovered. Writing the raw numbers costs a few kilobytes and makes all of that
# possible afterwards.
history_path = os.path.join(HERE, "training_history.json")

history_record = {
    "epochs_run": int(len(history.history["loss"])),
    "seed": 42,
    "batch_size": 64,
    "max_epochs": 30,
    "early_stopping": {"monitor": "val_loss", "patience": 5,
                       "restore_best_weights": True},
    "bandpass_enabled": bool(USE_BANDPASS),
    # The epoch whose weights were restored - the lowest validation loss.
    "best_epoch_1indexed": int(np.argmin(history.history["val_loss"]) + 1),
    "best_val_loss": float(np.min(history.history["val_loss"])),
    # One list per metric, one entry per epoch, in order.
    "per_epoch": {k: [float(v) for v in vals]
                  for k, vals in history.history.items()},
}

with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history_record, f, indent=2)

print(f"\nPer-epoch history ({history_record['epochs_run']} epochs) written to "
      f"{os.path.basename(history_path)}")
print(f"  best epoch: {history_record['best_epoch_1indexed']} "
      f"(val_loss {history_record['best_val_loss']:.4f})")

print("\nSaved:")
print(f"  {model.MODEL_PATH}    (the trained model)")
print(f"  {model.CONFIG_PATH}   (thresholds and settings)")
print(f"  {os.path.join(HERE, 'test_results.txt')}  (the scores above)")
print("\nNext step:  python app.py")
