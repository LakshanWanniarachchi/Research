"""
STEP 2 of 2  -  Train and test the 1D CNN (run this after step 1)
================================================================

What this script does, in plain words:
  1. Loads the .npy files that STEP 1 made.
  2. Cleans each signal in a simple way (z-score normalisation).
  3. Splits the data the official PTB-XL way, using the ready-made "fold"
     numbers so the same patient never appears in two groups:
        folds 1-8  -> training   (the model learns from these)
        fold  9    -> validation (used to stop at the right time)
        fold  10   -> test       (never seen during training; the real exam)
  4. Builds a small 1D Convolutional Neural Network (CNN).
  5. Trains it, using "class weights" so it can't cheat by always saying
     "Normal" (there are more Normal recordings than heart-attack ones).
  6. Tests it on the unseen test fold and prints Accuracy, Precision,
     Recall, F1, and ROC-AUC, plus a confusion matrix.
  7. Saves the trained model and the result pictures for your report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                # save pictures to files, no pop-up window
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

# Make results repeatable (same answer each run).
tf.random.set_seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the data made by step 1
# ---------------------------------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X.npy"))      # shape (N, 1000)
y = np.load(os.path.join(DATA_DIR, "y.npy"))      # 0 = Normal, 1 = MI
folds = np.load(os.path.join(DATA_DIR, "folds.npy"))
print("Loaded data:", X.shape)

# ---------------------------------------------------------------------------
# 2. Simple cleaning: z-score normalisation (per recording)
# ---------------------------------------------------------------------------
# Different devices record at different sizes/strengths. We rescale every
# signal so it has mean 0 and standard deviation 1. This lets the model focus
# on the SHAPE of the heartbeat, not how big the numbers happen to be.
mean = X.mean(axis=1, keepdims=True)
std = X.std(axis=1, keepdims=True) + 1e-8          # +tiny number: avoid /0
X = (X - mean) / std

# A Conv1D layer expects a "channel" dimension, so reshape (N, 1000) -> (N, 1000, 1)
X = X[..., np.newaxis]

# ---------------------------------------------------------------------------
# 3. Split into train / validation / test using the fold numbers
# ---------------------------------------------------------------------------
train_mask = np.isin(folds, [1, 2, 3, 4, 5, 6, 7, 8])
val_mask = folds == 9
test_mask = folds == 10

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train: {len(y_train)}   Val: {len(y_val)}   Test: {len(y_test)}")

# ---------------------------------------------------------------------------
# 4. Class weights: help the smaller (heart-attack) group count more
# ---------------------------------------------------------------------------
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print("Class weights:", class_weight)

# ---------------------------------------------------------------------------
# 5. Build the 1D CNN  (small and easy to understand)
# ---------------------------------------------------------------------------
# Each "block" looks for shapes in the signal and then shrinks it a bit:
#   Conv1D          -> slides small filters along the ECG to find patterns
#   BatchNorm       -> keeps the numbers in a healthy range (trains faster)
#   ReLU activation -> lets the network learn non-straight-line patterns
#   MaxPooling1D    -> halves the length, keeping the strongest signals
def build_model(input_length):
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),

        layers.Conv1D(16, kernel_size=7, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),

        layers.Conv1D(32, kernel_size=5, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),

        # Squeeze the whole signal into 64 summary numbers.
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),                       # turns off half the neurons
                                                   # during training -> less
                                                   # over-fitting
        layers.Dense(1, activation="sigmoid"),     # one number: chance of MI
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model(X.shape[1])
model.summary()

# ---------------------------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------------------------
# EarlyStopping  -> stop when the validation score stops improving and go back
#                   to the best version (saves time, avoids over-fitting).
# ModelCheckpoint-> always keep the best model on disk.
best_model_path = os.path.join(OUT_DIR, "mi_cnn_model.keras")
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5,
                            restore_best_weights=True),
    callbacks.ModelCheckpoint(best_model_path, monitor="val_loss",
                              save_best_only=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=1,
)

# ---------------------------------------------------------------------------
# 7. Test on the unseen test fold and report the scores
# ---------------------------------------------------------------------------
# The model outputs a probability (0..1). We call it "MI" if it is >= 0.5.
probs = model.predict(X_test).ravel()
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)          # recall = how many real MIs we caught
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)
cm = confusion_matrix(y_test, preds)

report_lines = [
    "==================  TEST-SET RESULTS  ==================",
    f"Accuracy : {acc:.4f}",
    f"Precision: {prec:.4f}   (of the MI alarms, how many were right)",
    f"Recall   : {rec:.4f}   (of the real MIs, how many we caught)  <-- most important",
    f"F1-score : {f1:.4f}",
    f"ROC-AUC  : {auc:.4f}",
    "",
    "Confusion matrix (rows = truth, columns = prediction):",
    "                 pred Normal   pred MI",
    f"  true Normal      {cm[0,0]:6d}     {cm[0,1]:6d}",
    f"  true MI          {cm[1,0]:6d}     {cm[1,1]:6d}",
    "",
    classification_report(y_test, preds, target_names=["Normal", "MI"]),
]
report = "\n".join(report_lines)
print("\n" + report)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# ---------------------------------------------------------------------------
# 8. Save pictures for the report
# ---------------------------------------------------------------------------
# (a) Training curves: accuracy and loss over time.
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy over epochs")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss over epochs")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=120)
plt.close()

# (b) Confusion matrix as a coloured grid.
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix")
plt.xticks([0, 1], ["Normal", "MI"]); plt.yticks([0, 1], ["Normal", "MI"])
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="black", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=120)
plt.close()

print("\nSaved model and pictures to:", OUT_DIR)
print("  - mi_cnn_model.keras   (the trained model, for the Flask app later)")
print("  - metrics.txt          (the scores above)")
print("  - training_curves.png  (accuracy/loss graphs)")
print("  - confusion_matrix.png (truth vs prediction grid)")
