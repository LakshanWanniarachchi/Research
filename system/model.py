"""
The deep learning model  -  a 1D Convolutional Neural Network (CNN)
===================================================================

This file describes the model and nothing else. It is imported by BOTH:

    2_train_model.py   to build and train it
    app.py             to load it and make predictions

Keeping it in one file matters. If the training script cleaned the signal one
way and the web app cleaned it another way, the app would quietly give wrong
answers and nothing would crash to tell us. Because both import preprocess()
from here, that mistake is impossible.

Why a 1D CNN?
    An ECG is a line that goes up and down over time. A CNN slides small
    "filters" along that line looking for shapes (like the spike of a
    heartbeat, or the flat bit that follows it). It learns the useful shapes
    by itself, so we never have to tell it what a heart attack looks like.
"""

import json
import os

# TensorFlow prints a lot of start-up noise. Quieten it before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt
from tensorflow.keras import layers, models

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.keras")
CONFIG_PATH = os.path.join(HERE, "model_config.json")

# An optional upgrade. If a set of models trained on Google Colab is dropped
# into system/ensemble/, the app serves the ensemble instead of the single
# local model. If the folder is absent or incomplete, nothing changes and
# model.keras is used - so the system always starts, which matters more during
# a demo than squeezing out the last fraction of a percent of accuracy.
ENSEMBLE_DIR = os.path.join(HERE, "ensemble")
ENSEMBLE_CONFIG_PATH = os.path.join(ENSEMBLE_DIR, "ensemble_config.json")

# Every recording is 10 seconds long, sampled 100 times a second.
SAMPLE_RATE_HZ = 100
INPUT_LENGTH = 1000

# ---------------------------------------------------------------------------
# Band-pass filter settings
# ---------------------------------------------------------------------------
# 0.5-40 Hz, following Davarmanesh et al. (2024), the Lead-I paper this study
# builds on. The low edge removes baseline wander - breathing and movement drag
# the trace up and down at well under 0.5 Hz. The high edge removes muscle
# noise and amplifier hash while keeping the QRS complex, whose useful energy is
# mostly below 40 Hz.
#
# Note on mains hum: at 100 Hz sampling, 50 Hz sits exactly at Nyquist and is
# not representable at all (a 50 Hz sine sampled at 100 Hz is identically zero),
# and PTB-XL's 100 Hz files were anti-alias filtered when they were downsampled
# from 500 Hz. So the high edge is doing very little on this dataset - it earns
# its place on live AD8232 input, where the ESP32 samples at 100 Hz with no
# anti-aliasing and out-of-band noise would otherwise fold down into the signal.
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 3

# Whether the filter is actually applied. This is OFF by default and is switched
# on from the loaded model's config - it is NOT a free-standing setting.
#
# Why it has to work that way. Every model trained so far was trained on
# z-scored-only signals; no filter existed in this pipeline when they were fit.
# Switching the filter on globally would hand those models an input distribution
# they have never seen. Nothing would crash - the model would keep returning
# confident-looking probabilities that are quietly wrong, which is the exact
# failure this module's docstring warns about. So the filter follows the model:
# a config that does not mention it (every existing one) gets no filter, and a
# model trained with it records that fact and gets it back at serving time.
_bandpass_enabled = False


def set_bandpass(enabled):
    """Turn the band-pass filter on or off for this process.

    Called automatically by load_trained_model() from the model's own config,
    and by the training script before it preprocesses anything. Callers should
    not normally need to touch it directly.
    """
    global _bandpass_enabled
    _bandpass_enabled = bool(enabled)


def bandpass_enabled():
    """Report whether the filter is currently applied."""
    return _bandpass_enabled


def bandpass(signal, sample_rate=SAMPLE_RATE_HZ,
             low=BANDPASS_LOW_HZ, high=BANDPASS_HIGH_HZ, order=BANDPASS_ORDER):
    """Apply a zero-phase Butterworth band-pass to one signal.

    filtfilt runs the filter forwards and then backwards. A one-way filter
    delays the signal by an amount that varies with frequency, which would shift
    the R peak relative to the ST segment and distort exactly the shapes the
    model is meant to read. Running it both ways cancels that delay entirely, at
    the cost of doubling the effective filter order.
    """
    nyquist = sample_rate / 2.0
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Cleaning the signal  (the SAME step for training and for live use)
# ---------------------------------------------------------------------------
def clean(signal):
    """Band-pass (if enabled), then z-score normalise.

    Z-scoring rescales the signal so mean = 0 and standard deviation = 1.
    Different ECG machines (and different people's skin) produce signals of
    very different sizes. Without this, the model could learn "big numbers =
    heart attack", which would be nonsense. After z-scoring, only the SHAPE
    of the heartbeat is left for the model to judge.

    The tiny 1e-8 stops us dividing by zero on a completely flat signal.

    ORDER MATTERS: filter first, normalise second. Baseline wander is a large
    slow swing, and it inflates the standard deviation of the whole recording.
    Z-scoring before removing it would divide the real heartbeat down by a
    number that is mostly drift, so how much the beat gets shrunk would depend
    on how much the patient happened to be breathing. Filtering first removes
    the drift, and the z-score that follows is then computed on the signal we
    actually care about.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if _bandpass_enabled:
        sig = bandpass(sig)
    return (sig - sig.mean()) / (sig.std() + 1e-8)


def preprocess(signal):
    """Clean ONE signal and shape it the way the model expects.

    A Conv1D layer wants three dimensions: (batch, time, channels).
    We have one recording of 1000 time points with 1 channel (Lead I),
    so the shape becomes (1, 1000, 1).
    """
    return clean(signal).reshape(1, -1, 1)


def preprocess_many(X):
    """The same cleaning, but for a whole pile of recordings at once.

    Used during training. `keepdims=True` makes the mean/std line up row by
    row, so each recording is scaled by its own numbers, not the group's.

    This must stay in step with clean() above - same filter, same order, same
    normalisation - or the model would be trained on one kind of input and
    served another.
    """
    X = np.asarray(X, dtype=np.float32)

    if _bandpass_enabled:
        # filtfilt takes an axis argument, so the whole batch is filtered in one
        # call rather than looped row by row.
        nyquist = SAMPLE_RATE_HZ / 2.0
        b, a = butter(BANDPASS_ORDER,
                      [BANDPASS_LOW_HZ / nyquist, BANDPASS_HIGH_HZ / nyquist],
                      btype="band")
        X = filtfilt(b, a, X, axis=1).astype(np.float32)

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mean) / std)[..., np.newaxis]      # (N, 1000) -> (N, 1000, 1)


# ---------------------------------------------------------------------------
# 2. The network itself
# ---------------------------------------------------------------------------
def build_model(input_length=INPUT_LENGTH):
    """Build the 1D CNN.

    It is three repeats of the same idea, then a small decision part.

    Each block does:
      Conv1D       slides filters along the signal to find patterns.
                   16 filters first (simple shapes), then 32, then 64
                   (patterns made of the earlier patterns).
      BatchNorm    keeps the numbers in a sensible range so training is
                   faster and steadier.
      ReLU         lets the network learn curved, non-straight-line rules.
      MaxPooling   halves the length, keeping only the strongest response.
                   1000 -> 500 -> 250 -> 125.

    Then:
      GlobalAveragePooling1D  squashes what is left into 64 summary numbers.
      Dense(64)               a small layer that weighs those numbers up.
      Dropout(0.5)            switches off half the neurons at random while
                              training, which stops the model memorising the
                              training recordings instead of learning.
      Dense(1, sigmoid)       ONE output between 0 and 1: the chance of MI.
    """
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

        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        # binary_crossentropy is the standard loss for a yes/no question.
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 3. Using a model that has already been trained
# ---------------------------------------------------------------------------
def _load_one(path):
    """Load a single .keras file, falling back to weights if that fails.

    A .keras file carries the version of Keras that wrote it, and Keras will
    refuse to open one written by a version it does not recognise. The models
    here may be trained on Colab and served on a laptop with a different
    version, so when the direct load fails we rebuild the architecture from the
    .arch.json saved beside it and pour the .weights.h5 numbers in. Weights are
    plain arrays with no version metadata, so that path works regardless.
    """
    try:
        return tf.keras.models.load_model(path)
    except Exception as direct_error:
        stem = path[: -len(".keras")]
        arch, weights = stem + ".arch.json", stem + ".weights.h5"

        if not (os.path.exists(arch) and os.path.exists(weights)):
            raise RuntimeError(
                f"Could not load {os.path.basename(path)}:\n    {direct_error}\n\n"
                f"This is usually a Keras version mismatch between the machine "
                f"that trained the model and this one.\n"
                f"Fix: in Colab, run the cell in Research/colab_compat_cell.py to "
                f"write {os.path.basename(arch)} and {os.path.basename(weights)} "
                f"beside it, then download those too."
            ) from direct_error

        with open(arch, encoding="utf-8") as f:
            rebuilt = tf.keras.models.model_from_json(f.read())
        rebuilt.load_weights(weights)
        print(f"  ({os.path.basename(path)} rebuilt from weights - "
              f"version mismatch handled)")
        return rebuilt


def load_trained_model(path=MODEL_PATH):
    """Load the model(s) the app will serve, WITH the settings they belong to.

    Returns (loaded, config):

      loaded  always a LIST, even for a single model. One model is just an
              ensemble of one, and returning the same type either way means
              predict_probability below needs no special case - which is how a
              whole class of "works with one model, silently wrong with five"
              bugs is avoided.

      config  the parsed JSON settings for whichever models were actually
              loaded: ensemble_config.json for the ensemble, model_config.json
              for the single model.

    The two are returned TOGETHER on purpose. They used to be fetched
    separately - app.py loaded the models here and then opened CONFIG_PATH
    itself - and because this function prefers the ensemble while CONFIG_PATH
    always names the single model's file, the app served five ensemble models
    at the thresholds tuned for a completely different network (0.38 / 0.17
    instead of 0.4433 / 0.2736). Nothing crashed; every verdict was simply
    decided at the wrong cut-off. Handing back the matching config makes that
    mismatch unrepresentable rather than merely fixed.
    """
    # Preferred: the ensemble, if a complete one is present.
    if os.path.exists(ENSEMBLE_CONFIG_PATH):
        with open(ENSEMBLE_CONFIG_PATH, encoding="utf-8") as f:
            ensemble_config = json.load(f)
        members = ensemble_config.get("members", [])

        paths = [os.path.join(ENSEMBLE_DIR, m) for m in members]
        missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]

        if members and not missing:
            print(f"Loading ensemble of {len(paths)} models ...")
            _apply_preprocessing_config(ensemble_config)
            return [_load_one(p) for p in paths], ensemble_config

        # An incomplete ensemble is a half-finished download, not a request to
        # serve fewer models. Averaging 3 of 5 would silently give numbers that
        # match nothing in the thesis, so say so and use the single model.
        print(f"  ensemble_config.json lists {len(members)} model(s) but "
              f"{len(missing)} are missing ({', '.join(missing[:3])}"
              f"{' ...' if len(missing) > 3 else ''}).")
        print("  Falling back to the single model.")
        # Falling through here also drops ensemble_config, which is the point:
        # the config must follow the models, not the folder that was present.

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the trained model at:\n    {path}\n\n"
            "Train it first by running:\n"
            "    python 1_extract_data.py\n"
            "    python 2_train_model.py"
        )

    with open(CONFIG_PATH, encoding="utf-8") as f:
        single_config = json.load(f)
    _apply_preprocessing_config(single_config)
    return [_load_one(path)], single_config


def _apply_preprocessing_config(config):
    """Switch preprocessing to match the model being loaded.

    Called from inside load_trained_model() rather than left to the caller,
    for the same reason the config is returned alongside the models: anything
    a caller has to remember to do is something a caller will eventually
    forget, and forgetting this one produces wrong answers silently.
    """
    enabled = bool(config.get("bandpass", {}).get("enabled", False))
    set_bandpass(enabled)

    if enabled:
        bp = config["bandpass"]
        print(f"  preprocessing: band-pass {bp.get('low_hz', BANDPASS_LOW_HZ)}"
              f"-{bp.get('high_hz', BANDPASS_HIGH_HZ)} Hz, then z-score")
    else:
        print("  preprocessing: z-score only "
              "(this model was trained without a band-pass filter)")


def describe(loaded):
    """A one-line description of what is actually being served.

    Used by /api/health so the reported model can never drift out of step with
    the model in memory.
    """
    n = len(loaded)
    if n == 1:
        return "1D CNN (single model)"
    return f"1D CNN ensemble ({n} models, mean of their probabilities)"


# True once this process has run one inference. The first call is far slower
# than the rest because TensorFlow traces the computation graph then, so the
# caller is told which call was the warm-up and can record it separately
# instead of letting a 2.4-second start-up cost pollute the average speed.
_warmed_up = False


def predict_probability(loaded, signal):
    """Return (probability, was_warmup) for one recording.

    probability  0.0 means "certainly Normal", 1.0 means "certainly MI".
                 This is NOT yet a yes/no answer - turning it into one needs a
                 threshold, and choosing that threshold is a separate decision
                 made in 2_train_model.py.

    was_warmup   True only for the first inference in this process. That call
                 measures TensorFlow starting up, not how fast the model is.

    With several models we take the MEAN of their probabilities. Each is
    trained from a different random start, so each makes slightly different
    mistakes; averaging cancels the disagreements and keeps what they agree on.
    """
    global _warmed_up

    batch = preprocess(signal)
    # verbose=0 stops Keras printing a progress bar for a single recording.
    probs = [float(m.predict(batch, verbose=0).ravel()[0]) for m in loaded]

    was_warmup = not _warmed_up
    _warmed_up = True

    return sum(probs) / len(probs), was_warmup
