"""
PASTE THIS AS THE LAST CELL of BOTH Colab notebooks
===================================================

Why this exists
---------------
A `.keras` file written by Colab's Keras will not always load on your laptop.
Colab and this machine are on different versions (laptop: TensorFlow 2.21 /
Keras 3.15.1 / Python 3.13), and Keras refuses to load a file written by a
version it does not recognise.

If that happens you would have to re-run the whole training, in Colab, at the
worst possible moment. This cell makes that impossible by saving a SECOND,
version-proof copy of every model:

    model_seed42.keras            <- normal save; fast path, may not load
    model_seed42.weights.h5       <- just the numbers; loads anywhere
    model_seed42.arch.json        <- just the shape; rebuild, then load weights

Weights files are plain arrays with no version metadata, so the fallback path
works even when the fast path does not.

Run this cell AFTER training has finished, in the same session (it only reads
files from Drive, so it is safe to re-run).
"""

import json
import os
import sys

import tensorflow as tf
import keras

# ---------------------------------------------------------------------------
# 1. Record exactly what built these files
# ---------------------------------------------------------------------------
# Phase 4 needs these numbers to diagnose a failed load in seconds rather than
# guessing. Written to Drive so it survives the session.
env = {
    "python": sys.version.split()[0],
    "tensorflow": tf.__version__,
    "keras": keras.__version__,
}
print("This Colab session:")
for k, v in env.items():
    print(f"  {k:12s} {v}")
print("\nYour laptop:  python 3.13.1 / tensorflow 2.21.0 / keras 3.15.1")
if env["keras"].split(".")[0] != "3":
    print("  !! Different Keras MAJOR version - the .keras files will NOT load "
          "locally. The weights fallback below is what you will use.")

with open(os.path.join(OUT_DIR, "environment.json"), "w") as f:
    json.dump(env, f, indent=2)


# ---------------------------------------------------------------------------
# 2. Save a version-proof copy of every model
# ---------------------------------------------------------------------------
def save_portable(keras_path):
    """Load one .keras file and write weights + architecture beside it."""
    stem = keras_path[: -len(".keras")]
    m = tf.keras.models.load_model(keras_path)

    m.save_weights(stem + ".weights.h5")
    with open(stem + ".arch.json", "w") as f:
        f.write(m.to_json())

    print(f"  {os.path.basename(stem):24s} weights + arch saved "
          f"({m.count_params():,} params)")


# Find every model this notebook produced, wherever it put them. Doing it by
# search rather than by variable name means the same cell works unchanged in
# both notebooks.
search_dirs = [OUT_DIR, os.path.join(OUT_DIR, "ensemble")]
found = []
for d in search_dirs:
    if os.path.isdir(d):
        found += [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.endswith(".keras")]

if not found:
    print("\nNo .keras files found - has training finished?")
else:
    print(f"\nMaking {len(found)} model(s) portable:")
    for path in found:
        save_portable(path)

print("\nDone. Download the WHOLE outputs/ folder to your laptop, "
      "including the .weights.h5 and .arch.json files.")
