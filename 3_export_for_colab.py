"""
STEP 3  -  Pack the dataset into ONE file to upload to Google Drive
==================================================================

Step 1 made three separate files (X.npy, y.npy, folds.npy) totalling ~58 MB.
Uploading three files and keeping their names straight in Colab is fiddly, so
this script squeezes all three into a single compressed file:

    data/ptbxl_leadI.npz

That one file is everything the Colab notebook needs. Drag it into Google Drive
and you are done.

Run:
    python 3_export_for_colab.py
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

OUT_PATH = os.path.join(DATA_DIR, "ptbxl_leadI.npz")

# ---------------------------------------------------------------------------
# 1. Load what step 1 produced
# ---------------------------------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X.npy"))       # (N, 1000) Lead I, 100 Hz
y = np.load(os.path.join(DATA_DIR, "y.npy"))       # 0 = Normal, 1 = MI
folds = np.load(os.path.join(DATA_DIR, "folds.npy"))  # 1..10, official PTB-XL

# ---------------------------------------------------------------------------
# 2. Sanity checks - better to fail here than halfway through training
# ---------------------------------------------------------------------------
assert X.ndim == 2, f"X should be 2-D (recordings, time points), got {X.shape}"
assert len(X) == len(y) == len(folds), "X, y and folds must have the same length"
assert set(np.unique(y)) <= {0, 1}, "labels must be only 0 and 1"
assert not np.isnan(X).any(), "X contains NaN values"
assert folds.min() >= 1 and folds.max() <= 10, "folds must be in 1..10"

# ---------------------------------------------------------------------------
# 3. Save the single bundle
# ---------------------------------------------------------------------------
# float32 is plenty of precision for an ECG and halves the file size vs float64.
np.savez_compressed(
    OUT_PATH,
    X=X.astype(np.float32),
    y=y.astype(np.int64),
    folds=folds.astype(np.int64),
)

size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)

print("=========== DONE ===========")
print(f"Recordings      : {len(y)}")
print(f"  Normal (0)    : {int((y == 0).sum())}")
print(f"  Heart attack  : {int((y == 1).sum())}")
print(f"Signal shape    : {X.shape}  (recordings, time points)")
print(f"Written to      : {OUT_PATH}")
print(f"File size       : {size_mb:.1f} MB")
print()
print("NEXT STEPS")
print("  1. In Google Drive, make a folder:  MyDrive/ecg_project/")
print("  2. Upload  data/ptbxl_leadI.npz  into that folder.")
print("  3. Open colab_train_eval.ipynb in Google Colab and run the cells.")
