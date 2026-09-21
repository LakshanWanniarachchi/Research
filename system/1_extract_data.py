"""
STEP 1  -  Extract the ECG data and put it into the SQLite database
==================================================================

Run this FIRST, once.

Where the data comes from:
    The PTB-XL research dataset was already turned into three fast-loading
    numpy files by the earlier project (Research/1_make_dataset.py):

        X.npy      the ECG signals   - 14,538 rows x 1000 numbers
        y.npy      the answers       - 0 = Normal, 1 = MI (heart attack)
        folds.npy  the split numbers - 1 to 10

What this script does:
    Reads those three files and writes every recording as one row in the
    `recordings` table of ecg.db.

Why bother putting it in a database at all?
    Because the rest of the system then has ONE place to ask for data. The
    web app can say "give me a random heart-attack recording from fold 10"
    in a single line, instead of loading a 58 MB file into memory and
    filtering it by hand. It also means the recordings and the predictions
    the system makes live side by side in the same file.

Running this twice is safe: it empties the recordings table and refills it.
"""

import os
import sys

import numpy as np

import models     # our own database file (models.py)

# ---------------------------------------------------------------------------
# 0. Find the .npy files
# ---------------------------------------------------------------------------
# This script lives in .../system/ and the data sits in .../Research/data/ ,
# so we go up one folder and back down. Building the path from this script's
# own location means it works no matter which folder you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "Research", "data")

REQUIRED = ["X.npy", "y.npy", "folds.npy"]

missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing:
    print("ERROR: could not find these files:")
    for f in missing:
        print("   ", os.path.join(DATA_DIR, f))
    print("\nBuild them first by running, inside the Research folder:")
    print("    python 1_make_dataset.py")
    sys.exit(1)

print("Reading the dataset from:", DATA_DIR)

# ---------------------------------------------------------------------------
# 1. Load the three files
# ---------------------------------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X.npy"))          # shape (14538, 1000)
y = np.load(os.path.join(DATA_DIR, "y.npy"))          # 0 or 1
folds = np.load(os.path.join(DATA_DIR, "folds.npy"))  # 1..10

print(f"  signals : {X.shape[0]} recordings x {X.shape[1]} numbers each")

# A quick safety check. If these three ever disagree, the labels would be
# attached to the wrong signals and everything after this point is rubbish.
if not (len(X) == len(y) == len(folds)):
    print(f"ERROR: the files disagree on how many recordings there are: "
          f"X={len(X)}, y={len(y)}, folds={len(folds)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Create the database and empty the recordings table
# ---------------------------------------------------------------------------
conn = models.connect()
models.create_tables(conn)

# DELETE (not DROP) so the table definition stays exactly as models.py says.
# We leave the `predictions` table alone - re-extracting the data should not
# wipe the history of what the system has already decided.
conn.execute("DELETE FROM recordings")
conn.commit()

# ---------------------------------------------------------------------------
# 3. Write every recording into the database
# ---------------------------------------------------------------------------
# Build the whole list first, then hand it to executemany(). Inserting rows
# one at a time with a commit each would take minutes; doing it in one
# transaction takes a couple of seconds.
print("Converting and inserting rows ...")

rows = []
for i in range(len(X)):
    rows.append((
        int(i),                             # id: the row number in X.npy
        models.signal_to_blob(X[i]),        # the 1000 numbers as raw bytes
        int(X.shape[1]),                    # n_samples = 1000
        int(y[i]),                          # label: 0 Normal, 1 MI
        int(folds[i]),                      # fold: 1..10
    ))

conn.executemany(
    "INSERT INTO recordings (id, signal, n_samples, label, fold) VALUES (?, ?, ?, ?, ?)",
    rows,
)
conn.commit()

# ---------------------------------------------------------------------------
# 4. Check what we stored, by reading it BACK out of the database
# ---------------------------------------------------------------------------
# Counting the numpy arrays again would prove nothing. These counts come from
# SQL, so they only look right if the data really did land in the database.
total = models.count_recordings(conn)
by_label = models.count_by_label(conn)

print("\n=========== DONE ===========")
print(f"Database file : {models.DB_PATH}")
print(f"Size on disk  : {os.path.getsize(models.DB_PATH) / 1e6:.1f} MB")
print(f"Recordings    : {total}")
print(f"  Normal (0)      : {by_label.get(0, 0)}")
print(f"  Heart attack (1): {by_label.get(1, 0)}")

print("\nRecordings in each fold:")
for row in conn.execute(
    "SELECT fold, COUNT(*) AS n, SUM(label) AS mi FROM recordings GROUP BY fold ORDER BY fold"
):
    note = ""
    if row["fold"] == 9:
        note = "  <- validation (used to choose the threshold)"
    elif row["fold"] == 10:
        note = "  <- test (never used for training)"
    print(f"  fold {row['fold']:2d}: {row['n']:5d} recordings "
          f"({row['mi']:4d} MI){note}")

# One last proof: pull a recording back out and check the numbers survived
# the trip into the database and back unchanged.
check_id = 0
row = models.get_recording(conn, check_id)
restored = models.blob_to_signal(row["signal"])
if np.array_equal(restored, X[check_id]):
    print(f"\nRead-back check: recording {check_id} matches the original exactly.")
else:
    print(f"\nWARNING: recording {check_id} changed on the way into the database!")

conn.close()
print("\nNext step:  python 2_train_model.py")
