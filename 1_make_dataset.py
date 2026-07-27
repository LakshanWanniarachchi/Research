"""
STEP 1 of 2  -  Build the dataset (run this first)
==================================================

What this script does, in plain words:
  1. Opens the PTB-XL description file (one row per ECG recording).
  2. Works out a simple label for each recording:
        0 = Normal        (a healthy "NORM" ECG)
        1 = Heart attack  (Myocardial Infarction, "MI")
     Recordings that are neither clearly Normal nor MI are skipped, so the
     two groups stay clean and easy to tell apart.
  3. Reads the actual ECG signal for each kept recording and keeps only
     Lead I (one lead, like a cheap single-sensor device would give).
  4. Saves everything into small .npy files so STEP 2 (training) can load
     them instantly instead of reading thousands of files again.

You only need to run this ONCE. After that, run 2_train_cnn.py as many
times as you like.
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb

# ---------------------------------------------------------------------------
# 0. Where are the files?  (paths are worked out automatically)
# ---------------------------------------------------------------------------
# This script lives in .../model_training/ .  The dataset sits one folder up,
# inside a folder with the same long name.  We build the paths from the
# script's own location so it works no matter where you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATASET_NAME = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Unzipping the download sometimes creates an extra folder level, so the real
# files end up at  <name>/<name>/ptbxl_database.csv  instead of <name>/... .
# Rather than guess, look for the file that must be there and use whichever
# layout we actually find.
def find_data_dir():
    """Return the folder that really contains ptbxl_database.csv."""
    candidates = [
        os.path.join(PROJECT_ROOT, DATASET_NAME),                  # normal
        os.path.join(PROJECT_ROOT, DATASET_NAME, DATASET_NAME),    # double-nested
        os.path.join(HERE, DATASET_NAME),                          # next to this script
        PROJECT_ROOT,
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "ptbxl_database.csv")):
            return path

    # Nothing matched - give a message that says exactly what to do.
    looked = "\n".join("    " + c for c in candidates)
    raise FileNotFoundError(
        "Could not find 'ptbxl_database.csv'.\n"
        f"Looked in:\n{looked}\n\n"
        "Make sure the PTB-XL dataset is unzipped and that the folder\n"
        "containing ptbxl_database.csv sits next to this project folder."
    )


DATA_DIR = find_data_dir()

OUT_DIR = os.path.join(HERE, "data")          # where the .npy files go
os.makedirs(OUT_DIR, exist_ok=True)

print("Reading dataset from:", DATA_DIR)

# ---------------------------------------------------------------------------
# 1. Load the description table (ptbxl_database.csv)
# ---------------------------------------------------------------------------
db = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"), index_col="ecg_id")

# The "scp_codes" column is stored as text that looks like a Python dict,
# e.g.  "{'NORM': 100.0, 'SR': 0.0}".  Turn that text back into a real dict.
db.scp_codes = db.scp_codes.apply(ast.literal_eval)

# ---------------------------------------------------------------------------
# 2. Load the code dictionary (scp_statements.csv)
# ---------------------------------------------------------------------------
# This file explains what each code means.  We only need "diagnostic" codes,
# and for each one we want its big group ("diagnostic_class"): NORM, MI, etc.
scp = pd.read_csv(os.path.join(DATA_DIR, "scp_statements.csv"), index_col=0)
scp = scp[scp.diagnostic == 1]                # keep only diagnostic codes
code_to_class = scp.diagnostic_class.to_dict()  # e.g. {'NORM':'NORM','IMI':'MI',...}


def label_for_record(scp_codes_dict):
    """Return 0 (Normal), 1 (MI), or None (skip) for one recording."""
    # Collect the big groups this recording belongs to.
    classes = set()
    for code in scp_codes_dict.keys():
        if code in code_to_class:
            classes.add(code_to_class[code])

    # Rule 1: pure Normal -> the ONLY group is NORM.
    if classes == {"NORM"}:
        return 0
    # Rule 2: any Myocardial Infarction present -> heart attack.
    if "MI" in classes:
        return 1
    # Anything else (other diseases, or mixed/unclear) -> skip.
    return None


# ---------------------------------------------------------------------------
# 3. Go through every recording, keep the clean Normal / MI ones, load Lead I
# ---------------------------------------------------------------------------
X_list, y_list, fold_list = [], [], []
kept, skipped = 0, 0

for ecg_id, row in db.iterrows():
    label = label_for_record(row.scp_codes)
    if label is None:
        skipped += 1
        continue

    # filename_lr is the 100 Hz version, e.g. "records100/00000/00001_lr".
    record_path = os.path.join(DATA_DIR, row.filename_lr)

    # wfdb reads the signal file. It returns (signal, info).
    # signal shape is (1000, 12): 1000 time points, 12 leads.
    signal, _ = wfdb.rdsamp(record_path)

    lead_I = signal[:, 0].astype(np.float32)  # column 0 = Lead I

    X_list.append(lead_I)
    y_list.append(label)
    fold_list.append(int(row.strat_fold))     # 1..10, used to split later

    kept += 1
    if kept % 1000 == 0:
        print(f"  ...loaded {kept} recordings so far")

# ---------------------------------------------------------------------------
# 4. Stack into arrays and save
# ---------------------------------------------------------------------------
X = np.stack(X_list)                # shape (N, 1000)
y = np.array(y_list, dtype=np.int64)
folds = np.array(fold_list, dtype=np.int64)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)
np.save(os.path.join(OUT_DIR, "folds.npy"), folds)

# ---------------------------------------------------------------------------
# 5. Print a quick summary so you can check it looks right
# ---------------------------------------------------------------------------
n_normal = int((y == 0).sum())
n_mi = int((y == 1).sum())
print("\n=========== DONE ===========")
print(f"Kept   : {kept} recordings")
print(f"Skipped: {skipped} recordings (not clearly Normal or MI)")
print(f"  Normal (0)      : {n_normal}")
print(f"  Heart attack (1): {n_mi}")
print(f"Signal shape saved : {X.shape}  (recordings, time points)")
print(f"Files written to   : {OUT_DIR}")
print("\nNext step:  python 2_train_cnn.py")
