"""
The database  -  what we store, and the small helpers to read/write it
=====================================================================

This file is the ONLY place that knows any SQL. Every other file in the
system calls the functions here instead of writing queries of its own, so if
a table ever changes there is exactly one file to fix.

We use SQLite, which comes free with Python (no server to install, no
password to remember). The whole database is a single file: ecg.db

Three tables:

  recordings   the 14,538 ECG recordings taken from the PTB-XL dataset.
               Filled once by 1_extract_data.py.

  predictions  one row every time the system makes a decision. Filled by the
               Flask app while it runs, so we keep a history instead of
               throwing each result away.

  patients     the people the system is monitoring. Each one owns an ESP32
               (identified by its device_id), so when a signal arrives we can
               say WHOSE heart it came from. Without this table the system can
               only answer "is this an MI?"; with it, it can answer "is this
               patient having an MI, and what did their last ten readings look
               like?" - which is the difference between a demo and a system.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np

# The database file lives next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "ecg.db")


# ---------------------------------------------------------------------------
# 1. The table definitions
# ---------------------------------------------------------------------------
# "IF NOT EXISTS" means running this twice is harmless.
CREATE_RECORDINGS = """
CREATE TABLE IF NOT EXISTS recordings (
    id         INTEGER PRIMARY KEY,   -- row number in the original .npy file
    signal     BLOB    NOT NULL,      -- 1000 float32 numbers stored as raw bytes
    n_samples  INTEGER NOT NULL,      -- how many numbers are in that blob (1000)
    label      INTEGER NOT NULL,      -- 0 = Normal, 1 = MI (heart attack)
    fold       INTEGER NOT NULL       -- 1..10, the official PTB-XL split number
)
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER,             -- which recording; NULL if it came from a device
    patient_id   INTEGER,             -- whose heart this was; NULL if nobody was named
    source       TEXT    NOT NULL,    -- 'simulation' or 'device'
    probability  REAL    NOT NULL,    -- the model's output, 0.0 to 1.0
    threshold    REAL    NOT NULL,    -- the cut-off we compared it against
    mode         TEXT    NOT NULL,    -- default | best_f1 | high_recall
    prediction   INTEGER NOT NULL,    -- 0 = said Normal, 1 = said MI
    truth        INTEGER,             -- the real answer; NULL if we don't know it
    correct      INTEGER,             -- 1 if prediction == truth, else 0; NULL if unknown
    inference_ms REAL    NOT NULL,    -- how long the model took, in milliseconds
    is_warmup    INTEGER NOT NULL DEFAULT 0,  -- 1 = first inference of a process
    created_at   TEXT    NOT NULL     -- when it happened
)
"""

# The people being monitored.
#
# NOTE ON THE DEMOGRAPHICS: a patient's name/age/sex here is entered by hand and
# is illustrative only. The ECG signal and its MI/Normal label are real; the
# person attached to it is not. The thesis must say so.
#
# This used to be forced on us - an earlier note here said ptbxl_database.csv was
# not on this machine and only the signal arrays had survived. That is no longer
# true (verified 2026-09-04): the full PTB-XL download is present, and its
# ptbxl_database.csv carries each recording's real age, sex, clinical report and
# infarction_stadium1. Real demographics COULD now be joined in by ecg_id.
# They deliberately are not, because a made-up name makes it obvious to anyone
# reading the dashboard that these are not real patients - which is the right
# default for a research prototype that is not a medical device.
#
# device_id is UNIQUE because one ESP32 sits on one chest. That single
# constraint is what lets an incoming POST identify its owner without the
# device needing to know any patient number.
CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    age        INTEGER,
    sex        TEXT,                  -- 'M', 'F' or NULL
    device_id  TEXT    UNIQUE,        -- the ESP32 bound to this patient
    notes      TEXT,
    created_at TEXT    NOT NULL
)
"""

# ---------------------------------------------------------------------------
# The application's own data, added 2026-09-23 with the MQTT web application
# ---------------------------------------------------------------------------
# A device belongs to the account that claimed it. That single row is what makes
# an ECG arriving over MQTT the property of one user: the device presents only
# its own identifier, the broker's ACL stops it publishing as any other device,
# and this table says whose chest it is on. `patients` is left exactly as it
# was, because Chapter 5 and 6 of the thesis report results from it.
CREATE_DEVICES = """
CREATE TABLE IF NOT EXISTS devices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT    NOT NULL UNIQUE,   -- as published by the firmware
    user_id     INTEGER NOT NULL,
    label       TEXT,
    token       TEXT,
    created_at  TEXT    NOT NULL,
    last_seen   TEXT,
    last_status TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
"""

# One finished ten-second window, with the waveform that produced the verdict.
#
# The waveform is stored, unlike in `predictions`, because a report the user
# cannot look at is not a report: a probability with no trace behind it can
# never be checked, re-scored or shown to a supervisor. It is kept as a BLOB of
# 1000 float32 values, 4 KB per recording, exactly the encoding `recordings`
# already uses. One row per window rather than one row per sample: a row per
# sample would be 1000 inserts for every ten seconds and would buy nothing,
# since the model only ever reads the whole window.
CREATE_ECG_REPORTS = """
CREATE TABLE IF NOT EXISTS ecg_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    device_id     TEXT,
    source        TEXT    NOT NULL,       -- 'mqtt' | 'serial' | 'simulation'
    started_at    TEXT    NOT NULL,
    duration_s    REAL    NOT NULL,
    lead          TEXT    NOT NULL DEFAULT 'I',
    sample_rate   INTEGER NOT NULL,
    n_samples     INTEGER NOT NULL,
    signal        BLOB    NOT NULL,       -- 1000 float32 raw ADC counts
    measured_rate REAL,
    bpm           REAL,
    beat_strength REAL,
    quality_json  TEXT,
    probability   REAL    NOT NULL,
    threshold     REAL    NOT NULL,
    mode          TEXT    NOT NULL,
    prediction    INTEGER NOT NULL,
    model_name    TEXT,
    model_version TEXT,
    inference_ms  REAL,
    created_at    TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
"""

# Nearly every query asks for "fold 10 only", so an index there makes the app
# feel instant instead of scanning all 14,538 rows each time.
CREATE_FOLD_INDEX = "CREATE INDEX IF NOT EXISTS idx_recordings_fold ON recordings(fold)"

# Every history page asks "this user's reports, newest first", and every report
# page asks for one row by id and user. Both go through this index.
CREATE_REPORT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_reports_user ON ecg_reports(user_id, created_at)"
)
CREATE_DEVICE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_devices_user ON devices(user_id)"
)

# The patient page asks "all predictions for patient X, newest first" on every
# load, so give that lookup an index too.
CREATE_PATIENT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id)"
)


def connect():
    """Open the database file and return a connection.

    row_factory = sqlite3.Row lets us write row["label"] instead of row[3],
    which is much easier to read.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    """Make the tables if they are not there yet, and bring an old one up to date."""
    conn.execute(CREATE_RECORDINGS)
    conn.execute(CREATE_PREDICTIONS)
    conn.execute(CREATE_PATIENTS)
    conn.execute(CREATE_USERS)
    conn.execute(CREATE_FOLD_INDEX)

    # `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
    # so a database built before patients were added would silently keep the old
    # predictions table and every patient query would fail with "no such column".
    # Add the column to those older files instead of asking anyone to delete
    # their database and lose the 14,538 recordings in it.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "patient_id" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN patient_id INTEGER")

    # Same story for is_warmup. Rows written before this column existed default
    # to 0, which is the safe reading: an unmarked row is treated as a normal
    # timed call rather than silently excluded from the average.
    if "is_warmup" not in columns:
        conn.execute(
            "ALTER TABLE predictions ADD COLUMN is_warmup INTEGER NOT NULL DEFAULT 0"
        )

    # A device proves its identity with a shared secret rather than by naming a
    # device_id alone, which anyone could guess. Added by migration so existing
    # databases keep their 14,538 recordings.
    pcols = {row["name"] for row in conn.execute("PRAGMA table_info(patients)")}
    if "device_token" not in pcols:
        conn.execute("ALTER TABLE patients ADD COLUMN device_token TEXT")

    # The web application asks each account for its own date of birth, height
    # and weight. They live on the account rather than in a separate profile
    # table because there is exactly one of each per user, and a one-to-one
    # table would add a join to every page for nothing. Added by migration so
    # the existing database keeps its accounts.
    ucols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    for column, decl in (("full_name", "TEXT"), ("dob", "TEXT"),
                         ("height_cm", "REAL"), ("weight_kg", "REAL"),
                         ("updated_at", "TEXT")):
        if column not in ucols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {column} {decl}")

    conn.execute(CREATE_DEVICES)
    conn.execute(CREATE_ECG_REPORTS)

    # Built after the ALTERs above, because the columns have to exist first.
    conn.execute(CREATE_PATIENT_INDEX)
    conn.execute(CREATE_REPORT_INDEX)
    conn.execute(CREATE_DEVICE_INDEX)
    conn.commit()


# ---------------------------------------------------------------------------
# 2. Turning an ECG signal into something SQLite can hold (and back again)
# ---------------------------------------------------------------------------
# An ECG recording is 1000 decimal numbers. SQLite has no "list of numbers"
# column, so we store the numbers as raw bytes in a BLOB column.
#
# 1000 numbers x 4 bytes each (float32) = 4000 bytes per recording. Storing
# them as text instead would take roughly 3x more space and would lose a
# little precision every time we converted.
def signal_to_blob(signal):
    """numpy array of numbers  ->  raw bytes, ready to store."""
    return np.asarray(signal, dtype=np.float32).tobytes()


def blob_to_signal(blob):
    """raw bytes from the database  ->  numpy array of numbers."""
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Reading recordings
# ---------------------------------------------------------------------------
def count_recordings(conn):
    """How many recordings are stored in total."""
    return conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]


def count_by_label(conn, fold=None):
    """Return {0: how many Normal, 1: how many MI}, optionally for one fold."""
    if fold is None:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n FROM recordings GROUP BY label"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n FROM recordings WHERE fold = ? GROUP BY label",
            (fold,),
        ).fetchall()
    return {row["label"]: row["n"] for row in rows}


def get_recording(conn, recording_id):
    """Fetch one recording by its id. Returns None if there is no such row."""
    return conn.execute(
        "SELECT * FROM recordings WHERE id = ?", (recording_id,)
    ).fetchone()


def random_recording(conn, fold, label=None):
    """Pick one recording at random from a fold.

    label=None means "any", 0 means Normal only, 1 means MI only.

    ORDER BY RANDOM() LIMIT 1 is SQLite's way of saying "shuffle, take one".
    It is fine here because the fold index keeps the candidate set small.
    """
    if label is None:
        sql = "SELECT * FROM recordings WHERE fold = ? ORDER BY RANDOM() LIMIT 1"
        params = (fold,)
    else:
        sql = ("SELECT * FROM recordings WHERE fold = ? AND label = ? "
               "ORDER BY RANDOM() LIMIT 1")
        params = (fold, label)
    return conn.execute(sql, params).fetchone()


def load_fold(conn, folds):
    """Load whole folds into memory for training.

    `folds` is a list like [1,2,3,4,5,6,7,8] or [10].
    Returns (X, y) where X has shape (number of recordings, 1000).
    """
    # SQLite needs one "?" per value, so build "?,?,?" of the right length.
    placeholders = ",".join("?" for _ in folds)
    rows = conn.execute(
        f"SELECT signal, label FROM recordings WHERE fold IN ({placeholders}) ORDER BY id",
        tuple(folds),
    ).fetchall()

    X = np.stack([blob_to_signal(row["signal"]) for row in rows])
    y = np.array([row["label"] for row in rows], dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# 4. Writing and reading the prediction history
# ---------------------------------------------------------------------------
def save_prediction(conn, source, probability, threshold, mode, prediction,
                    inference_ms, recording_id=None, truth=None, patient_id=None,
                    is_warmup=False):
    """Store one decision the system made. Returns the new row's id.

    `truth` is only known in simulation mode (we are replaying a recording
    whose real diagnosis is in the dataset). For a signal arriving from the
    ESP32 there is no truth, so it stays NULL and `correct` stays NULL too.

    `patient_id` is optional so that every existing caller keeps working
    unchanged; it is filled in when the request named a patient, or when a
    device_id was recognised as belonging to one.

    `is_warmup` marks the first inference made by a freshly started process.
    TensorFlow traces the computation graph on that call, so it costs seconds
    rather than milliseconds and describes the framework starting up, not the
    speed of the model. Recording it as a flag rather than dropping the row
    keeps the history complete while letting prediction_stats() report the two
    separately.
    """
    correct = None if truth is None else int(prediction == truth)

    cursor = conn.execute(
        """
        INSERT INTO predictions
            (recording_id, patient_id, source, probability, threshold, mode,
             prediction, truth, correct, inference_ms, is_warmup, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (recording_id, patient_id, source, float(probability), float(threshold),
         mode, int(prediction), truth, correct, float(inference_ms),
         int(bool(is_warmup)), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cursor.lastrowid


def recent_predictions(conn, limit=20):
    """The most recent decisions, newest first."""
    return conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def prediction_stats(conn):
    """A quick summary of everything the system has decided so far.

    `avg_inference_ms` is the model's part of the end-to-end latency, which the
    research asks us to measure. It counts WARM calls only.

    Why warm calls only. The first prediction made by a freshly started process
    takes roughly 2.4 seconds while TensorFlow traces its computation graph;
    every call after it takes about 0.3 seconds. Averaging the two together
    produced 541.5 ms across 18 rows - a number that describes neither the
    start-up cost nor the running speed, and that drifts purely with how often
    the server happened to be restarted. The two are reported separately
    instead, so `avg_inference_ms` answers "how fast is a prediction?" and
    `warmup_ms` answers "what does starting up cost?".
    """
    row = conn.execute(
        """
        SELECT COUNT(*)       AS total,
               SUM(correct)   AS n_correct,
               COUNT(correct) AS n_known
        FROM predictions
        """
    ).fetchone()

    warm = conn.execute(
        """
        SELECT COUNT(*)          AS n,
               AVG(inference_ms) AS avg_ms
        FROM predictions
        WHERE is_warmup = 0
        """
    ).fetchone()

    cold = conn.execute(
        """
        SELECT COUNT(*)          AS n,
               AVG(inference_ms) AS avg_ms
        FROM predictions
        WHERE is_warmup = 1
        """
    ).fetchone()

    total = row["total"]
    n_known = row["n_known"] or 0
    n_correct = row["n_correct"] or 0

    return {
        "total": total,
        # Warm calls only - see the docstring.
        "avg_inference_ms": round(warm["avg_ms"], 1) if warm["avg_ms"] else 0.0,
        "n_warm_calls": warm["n"],
        # Reported beside it rather than folded into it.
        "warmup_ms": round(cold["avg_ms"], 1) if cold["avg_ms"] else None,
        "n_warmup_calls": cold["n"],
        # Accuracy only counts rows where we actually know the right answer.
        "accuracy_so_far": round(n_correct / n_known, 4) if n_known else None,
        "n_with_known_truth": n_known,
    }


# ---------------------------------------------------------------------------
# 5. Patients
# ---------------------------------------------------------------------------
def create_patient(conn, name, age=None, sex=None, device_id=None, notes=None):
    """Enrol one patient. Returns the new patient's id.

    Raises sqlite3.IntegrityError if the device_id is already bound to somebody
    else - which is the behaviour we want, because two patients sharing one
    ESP32 would make every reading ambiguous.
    """
    cursor = conn.execute(
        """
        INSERT INTO patients (name, age, sex, device_id, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, age, sex, device_id, notes,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cursor.lastrowid


def list_patients(conn):
    """Every patient, with a count of how many readings each one has.

    LEFT JOIN (not a plain JOIN) so a patient who has just been enrolled and
    has no readings yet still appears, with a count of 0.
    """
    return conn.execute(
        """
        SELECT p.*,
               COUNT(pr.id)      AS n_predictions,
               SUM(pr.prediction) AS n_mi_flags,
               MAX(pr.created_at) AS last_seen
        FROM patients p
        LEFT JOIN predictions pr ON pr.patient_id = p.id
        GROUP BY p.id
        ORDER BY p.id
        """
    ).fetchall()


def get_patient(conn, patient_id):
    """Fetch one patient by id. Returns None if there is no such row."""
    return conn.execute(
        "SELECT * FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()


def get_patient_by_device(conn, device_id):
    """Find the patient a device belongs to. Returns None if unrecognised.

    This is the lookup that turns an anonymous POST from an ESP32 into a
    reading attached to a person.
    """
    return conn.execute(
        "SELECT * FROM patients WHERE device_id = ?", (device_id,)
    ).fetchone()


def patient_predictions(conn, patient_id, limit=50):
    """One patient's readings, newest first."""
    return conn.execute(
        "SELECT * FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
        (patient_id, limit),
    ).fetchall()


def patient_stats(conn, patient_id):
    """A summary of one patient's readings.

    Deliberately the same shape as prediction_stats() above, plus the two
    numbers that only make sense per-person: how many readings were flagged as
    MI, and when we last heard from their device.
    """
    row = conn.execute(
        """
        SELECT COUNT(*)           AS total,
               SUM(prediction)    AS n_flagged,
               MAX(created_at)    AS last_seen,
               AVG(probability)   AS avg_probability
        FROM predictions
        WHERE patient_id = ?
        """,
        (patient_id,),
    ).fetchone()

    # Warm calls only, for the same reason as prediction_stats() above.
    warm = conn.execute(
        "SELECT AVG(inference_ms) AS avg_ms FROM predictions "
        "WHERE patient_id = ? AND is_warmup = 0",
        (patient_id,),
    ).fetchone()

    total = row["total"]
    return {
        "total": total,
        "avg_inference_ms": round(warm["avg_ms"], 1) if warm["avg_ms"] else 0.0,
        "n_flagged_mi": row["n_flagged"] or 0,
        "avg_probability": round(row["avg_probability"], 4) if row["avg_probability"] else None,
        "last_seen": row["last_seen"],
    }


# ---------------------------------------------------------------------------
# 6. Users  (added for the authenticated dashboard)
# ---------------------------------------------------------------------------
# Passwords are never stored. Werkzeug's generate_password_hash applies a salted
# PBKDF2 hash, so the database holds a value from which the password cannot be
# recovered. Storing the password itself would mean anyone with the .db file had
# every user's credentials, and people reuse passwords across systems.
CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'clinician',  -- clinician | admin
    created_at    TEXT    NOT NULL,
    last_login    TEXT
)
"""


def create_user(conn, username, password_hash, role="clinician"):
    """Add a user. Raises sqlite3.IntegrityError if the username is taken."""
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) "
        "VALUES (?, ?, ?, ?)",
        (username, password_hash, role,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cur.lastrowid


def get_user_by_name(conn, username):
    """Fetch one user by username, or None."""
    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def get_user(conn, user_id):
    """Fetch one user by id, or None."""
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def touch_last_login(conn, user_id):
    """Record that a user has just signed in."""
    conn.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id),
    )
    conn.commit()


def count_users(conn):
    """How many accounts exist. Used to decide whether to seed a first admin."""
    return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# ---------------------------------------------------------------------------
# 7. Device credentials
# ---------------------------------------------------------------------------
def get_patient_by_device_token(conn, device_id, token):
    """Resolve a device to its patient only if the token matches.

    Returns None both when the device is unknown and when the token is wrong,
    deliberately: telling an unauthenticated caller which of the two happened
    would let it enumerate valid device identifiers.
    """
    return conn.execute(
        "SELECT * FROM patients WHERE device_id = ? AND device_token = ?",
        (device_id, token),
    ).fetchone()


def set_device_token(conn, patient_id, token):
    """Attach or rotate the shared secret a device uses to identify itself."""
    conn.execute("UPDATE patients SET device_token = ? WHERE id = ?",
                 (token, patient_id))
    conn.commit()


# ---------------------------------------------------------------------------
# 7. The web application's own data: profiles, devices, ECG reports
# ---------------------------------------------------------------------------
# Every function here takes a user_id and puts it in the WHERE clause. That is
# the whole of the data-isolation rule, kept in one file rather than trusted to
# each route to remember: a query that cannot name a user cannot return another
# user's ECG.

def update_user_profile(conn, user_id, full_name=None, dob=None,
                        height_cm=None, weight_kg=None):
    """Store the profile fields the dashboard shows.

    Values are validated by the caller (web_account.validate_profile); this
    only writes them.
    """
    conn.execute(
        "UPDATE users SET full_name = ?, dob = ?, height_cm = ?, weight_kg = ?, "
        "updated_at = ? WHERE id = ?",
        (full_name, dob, height_cm, weight_kg,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id),
    )
    conn.commit()
    return get_user(conn, user_id)


def claim_device(conn, user_id, device_id, label=None, token=None):
    """Bind a device to an account.

    UNIQUE on device_id means a second account claiming the same hardware gets
    an IntegrityError rather than quietly taking someone's readings; the route
    turns that into a message naming the conflict.
    """
    cur = conn.execute(
        "INSERT INTO devices (device_id, user_id, label, token, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (device_id, user_id, label, token,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cur.lastrowid


def list_devices(conn, user_id):
    """This account's devices, with how many reports each has produced."""
    return conn.execute(
        "SELECT d.*, COUNT(r.id) AS reports, MAX(r.created_at) AS last_report "
        "FROM devices d LEFT JOIN ecg_reports r "
        "  ON r.device_id = d.device_id AND r.user_id = d.user_id "
        "WHERE d.user_id = ? GROUP BY d.id ORDER BY d.created_at",
        (user_id,),
    ).fetchall()


def get_device(conn, device_id):
    """One device by its hardware id, whoever owns it (used by the ingester)."""
    return conn.execute(
        "SELECT * FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()


def get_user_device(conn, user_id, device_id):
    """One device, but only if this account owns it. None otherwise."""
    return conn.execute(
        "SELECT * FROM devices WHERE user_id = ? AND device_id = ?",
        (user_id, device_id),
    ).fetchone()


def release_device(conn, user_id, device_id):
    """Unbind a device from this account. Returns True if a row was removed."""
    cur = conn.execute("DELETE FROM devices WHERE user_id = ? AND device_id = ?",
                       (user_id, device_id))
    conn.commit()
    return cur.rowcount > 0


def touch_device(conn, device_id, status=None):
    """Record that this device was heard from just now."""
    conn.execute(
        "UPDATE devices SET last_seen = ?, last_status = COALESCE(?, last_status) "
        "WHERE device_id = ?",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, device_id),
    )
    conn.commit()


def save_ecg_report(conn, user_id, signal, probability, threshold, mode,
                    prediction, source="mqtt", device_id=None, started_at=None,
                    sample_rate=100, measured_rate=None, bpm=None,
                    beat_strength=None, quality_json=None, model_name=None,
                    model_version=None, inference_ms=None, lead="I"):
    """Store one scored ten-second window, waveform included.

    The signal is written as float32 bytes, the same encoding `recordings`
    uses, so a report can be redrawn later exactly as it was scored. A
    probability with no trace behind it can never be checked afterwards, which
    is why this table stores what `predictions` does not.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blob = signal_to_blob(signal)
    n_samples = len(blob) // 4
    cur = conn.execute(
        "INSERT INTO ecg_reports (user_id, device_id, source, started_at, "
        " duration_s, lead, sample_rate, n_samples, signal, measured_rate, bpm, "
        " beat_strength, quality_json, probability, threshold, mode, prediction, "
        " model_name, model_version, inference_ms, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, device_id, source, started_at or now,
         round(n_samples / float(sample_rate), 2), lead, sample_rate, n_samples,
         blob, measured_rate, bpm, beat_strength, quality_json,
         float(probability), float(threshold), mode, int(prediction),
         model_name, model_version, inference_ms, now),
    )
    conn.commit()
    return cur.lastrowid


def list_ecg_reports(conn, user_id, limit=20, offset=0, result=None,
                     date_from=None, date_to=None):
    """This user's reports, newest first, with optional filters.

    `result` is 'mi' or 'normal'; the dates are ISO days (YYYY-MM-DD). The
    waveform column is deliberately not selected: a history page listing fifty
    reports does not need fifty waveforms, and loading them would be 200 KB of
    blobs nobody looks at.
    """
    sql = ("SELECT id, device_id, source, started_at, duration_s, lead, "
           "sample_rate, n_samples, measured_rate, bpm, probability, threshold, "
           "mode, prediction, model_name, model_version, inference_ms, created_at "
           "FROM ecg_reports WHERE user_id = ?")
    params = [user_id]
    if result == "mi":
        sql += " AND prediction = 1"
    elif result == "normal":
        sql += " AND prediction = 0"
    if date_from:
        sql += " AND date(created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        sql += " AND date(created_at) <= date(?)"
        params.append(date_to)
    sql += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ? OFFSET ?"
    params += [int(limit), int(offset)]
    return conn.execute(sql, params).fetchall()


def count_ecg_reports(conn, user_id, result=None, date_from=None, date_to=None):
    """How many reports match the same filters, so history can paginate."""
    sql = "SELECT COUNT(*) AS n FROM ecg_reports WHERE user_id = ?"
    params = [user_id]
    if result == "mi":
        sql += " AND prediction = 1"
    elif result == "normal":
        sql += " AND prediction = 0"
    if date_from:
        sql += " AND date(created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        sql += " AND date(created_at) <= date(?)"
        params.append(date_to)
    return conn.execute(sql, params).fetchone()["n"]


def get_ecg_report(conn, user_id, report_id):
    """One report with its waveform, or None if it is not this user's.

    The user_id in the WHERE clause is what stops a guessed id in the URL from
    opening somebody else's recording; the route turns None into a 404.
    """
    return conn.execute(
        "SELECT * FROM ecg_reports WHERE id = ? AND user_id = ?",
        (report_id, user_id),
    ).fetchone()


def ecg_report_stats(conn, user_id):
    """The four numbers the dashboard shows above the history table."""
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "       SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS flagged, "
        "       SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS normal, "
        "       MAX(created_at) AS last_at "
        "FROM ecg_reports WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return {"total": row["total"] or 0,
            "flagged": row["flagged"] or 0,
            "normal": row["normal"] or 0,
            "last_at": row["last_at"]}
