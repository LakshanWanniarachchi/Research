# Heart Attack Detection System

A single-lead ECG goes in, a screening probability comes out, and every result
is stored with the waveform that produced it.

```
AD8232  ->  ESP32  --Wi-Fi-->  Mosquitto  -->  Flask backend  ->  1D CNN
                                                    |
                                                    v
                                          SQLite (recordings,
                                          devices, ECG reports)
                                                    |
                                                    v
                                        authenticated web dashboard
```

The device publishes one second of samples at a time over MQTT. The backend
reassembles the stream, forms ten-second windows, scores each one with the
trained model, and stores it against the account that owns the device. USB is
used for power, flashing and the debug console only.

Two other paths remain and are still tested: a **USB-serial** monitor
(`ecg_serial_monitor.py`, useful when no network is available) and a
**simulation** page that replays held-out PTB-XL recordings. Both go through
the same `decide()` call as MQTT, so no path has a second opinion about what an
ECG means.

---

## The files

| File | What it is |
|---|---|
| `config.py` | **Settings.** Reads `.env` into the environment; no secret has a default. |
| `models.py` | **The database.** The tables, and the only place any SQL lives. |
| `auth.py` | **Sign-in.** Accounts, password hashing, and the decorators that protect a page. |
| `web_account.py` | Registration, profile (date of birth, height, weight), device claiming. |
| `web_ecg.py` | Live monitor, ECG history, one ECG report. Every query names the signed-in user. |
| `app.py` | The Flask application: simulation page, device endpoint, dashboard, and the scorers. |
| `model.py` | **The deep learning model.** The 1D CNN, the preprocessing, the ensemble. |
| `mqtt_bridge.py` | **The MQTT client.** Subscribes, routes chunks to the ingester, publishes verdicts. |
| `mqtt_stream.py` | **The ingester.** Reassembles chunks into windows; counts losses instead of hiding them. |
| `mqtt_stream_simulator.py` | Publishes chunks exactly as the firmware does, for testing without hardware. |
| `mqtt_device_simulator.py` | The older whole-window MQTT simulator. |
| `esp32_simulator.py` | Stands in for the device over plain HTTP. |
| `ecg_serial_monitor.py` | The USB-serial monitor: live plot, CSV, windowing, offline scoring. |
| `live_serial.py` | The web application's USB-serial session. |
| `firmware/ecg_esp32_mqtt/` | **The flashed sketch.** Streams one-second chunks over Wi-Fi and MQTT. |
| `firmware/ecg_esp32_serial/` | USB-serial firmware, used for bench debugging. |
| `firmware/ecg_esp32.ino` | The original HTTP-POST sketch. Compiles; superseded by the MQTT sketch. |
| `mosquitto/mosquitto.windows.conf` | Broker configuration for a native Windows Mosquitto. |
| `mosquitto/mosquitto.conf` | The same for the Docker image (container paths). |
| `mosquitto/acl` | Which client may publish or read which topic. Not in version control. |
| `static/`, `templates/` | One stylesheet, one ECG drawing module, and the pages. |
| `tests/` | The pytest suite: 131 tests covering auth, the bridge, the ingester, the web app and the model path. |
| `1_extract_data.py` | **Step 1.** Reads the PTB-XL arrays and fills the database. |
| `2_train_model.py` | **Step 2.** Trains the CNN and picks the decision thresholds. |

Files that appear after you run the steps:

| File | Made by |
|---|---|
| `ecg.db` | step 1 — the database (~60 MB) |
| `model.keras` | step 2 — the trained model |
| `model_config.json` | step 2 — the thresholds and settings |
| `test_results.txt` | step 2 — the scores |

These four are **build products**. Delete them any time and re-run steps 1 and 2.

---

## How to run it

### 0. Install the libraries (once)
```
pip install -r requirements.txt
```

### 1. Put the data into the database (once)
```
python 1_extract_data.py
```
Reads `../Research/data/X.npy`, `y.npy` and `folds.npy` and writes 14,538 recordings
into `ecg.db`. Takes a few seconds. It prints the counts so you can check them:
**9,069 Normal** and **5,469 MI**.

### 2. Train the model (once, 10–20 minutes)
```
python 2_train_model.py
```
Reads the recordings back out of the database, trains the CNN, chooses the
threshold on the validation fold, tests once on the test fold, and saves
`model.keras`.

### 3. Configure and start the web app
```
copy .env.example .env      # then fill it in; see "Running over MQTT"
python app.py
```
Then open **http://127.0.0.1:5000**. Press `Ctrl + C` to stop.

| Page | What it is for |
|---|---|
| `/register`, `/login` | create an account, sign in |
| `/dashboard` | your recordings, your devices, and the state of the system |
| `/profile` | date of birth, height, weight |
| `/devices` | claim or release an ESP32 by its device id |
| `/ecg/live` | the live waveform arriving over MQTT, and the latest verdict |
| `/ecg/history` | every recording of yours, filtered by result or date |
| `/ecg/<id>` | one recording: its waveform, probability, threshold and model |
| `/` | the simulation page: replay a held-out PTB-XL recording |
| `/live` | the USB-serial monitor, for bench work without a network |

A recording belongs to the account that claimed the device that produced it.
Another account asking for it gets a 404, not a redirect and not an empty page:
it has no business learning that the recording exists.

---

## The database

Everything lives in one file, `ecg.db`. SQLite comes with Python, so there is no
server to install and no password to remember.

**`recordings`** — the dataset, filled once by step 1.

| Column | Meaning |
|---|---|
| `id` | row number in the original data |
| `signal` | the 1000 ECG numbers, stored as raw bytes (a BLOB) |
| `n_samples` | how many numbers that is (1000) |
| `label` | 0 = Normal, 1 = MI (heart attack) |
| `fold` | 1–10, the official PTB-XL split number |

**`predictions`** — one row every time the system decides something.

| Column | Meaning |
|---|---|
| `recording_id` | which recording; empty if the signal came from a device |
| `patient_id` | whose heart this was; empty if no patient was named |
| `source` | `simulation` or `device` |
| `probability` | what the model output, 0.0 to 1.0 |
| `threshold` | the cut-off it was compared against |
| `mode` | `default`, `best_f1` or `high_recall` |
| `prediction` | 0 = said Normal, 1 = said MI |
| `truth` | the real answer — only known in simulation |
| `correct` | 1 if right, 0 if wrong |
| `inference_ms` | how long the model took, in milliseconds |
| `created_at` | when it happened |

### `patients`

| Column | Meaning |
|---|---|
| `id` | patient number |
| `name`, `age`, `sex` | who they are |
| `device_id` | the ESP32 bound to this patient — **UNIQUE** |
| `notes` | anything else |
| `created_at` | when they were enrolled |

`device_id` is unique because one ESP32 sits on one chest. That single
constraint is what lets an incoming POST identify its owner without the device
knowing any patient number — and it means re-assigning a device to a different
patient is a database edit, not a reflash.

> **The demographics are illustrative.** The ECG and its Normal/MI label are
> real, but the person it is filed under was typed in by hand. This demonstrates
> *attribution*, not a clinical record, and the thesis says so.
>
> An earlier version of this note said the real demographics were unavailable
> because `ptbxl_database.csv` was not part of the extracted dataset. That is no
> longer true — the full PTB-XL download is present and does carry each
> recording's age, sex, clinical report and infarction stage. Real values could
> be joined in by `ecg_id`; they are left invented on purpose, so that nobody
> reading the dashboard mistakes these for real patient records.

### `users`

| Column | Meaning |
|---|---|
| `id`, `username`, `password_hash`, `role` | the account; only a salted hash is stored |
| `full_name`, `dob`, `height_cm`, `weight_kg` | the profile, all optional |
| `created_at`, `last_login`, `updated_at` | when |

Age is derived from `dob` when a page needs it rather than stored, because a
stored age is wrong within a year.

### `devices`

| Column | Meaning |
|---|---|
| `device_id` | as published by the firmware — **UNIQUE** |
| `user_id` | the account that claimed it |
| `label`, `token` | a human name, and a per-device secret |
| `last_seen`, `last_status` | when the backend last heard from it |

This one row is what makes an ECG arriving over MQTT somebody's recording. The
device never learns who owns it: it publishes under its own id, the broker's
ACL stops it publishing as any other device, and this table does the rest.

### `ecg_reports`

One finished ten-second window, **with the waveform that produced the verdict**.

| Column | Meaning |
|---|---|
| `user_id`, `device_id`, `source` | whose, from what, over which transport |
| `signal` | 1000 float32 raw ADC counts, 4 KB, as a BLOB |
| `sample_rate`, `measured_rate`, `n_samples`, `duration_s`, `lead` | how it was recorded |
| `bpm`, `beat_strength`, `quality_json` | what the signal checks found |
| `probability`, `threshold`, `mode`, `prediction` | the verdict and the cut-off it used |
| `model_name`, `model_version`, `inference_ms` | which model said so, and how long it took |

The waveform is stored because a probability with nothing behind it can never be
checked afterwards. One row per window, not per sample: the model only ever
reads whole windows, so a row per sample would be a thousand inserts every ten
seconds and would buy nothing. `model_version` records the model actually
serving, so a result can always be traced back to it — three trained runs exist
with different AUCs, and a stored result must not be ambiguous about which one
produced it.

### Running the hardware stand-in

```
python app.py                                    # in one terminal
python esp32_simulator.py --device ESP32-001     # in another
```

Enrol a patient with device id `ESP32-001` first (dashboard → **+ enrol**), or
the readings arrive unattributed — stored, but belonging to nobody. Add
`--noise 0.05` to roughen the clean hospital signal and see how the model holds
up on something closer to what a cheap electrode produces.

Storing `inference_ms` on every row means the speed of the system can be measured
with one query instead of a stopwatch:

```sql
SELECT AVG(inference_ms) FROM predictions;
```

You can open the file with any SQLite browser, or from Python:

```python
import models
conn = models.connect()
for row in conn.execute("SELECT COUNT(*), label FROM recordings GROUP BY label"):
    print(dict(row))
```

---

## The web endpoints

| Endpoint | What it does |
|---|---|
| `GET /` | the dashboard |
| `GET /api/record?type=mi\|normal\|any&mode=…&patient_id=…` | replays a random test-fold recording |
| `POST /api/predict` | **the endpoint the ESP32 uses** |
| `POST /api/rescore` | re-scores a signal already on screen **without storing it** |
| `GET /api/patients` · `POST /api/patients` | list or enrol patients |
| `GET /patient/<id>` | one patient's record and reading history |
| `GET /api/history?limit=20` | the recent predictions from the database |
| `GET /api/health` | check the server, model and database are alive |

### Why `/api/rescore` is separate from `/api/predict`

The dashboard lets you flip between the three operating points to watch the
verdict change on the trace in front of you. It used to do that by POSTing the
trace to `/api/predict`, which logged every flip as a **new reading from a
device with no known truth**. Three clicks on one simulated recording therefore
wrote three phantom `device` rows and discarded the ground truth we had.

That quietly corrupted two reported numbers: the simulation/device split, and
`accuracy_so_far`, which only averages rows whose truth is known. Re-scoring is
a way of *looking* at a result, not a new event, so it now goes to an endpoint
that stores nothing. Looking at a result can no longer change the results.

The device endpoint:

```
POST /api/predict
{"signal": [ ...1000 numbers... ], "mode": "high_recall"}

->  {"probability": 0.83, "threshold": 0.29, "prediction": 1,
     "label": "MI PATTERN DETECTED", "inference_ms": 41.2}
```

Send exactly **1000 numbers** (10 seconds at 100 Hz) of Lead I, **raw** — the server
does the cleaning itself, using the same code the model was trained with.

---

## Signing in

Human-facing pages now require an account. The first time the app starts against
a database with no users it creates an `admin` account and prints a generated
password **once**, to the console. Sign in with it and change it.

A fixed default password would ship a known credential in a public repository,
which is why the password is generated rather than hard-coded.

Set `ECG_SECRET_KEY` in the environment for anything beyond local use. Without
it a key is generated per process, so every restart signs everybody out.

Devices are not people and are authenticated differently: an ESP32 has no
browser and cannot hold a session cookie, so it identifies itself with a
`device_id` and a `device_token` held in its firmware. `/api/predict` and the
MQTT bridge therefore do not require a session.

---

## Running over MQTT

USB tied the device to one laptop by a cable. MQTT does not: the device and the
backend each make an *outbound* connection to a broker, so neither needs a
public address or a forwarded port, and the laptop no longer has to be attached
to the ESP32 at all.

### 1. Configure the secrets

Copy `.env.example` to `.env` and fill it in. `.env` is not in version control
and nothing in the code has a secret as its default.

```
ECG_SECRET_KEY=...            python -c "import secrets; print(secrets.token_hex(32))"
MQTT_ENABLED=1
MQTT_HOST=127.0.0.1
MQTT_USERNAME=ecg-server
MQTT_PASSWORD=...
```

### 2. Start the broker

Mosquitto is the broker; this project does not implement one. Writing an MQTT
broker would mean re-implementing mature, security-sensitive infrastructure for
no research benefit.

```
winget install EclipseFoundation.Mosquitto        (once, Windows)
```

The installer registers a service that listens on 127.0.0.1 only, which the
ESP32 cannot reach. Stop it and run the project's own configuration instead,
which listens on every interface:

```
net stop mosquitto                                 (needs administrator)
cd mosquitto
"C:\Program Files\mosquitto\mosquitto.exe" -c mosquitto.windows.conf -v
```

Create the accounts before the first start. Anonymous access is refused, so an
empty password file means nobody can connect, including the backend:

```
"C:\Program Files\mosquitto\mosquitto_passwd.exe" -c passwd ecg-server
"C:\Program Files\mosquitto\mosquitto_passwd.exe"    passwd esp32-001
```

`mosquitto/acl` then restricts each account to its own topics: a device may
publish only under its own device id, and only the backend may answer. Without
that, one compromised device could file readings against another user.

Windows Firewall blocks port 1883 from other machines by default. Allow it once:

```
New-NetFirewallRule -DisplayName "ECG prototype MQTT broker (1883)" `
  -Direction Inbound -Protocol TCP -LocalPort 1883 -Action Allow    (administrator)
```

### 3. Flash the device

Copy `firmware/ecg_esp32_mqtt/secrets_example.h` to `secrets.h` in the same
folder and fill in the Wi-Fi network, the broker address (this laptop's address
on that network, **not** 127.0.0.1) and the device account. Then:

```
arduino-cli compile --upload -p COM7 --fqbn esp32:esp32:esp32 firmware/ecg_esp32_mqtt
```

The ESP32 joins WPA2-Personal networks only: a home network or a phone hotspot,
not WPA2-Enterprise university Wi-Fi.

### 4. Start the app and claim the device

```
python app.py
```

Sign in, open **Devices**, and enter the device id the firmware prints at
start-up (`ESP32-001` by default). An ECG published by that device from then on
belongs to that account, and to no other. Open **Live ECG** to watch it arrive.

If the broker is unreachable the app still starts and says so: the dashboard,
the simulation page and the HTTP endpoint keep working, and only the MQTT
transport is missing.

### Testing without the hardware

```
python mqtt_stream_simulator.py --username esp32-001 --password ... --seconds 12
python mqtt_stream_simulator.py --username esp32-001 --password ... --drop 5
```

The first publishes twelve one-second chunks in exactly the format the firmware
uses. The second deliberately loses one, which the backend must notice: the
window spanning the gap is discarded rather than stitched together. This also
separates faults - if the simulator works and the device does not, the fault is
in the device.

### Topics

No patient information appears in a topic; a device id is the only identifier.

| Topic | Direction | Payload |
|---|---|---|
| `ecg/devices/<id>/data` | device to server, QoS 0 | one second of samples |
| `ecg/devices/<id>/status` | device to server, QoS 1, retained | `online` / `offline`, firmware, IP, RSSI |
| `ecg/devices/<id>/event` | device to server, QoS 1 | reconnects, sampling overruns |
| `ecg/devices/<id>/result` | server to device, QoS 1 | the screening verdict |
| `ecg/<id>/reading` | device to server, QoS 1 | legacy: one whole 1000-sample window |
| `ecg/<id>/result` | server to device, QoS 1 | legacy verdict |

The chunk payload:

```json
{"device_id": "ESP32-001", "seq": 41, "t_us": 123456789, "fs": 100, "n": 100,
 "lop": "000...0", "lom": "000...0", "adc": [2071, 2068, ...]}
```

`seq` increases by one per chunk, so a lost or duplicated chunk is visible
rather than silent. `t_us` is the device's own clock at the first sample, which
is what the measured sampling rate is computed from - the arrival time at the
laptop includes network jitter and would flatter the result. `lop` and `lom`
carry one character per sample, so lead-off is known per sample rather than per
second. The device sends **raw ADC counts and no normalisation**: the server
z-scores the window with the same code the model was trained with, so the two
can never disagree about a statistic.

### What the backend refuses

A window is scored only if all 1000 samples are lead-on, gap-free, unsaturated
and contain a detectable heartbeat. Anything else is discarded and counted, and
the counts appear on the live page. Missing data is never interpolated: a
stitched-over gap produces a waveform that looks plausible and never happened,
and the model would score it as confidently as a real one.

| Reason | Meaning |
|---|---|
| `lead_off` | an electrode was off during the window |
| `chunk_gap` | a chunk was lost on the way |
| `device_restart` | the sequence counter began again; the device rebooted |
| `malformed_chunk` | the payload failed validation |
| `saturated` | the ADC sat on a rail: not an ECG |
| `no_heartbeat` | electrically fine, but no QRS complexes |

---

## Tests

```
python -m pytest tests/ -q
```

131 tests. They run against a temporary database and a stub model, so they never
touch `ecg.db` and do not need the trained `.keras` files. That keeps the suite
fast and keeps it testing this system's logic rather than TensorFlow's.

---

## Security limitations of the prototype

Stated here rather than left implicit, and repeated in the thesis.

- **Port 1883 is unencrypted.** Physiological data and the device token travel
  in clear text. A deployment needs TLS on 8883.
- **`REQUIRE_DEVICE_TOKEN` is off by default**, so by default any client that
  knows a device id can submit readings as that device.
- **MQTT and serial recordings store their waveform** in `ecg_reports`, so a
  report can be reopened and checked. The older `/api/predict` endpoint still
  stores only the verdict, so a reading posted that way cannot be replayed.
- **No CSRF tokens.** The forms rely on `SameSite=Lax` session cookies. A
  deployment needs proper CSRF protection.
- **No rate limiting.** Nothing stops a client submitting readings continuously.
- Not a medical device, and not validated for clinical use.

---

## Two rules that keep the results honest

**1. The same patient is never in two groups.** PTB-XL numbers every recording 1–10
with this already taken care of, so the split just uses those numbers:

| Folds | Used for |
|---|---|
| 1–8 | training — the model learns from these |
| 9 | validation — choosing when to stop, and the threshold |
| 10 | test — touched once, at the very end |

**2. Nothing is ever chosen by looking at the test fold.** The threshold is picked on
fold 9 and applied unchanged to fold 10. Trying thresholds on fold 10 and keeping
the best one would give a flattering, meaningless score.

The dashboard only ever replays fold 10 for the same reason — demoing on data the
model trained on would look impressive and prove nothing.

---

## Why the threshold matters

The model outputs a probability between 0 and 1. Turning that into a yes/no answer
needs a cut-off, and **0.5 is the wrong one for medicine** — it treats a false alarm
and a missed heart attack as equally bad, when a missed heart attack is far worse.

So the system offers three operating points, and the dashboard lets you switch
between them on the *same* recording to watch the verdict change:

| Mode | What it is for |
|---|---|
| `default` | the naive 0.5 cut-off, shown to demonstrate why it is a poor choice |
| `best_f1` | the most balanced trade-off |
| `high_recall` | catches at least 90% of real heart attacks — what a screening tool should use |

Lowering the threshold catches more real heart attacks but raises more false alarms.
That trade is a medical decision, not a technical one.

---

## Notes

- **Not a medical device.** Not validated for clinical use.
- The model is trained on clean hospital recordings. A real ESP32 signal will be
  noisier — a known limitation.
- Of the 5,469 MI recordings used, only **146 (2.7%)** are labelled *acute*
  (`infarction_stadium1 = "Stadium I"` in `ptbxl_database.csv`); the rest are
  prior, healed, or of unrecorded stage. Note that 3,372 of them (61.7%) have an
  unknown stadium, so among the 2,097 whose stage *is* recorded, 7.0% are acute —
  which is the figure an earlier draft rounded to "about 6%". The unambiguous
  number to quote is **2.7% of MI recordings**, because it does not depend on
  discarding the majority whose stage was never filled in. Either way the system
  detects MI **patterns**, so it is best described as a screening and triage
  tool, not an emergency alarm.
- This trains one model on a laptop CPU. The five-model GPU ensemble from the
  Colab notebook scores a little higher and can be dropped in later without
  changing how any of this is structured.
