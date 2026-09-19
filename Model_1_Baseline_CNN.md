# IoT-Based Heart Attack Pattern Detection Using Deep Learning

### Model 1 — The Baseline 1D CNN

*An explanation of the notebook `colab_train_eval.ipynb`*

W A K Lakshan Lakshitha | Student ID: 28508 | BSc (Hons) Computer Science
Supervisor: Mr. Madusanka Mithrananda

---

## 1. What this notebook is for

This notebook trains the first, simplest version of the heart-attack detection model. It is
the baseline: the honest starting point that every later improvement has to beat.

It answers one yes/no question. Given ten seconds of a single-lead ECG, is this recording
**Normal** or does it show a pattern of **Myocardial Infarction** (MI, a heart attack)?

It runs on Google Colab rather than on a laptop, because Colab lends you a free graphics
card (GPU) and training finishes in about three to six minutes instead of roughly ten times
longer on a normal processor.

---

## 2. What goes into the model

The model is deliberately given very little to work with, and that is a design decision
rather than a limitation of the dataset:

| Property | Value | Why this choice |
|---|---|---|
| Leads used | **Lead I only** | A cheap two-electrode device (ESP32 + AD8232) can physically obtain Lead I and nothing more. Training on all 12 leads would produce a model the hardware could never feed. |
| Length | 10 seconds | The standard PTB-XL recording length. |
| Sample rate | 100 Hz | 100 measurements per second. |
| Numbers per recording | **1000** | 10 seconds × 100 Hz = 1000 values. |
| Recordings | 14,538 | 9,069 Normal and 5,469 MI, filtered from PTB-XL. |

The notebook does not read the original PTB-XL files. It reads one packed file,
`ptbxl_leadI.npz` (about 22 MB), which was prepared on the laptop beforehand and uploaded
to Google Drive.

### 2.1 How the data is split

PTB-XL already numbers every recording from 1 to 10. Those numbers were built so that the
same patient never appears in two different groups. The notebook simply uses them:

| Folds | Used as | What it means |
|---|---|---|
| 1 – 8 | **Training** | The model learns from these 11,622 recordings. |
| 9 | **Validation** | 1,454 recordings used to decide when to stop training and where to set the cut-off. |
| 10 | **Test** | 1,462 recordings, looked at once at the very end. The real exam. |

> **Why this matters.** If the same patient appeared in both training and test, the model
> could recognise the person rather than the illness, and the final score would be flattering
> and meaningless. Using the official fold numbers prevents that automatically.

---

## 3. What the model is, in plain terms

The model is a **1D Convolutional Neural Network** — usually shortened to **1D CNN**.

The easiest way to picture it is by comparison with photographs. A normal CNN, the kind used
to recognise cats in pictures, slides small filters across an image looking for edges,
corners and textures. A 1D CNN does exactly the same thing, but it slides those filters
along a line instead of across a picture.

An ECG is a line that rises and falls over time, so this fits naturally. The filters learn
to spot shapes such as the sharp spike of a heartbeat or the sagging segment that follows
it, and because the filter slides along the whole recording, it finds those shapes wherever
they happen to occur in the ten seconds. Nobody tells the model what a heart attack looks
like. It works out which shapes matter by being shown thousands of examples of each kind.

The model's final answer is not a yes or a no. It is a single number between 0 and 1: its
estimated probability that this recording shows an MI. Turning that number into a decision
is a separate step, covered in section 6.

---

## 4. The structure of the network

The network is built from four repeating blocks, followed by a small decision part. Each
block performs the same four steps:

| Step | What it does |
|---|---|
| **Conv1D** | Slides filters along the signal to find patterns. This is the part that actually learns. |
| **BatchNormalization** | Keeps the numbers in a sensible range so training is stable and faster. |
| **ReLU** | Allows the network to learn curved rules rather than only straight-line ones. |
| **MaxPooling1D** | Halves the length of the signal, keeping the strongest responses. |
| **Dropout** | Randomly switches off some connections while training, which discourages memorising. |

Stacking four of these blocks matters. Each block halves the length, so later blocks see a
wider stretch of time at once. The first block looks at fractions of a second; by the fourth
block the network is looking at whole heartbeats. The four blocks are:

| Block | Filters | Filter width | Dropout | Length after the block |
|---|---|---|---|---|
| 1 | 32 | 7 | 0.10 | 1000 → 500 |
| 2 | 64 | 5 | 0.10 | 500 → 250 |
| 3 | 128 | 3 | 0.20 | 250 → 125 |
| 4 | 128 | 3 | 0.20 | 125 → 62 |

Each block actually contains **two** convolutions rather than one — the second refines what
the first found.

### 4.1 The decision part at the end

After the four blocks, the signal has been reduced to 62 time steps with 128 descriptions
each. That still has to become one number. This happens in four steps:

- **Two kinds of pooling at once.** The average response answers "was this pattern present
  overall?", and the maximum response answers "did it appear strongly anywhere?" Both are
  kept and joined together, because the two questions carry different information.
- **Dense(64).** A small layer that weighs up that summary.
- **Dropout(0.5).** Switches off half the connections during training, which is a strong
  defence against over-fitting.
- **Dense(1, sigmoid).** Produces the single probability between 0 and 1.

### 4.2 How big is it?

| Measure | Value |
|---|---|
| Total parameters | **229,473** |
| Trainable parameters | 228,065 |
| Non-trainable parameters | 1,408 |
| Layers | 39 |

A parameter is one adjustable number inside the network. Training means repeatedly nudging
all 229,473 of them until the model's answers match the known diagnoses. The 1,408
non-trainable ones belong to the BatchNormalization layers, which record running averages
rather than learning directly.

By modern standards this is a small network, and that is intentional. A larger model would
memorise 11,622 recordings rather than learn from them.

---

## 5. How it is trained

| Setting | Value | Meaning |
|---|---|---|
| Optimiser | Adam, rate 0.001 | The method used to adjust the parameters. |
| Loss | binary cross-entropy | The standard scoring rule for a yes/no question. |
| Maximum epochs | 60 | An epoch is one pass through all the training data. |
| Batch size | 64 | How many recordings are looked at before each adjustment. |
| Random seed | 42 | Fixed, so re-running gives the same result. |

### 5.1 Class weights

There are roughly 1.7 Normal recordings for every MI recording. Left alone, a lazy model
could score about 62% simply by answering "Normal" every single time. Class weights make
each MI mistake cost more, so that shortcut stops paying off.

### 5.2 Three helpers that run automatically

- **EarlyStopping.** Watches the validation AUC. If it has not improved for 10 epochs,
  training stops and the best weights seen are restored. This is the main defence against
  over-fitting.
- **ReduceLROnPlateau.** When progress stalls, take smaller steps rather than giving up.
- **ModelCheckpoint.** Saves the model to Google Drive every time the validation score
  improves.

---

## 6. Turning the probability into an answer

The model gives a probability. To say "MI" or "Normal" you need a cut-off, called a
threshold. The obvious choice of 0.5 is the wrong one for medicine, because it treats a
false alarm and a missed heart attack as equally bad. They are not: missing a real heart
attack is far worse.

So the notebook tries many cut-offs on the **validation** fold and reports three, then
applies them unchanged to the test fold. The values recorded for this model were:

| Operating point | Threshold | What it is for |
|---|---|---|
| Default | 0.5000 | The naive cut-off, shown mainly to demonstrate why it is a poor choice here. |
| Best F1 | 0.3136 | The most balanced trade-off between false alarms and misses. |
| **High recall** | **0.1406** | Catches at least 90% of real heart attacks. What a screening tool should use. |

> **The rule that keeps this honest.** The thresholds are chosen on fold 9 and then applied
> to fold 10 without further change. Trying thresholds on fold 10 and keeping whichever
> scored best would inflate the result and stop it being a fair test.

---

## 7. How many models does this notebook create?

**Exactly one.**

| What is produced | How many | File name |
|---|---|---|
| Trained model | **1** | `mi_cnn_model.keras` |

> **A point that often causes confusion.** Training runs for many epochs, and the model is
> saved repeatedly during that time. It is saved to the SAME file each time, overwriting the
> previous version whenever the score improves. So you end up with one file containing the
> best version found — not one file per epoch.

---

## 8. Measured result

| Measure | Result | Meaning |
|---|---|---|
| **ROC-AUC** | **0.9017** | How well the two classes separate overall. 1.0 is perfect, 0.5 is random guessing. |

This is the number the second notebook sets out to beat.

> **Where this figure comes from.** The notebook's saved outputs were cleared before it was
> committed, so it contains no stored results. This value is taken from the project's
> recorded configuration files and the *Research Objectives, Novelty and Outcomes* document.
> Re-running the notebook will reproduce it, but it is not currently printed inside the
> notebook file itself.

---

## 9. Where the model is saved and how to access it

### 9.1 Where it is saved

Everything is written into Google Drive, not onto the Colab machine. This matters because
Colab deletes its own machine when the session ends; only what was written to Drive survives.

```
MyDrive/ecg_project/
    ptbxl_leadI.npz            <- the input you upload before running
    outputs/
        mi_cnn_model.keras     <- THE TRAINED MODEL
        model_config.json      <- thresholds and settings
        metrics.json  metrics.txt  metrics_table.csv
        training_curves.png  confusion_matrix.png  roc_pr_curves.png
        test_predictions.npz
```

The single folder path is set in one line near the top of the notebook, so it is easy to
change:

```python
DRIVE_DIR = "/content/drive/MyDrive/ecg_project"
```

### 9.2 Opening the model in Python

The file is a standard Keras model. Loading it takes one line:

```python
import tensorflow as tf

model = tf.keras.models.load_model("mi_cnn_model.keras")

# signal must be 1000 numbers, z-scored, shaped (1, 1000, 1)
probability = float(model.predict(signal)[0][0])
```

> **The model alone is not enough.** You also need `model_config.json`. It records that the
> input must be 1000 samples at 100 Hz and that each recording is z-scored before being fed
> in. If live signals are prepared differently from the training data, the model still
> returns confident-looking numbers — they are simply wrong, and nothing crashes to warn you.

### 9.3 Getting it onto your laptop

There are two ways:

- **From Google Drive directly** — open the `ecg_project/outputs/` folder and download the
  file.
- **From the notebook** — the last code cell calls `files.download(...)`, which sends the
  model and its config straight to your browser's downloads folder.

### 9.4 How the Flask web application uses it

The web application looks for an ensemble folder first (that is Model 2). If no ensemble is
present, it falls back to this single baseline model and says so in the page header, so you
always know which model produced a result. The relevant code is in `webapp/app.py`.

> **Important: this file is not in the project repository.** Trained models are excluded by
> `.gitignore` because they are large and are regenerated by training. Only the small JSON
> configuration files are stored in the project. If you clone this project onto a new
> machine, `mi_cnn_model.keras` will not be there — you must copy it down from Google Drive
> or re-run this notebook.

---

## 10. Honest limitations

- **Trained on clean hospital recordings.** A real signal from an ESP32 and AD8232 will be
  noisier, so live performance will be lower than the figure above.
- **One lead carries less information than twelve.** The standard clinical criteria for
  locating an infarction need the other leads, so some infarctions are genuinely hard to see
  in Lead I.
- **It detects patterns, not emergencies.** Only 146 of the 5,469 MI recordings — **2.7%** —
  are labelled acute (`infarction_stadium1 = "Stadium I"`); the rest are prior, healed, or of
  unrecorded stage. (61.7% have an unknown stadium; among those with a recorded stage, 7.0%
  are acute, which is what an earlier draft rounded to "about 6%".) The system is best
  described as a screening and triage tool.
- **Not a medical device.** Not validated for clinical use.
