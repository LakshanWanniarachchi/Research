# Heart Attack Detection Model — Training

This folder trains a simple deep learning model (a **1D CNN**) that looks at a
single-lead ECG signal and decides:

- **Normal** (a healthy heart), or
- **MI** = Myocardial Infarction (a heart attack).

It uses the **PTB-XL** dataset that is already downloaded one folder up.

Everything is kept as simple as possible on purpose — this is an undergraduate
research project, not a hospital product.

---

## The choices we made (and why)

| Choice | What we picked | Why |
|--------|----------------|-----|
| Lead | **Lead I** only | A cheap 2-electrode device (ESP32 + AD8232) can realistically get Lead I. |
| Sample rate | **100 Hz** | Smaller and faster; the standard PTB-XL benchmark rate. Runs fine on a laptop. |
| Classes | **Normal vs MI** | The clearest, most useful two-class problem for this project. |
| Data split | **Official folds** | PTB-XL already numbers records 1–10 so the *same patient never leaks* between train and test. |
| Cleaning | **z-score normalise** | One simple, standard step so signal size doesn't confuse the model. |
| Balance | **Class weights** | There are more Normal than MI records; this stops the model cheating by always saying "Normal". |

---

## Two ways to train

| | Where it runs | Use it when |
|---|---|---|
| **A. Google Colab** (recommended) | Free GPU in the browser | You want the faster, bigger model and the full evaluation (ROC/PR curves, threshold tuning, confidence intervals). |
| **B. Local** | Your own laptop CPU | You just want a quick baseline without uploading anything. |

Both start from the same step 1.

### 0. Install the libraries (once)
```
pip install -r requirements.txt
```

### 1. Build the dataset (once, ~a few minutes)
```
python 1_make_dataset.py
```
This reads the ECG files, keeps the clean Normal and MI recordings, takes Lead I,
and saves fast-loading `.npy` files into `data/`. It prints how many of each class
it found (**9,069 Normal** and **5,469 MI**, 14,538 in total).

---

## Route A — Google Colab (recommended)

### 2a. Pack the data into one file
```
python 3_export_for_colab.py
```
This squeezes `X.npy`, `y.npy` and `folds.npy` into a single **22 MB** file,
`data/ptbxl_leadI.npz`. One file is much easier to upload and to point at from
Colab than three.

### 3a. Put it in Google Drive
1. In Google Drive, make a folder called **`ecg_project`** in *My Drive*.
2. Upload `data/ptbxl_leadI.npz` into it.

So the final path is `MyDrive/ecg_project/ptbxl_leadI.npz`. If you prefer a
different folder, change the single `DRIVE_DIR` line near the top of the notebook.

### 4a. Run the notebook
1. Go to [colab.research.google.com](https://colab.research.google.com) → **Upload** →
   choose `colab_train_eval.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**, and approve the Google Drive pop-up when it appears.

Training takes about **3–6 minutes** on the GPU. Everything it produces is written
back to `MyDrive/ecg_project/outputs/`, so it is still there after the Colab
session shuts down.

> **Note:** the raw PTB-XL download never goes to Drive — only the 22 MB packed
> file. That keeps the upload quick and stays inside a free Drive quota.

### 5a. (Optional) Try to improve it

Upload `colab_train_improved.ipynb` and run it the same way. It keeps the same
data and the same network, and adds two things that need no new recordings:

- **Data augmentation** — random time shift (±1 s), baseline wander (0.15–0.5 Hz)
  and electrical noise, applied fresh every epoch. These mimic real electrode
  artefacts, so this also builds evidence for **RQ1**.
- **A 5-seed ensemble** — the same network trained 5 times and averaged.

It scores the old baseline and the new ensemble on the same test fold and reports
whether the difference is **statistically real** (bootstrap CI of the difference),
rather than just showing a bigger number. Takes ~25–35 minutes on a T4.

Results land in `outputs/ensemble/`. Download that whole folder into this project
and the Flask app picks it up automatically — no code change needed.

> **Do not tune against fold 10.** Everything is chosen on fold 9 (validation).
> If you repeatedly change things based on the test fold, it stops being a fair
> test — the same leakage problem the fold split exists to prevent.

---

## Route B — Local training

```
python 2_train_cnn.py
```
This trains the smaller baseline CNN on your CPU and tests it on recordings it has
never seen, saving results into `outputs/`. It is the simpler script and a fair
baseline to compare the Colab model against in the report.

You can re-run this as many times as you like without redoing step 1.

---

## What you get (in `outputs/`)

| File | What it is | Route |
|------|-----------|-------|
| `mi_cnn_model.keras` | The trained model. The Flask web app will load this later. | both |
| `metrics.txt` | Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrix. | both |
| `training_curves.png` | Loss / accuracy / AUC while training — good figure for the report. | both |
| `confusion_matrix.png` | Truth vs prediction grid — another report figure. | both |
| `roc_pr_curves.png` | ROC and Precision–Recall curves. | Colab |
| `metrics.json` | The same scores in machine-readable form, incl. confidence intervals. | Colab |
| `metrics_table.csv` | The results table — paste straight into the thesis. | Colab |
| `model_config.json` | Sample rate, input length, normalisation and the chosen threshold. **The Flask app needs this to preprocess live signals the same way.** | Colab |
| `test_predictions.npz` | Raw test-set probabilities, so you can redo any analysis without retraining. | Colab |

---

## What the Colab notebook does beyond the local script

The local script is deliberately minimal. The notebook adds the things a thesis
examiner will look for:

- **A bigger model** — four convolution blocks instead of three, two convolutions
  per block, and both average- and max-pooling at the end. Practical because the
  GPU makes it cheap.
- **Threshold tuning done honestly.** The 0.5 cut-off is arbitrary and wrong for
  medicine, where a missed heart attack costs far more than a false alarm. The
  notebook picks the threshold on the **validation fold** and then applies it
  unchanged to the test fold. Picking it on the test fold would inflate the score.
- **Two operating points reported** — best-F1 (balanced) and high-recall (catches
  ≥ 90% of real MIs), because a screening tool would be tuned toward the second.
- **ROC-AUC and PR-AUC**, which describe performance at every threshold. PR-AUC is
  the fairer headline when classes are unbalanced, as they are here.
- **Bootstrap 95% confidence intervals.** The test fold is only ~1,460 recordings,
  so every score carries sampling noise. Reporting `0.87 (0.85–0.89)` instead of a
  bare `0.87` is the honest way to write it up, and it shows which differences
  between models are real rather than luck.

---

## Running the web app (testing the model)

This is the "simulation mode" from the thesis system diagram: it replays real ECG
recordings through the same code path the ESP32 will use once the hardware is ready.

```
cd webapp
python app.py
```

Then open **http://127.0.0.1:5000**. Startup takes 20–30 seconds (TensorFlow, the
models, and the dataset all load once). Press `Ctrl + C` to stop.

**What it needs** (all should already be in the project root):

| File / folder | Purpose |
|---|---|
| `mi_cnn_model.keras` + `model_config.json` | the single baseline model |
| `ensemble/` (5 × `.keras` + `ensemble_config.json`) | the 5-model ensemble — used automatically if present |
| `data/ptbxl_leadI.npz` | the recordings replayed in simulation mode |

If the ensemble folder is missing the app falls back to the single model and says so
in the page header, so you always know which one you are looking at.

**Using it:** the buttons pull a random heart-attack or normal recording from
**fold 10 only** — the test fold the model never trained on. Demoing on training data
would give falsely confident results. The dashboard shows the ECG trace, the model's
probability, the threshold, and the true diagnosis so you can see whether it was right.
The operating-point buttons re-score the *same* recording, so you can watch the
verdict change as the threshold moves.

### The endpoint the ESP32 will use

```
POST /api/predict     {"signal": [ ...1000 numbers... ], "mode": "high_recall"}
  -> {"probability": 0.83, "threshold": 0.2974, "prediction": 1, "label": "MI PATTERN DETECTED"}

GET  /api/health      check the server is up and see which model is loaded
```

The signal must be exactly **1000 samples** (10 seconds at 100 Hz) of Lead I. The
server does the z-scoring itself, so send the raw values. Server-side prediction
takes about **0.35 s** with the 5-model ensemble.

---

## How to read the results

- **Accuracy** — overall, how many predictions were correct.
- **Precision** — when it shouts "heart attack!", how often it was right.
- **Recall (Sensitivity)** — of the *real* heart attacks, how many it caught.
  **This is the most important one** in health: missing a real heart attack is
  worse than a false alarm.
- **Specificity** — of the *healthy* people, how many it correctly left alone.
  The mirror image of recall; a model that flags everyone has perfect recall and
  terrible specificity, so the two must be read together.
- **F1-score** — a fair single score that balances precision and recall.
- **ROC-AUC** — how well it separates the two classes overall (1.0 = perfect,
  0.5 = random guessing).
- **PR-AUC (Average Precision)** — the same idea, but it ignores the easy
  true-negatives, which makes it the fairer headline number on unbalanced data.
- **Confusion matrix** — the raw counts of right/wrong for each class. The
  **bottom-left cell is missed heart attacks** — the number to keep small.

---

## Notes / troubleshooting

- Needs Python with TensorFlow. If TensorFlow is hard to install on your Python
  version, make a virtual environment with **Python 3.11** and try again, or use
  `pip install tensorflow-cpu`. Colab already has TensorFlow installed, so
  route A needs no local TensorFlow at all — only `numpy` for step 1 and the export.
- No GPU is needed for route B — the small model trains on a laptop CPU in a few
  minutes. Route A uses Colab's free T4 GPU.
- **Colab: "Could not find .../ptbxl_leadI.npz"** — the file is not where the
  notebook expects. Check it sits directly inside `MyDrive/ecg_project/`, then
  re-run that cell (no need to re-mount Drive).
- **Colab disconnects if left idle.** If it drops mid-training, re-run from the top;
  the dataset is still in Drive, so only the training has to be redone.
- This trains on clean hospital data. Real signals from your ESP32 device will be
  noisier, which is a known limitation discussed in the thesis.
- Not a medical device, and not validated for clinical use.
