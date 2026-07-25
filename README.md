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

## How to run it (two commands)

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
it found (expect roughly **~9000 Normal** and **~5000 MI**).

### 2. Train and test the model
```
python 2_train_cnn.py
```
This trains the CNN and then tests it on recordings it has never seen. It prints
the scores and saves everything into `outputs/`.

You can re-run step 2 as many times as you like without redoing step 1.

---

## What you get (in `outputs/`)

| File | What it is |
|------|-----------|
| `mi_cnn_model.keras` | The trained model. The Flask web app will load this later. |
| `metrics.txt` | Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrix. |
| `training_curves.png` | Graphs of accuracy/loss while training — good figure for the report. |
| `confusion_matrix.png` | A grid of truth vs prediction — another report figure. |

---

## How to read the results

- **Accuracy** — overall, how many predictions were correct.
- **Precision** — when it shouts "heart attack!", how often it was right.
- **Recall (Sensitivity)** — of the *real* heart attacks, how many it caught.
  **This is the most important one** in health: missing a real heart attack is
  worse than a false alarm.
- **F1-score** — a fair single score that balances precision and recall.
- **ROC-AUC** — how well it separates the two classes overall (1.0 = perfect,
  0.5 = random guessing).
- **Confusion matrix** — the raw counts of right/wrong for each class.

---

## Notes / troubleshooting

- Needs Python with TensorFlow. If TensorFlow is hard to install on your Python
  version, make a virtual environment with **Python 3.11** and try again, or use
  `pip install tensorflow-cpu`.
- No GPU is needed — this small model trains on a normal laptop CPU in a few minutes.
- This trains on clean hospital data. Real signals from your ESP32 device will be
  noisier, which is a known limitation discussed in the thesis.
