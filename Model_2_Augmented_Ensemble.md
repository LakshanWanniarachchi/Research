# IoT-Based Heart Attack Pattern Detection Using Deep Learning

### Model 2 — The Augmented 5-Model Ensemble

*An explanation of the notebook `colab_train_improved.ipynb`*

W A K Lakshan Lakshitha | Student ID: 28508 | BSc (Hons) Computer Science
Supervisor: Mr. Madusanka Mithrananda

---

## 1. What this notebook is for

This notebook tries to beat the baseline model without collecting a single new recording. It
uses two well-established techniques, applies both, and then — importantly — checks whether
the improvement is real or just luck.

The task is unchanged: given ten seconds of Lead I ECG, decide Normal versus Myocardial
Infarction (MI, a heart attack). The dataset, the preprocessing and the fold split are all
identical to the baseline notebook.

| Change | Idea in one sentence |
|---|---|
| **Data augmentation** | Make slightly messed-up copies of the training recordings every epoch, so the model sees more variety and learns to cope with noise. |
| **A 5-model ensemble** | Train the same network five times from different random starting points and average their answers. |

Running time is roughly 25 to 35 minutes on a Colab T4 GPU, compared with three to six
minutes for the baseline — because it trains five models instead of one, each for longer.

---

## 2. What the model is

The network is a **1D Convolutional Neural Network** and it is **exactly the same
architecture as the baseline**. Not a similar one — the same one. Nothing about the network
was changed.

A 1D CNN slides small filters along a signal the way an image recogniser slides them across
a photograph. Because an ECG is a line rising and falling over time, the filters learn to
recognise beat shapes wherever they occur in the ten seconds. The output is one number
between 0 and 1: the estimated probability of MI.

### 2.1 The structure, in brief

| Part | Detail |
|---|---|
| Four convolution blocks | 32, 64, 128 and 128 filters; widths 7, 5, 3, 3; two convolutions per block, each followed by BatchNormalization and ReLU, then MaxPooling and Dropout. |
| Two-way pooling | Average pooling (was the pattern present overall?) and maximum pooling (did it appear strongly anywhere?), joined together. |
| Decision layers | Dense(64), Dropout(0.5), then Dense(1) with a sigmoid to give the probability. |

| Measure | Value |
|---|---|
| Total parameters | **229,473** |
| Trainable parameters | 228,065 |
| Non-trainable parameters | 1,408 |
| Layers | 39 |

> **So what actually changed?** Only the training. The architecture, the dataset, the Lead I
> input, the z-score preprocessing and the fold split are all identical to Model 1. This is
> deliberate: if you change the network and the training at the same time, you cannot tell
> which change caused any difference in the result.

---

## 3. Change one — data augmentation

Every epoch, each training recording is altered slightly before the model sees it, so the
model never sees exactly the same input twice. Three alterations are applied, each chosen
because it imitates something that genuinely happens to a real ECG:

| Augmentation | What it imitates in real life | Setting |
|---|---|---|
| **Time shift** | The heartbeat not starting neatly at the beginning of the recording. | ±1 second |
| **Baseline wander** | Breathing and chest movement slowly pulling the trace up and down. | 0.15–0.5 Hz, amplitude ≤ 0.15 |
| **Electrical noise** | Interference from cheap electrodes and wiring. | Gaussian, σ ≤ 0.05 |

Two details are worth understanding, because both were deliberate decisions:

- **The time shift uses reflect padding, not wrapping.** Wrapping the end of the signal round
  to the start would create an artificial jump that does not exist in any real ECG, and the
  model would learn to look for it. Reflecting avoids that.
- **There is no amplitude scaling, on purpose.** It would do nothing at all. The preprocessing
  z-scores each recording last, and scaling a signal before z-scoring gives back exactly the
  same result — the normalisation cancels it out. That particular robustness is already free.

Each augmented signal is **re-z-scored at the very end**, because that is the order the live
device will use: whatever arrives from the ESP32 is normalised last.

> **This also supports Research Question 1.** Because the augmentations imitate real
> electrode artefacts — movement, breathing and electrical interference — this experiment is
> evidence about how well the model copes with a messy signal, not only about raw accuracy.

*The validation and test recordings are never augmented. They must stay realistic, otherwise
the score would not describe real performance.*

---

## 4. Change two — training five models

The same network is trained five separate times. The only difference between the five runs
is the random seed, which controls the random starting values of the parameters and the
order the data is shuffled in.

```python
SEEDS = [42, 43, 44, 45, 46]
```

Averaging the five answers is what makes the ensemble. The reasoning is simple: each model
makes some mistakes because of where it happened to start, and those mistakes are largely
random. Averaging cancels part of that randomness out. In code this is a single line —
`np.mean` of the five predicted probabilities.

> **Be honest about what this does.** Augmentation and ensembling add no new information.
> They reduce variance and over-fitting. The ceiling is still set by what Lead I can
> physically show, which is why inferior and posterior infarctions remain hard to detect.

### 4.1 Other training differences

| Setting | Baseline (Model 1) | This notebook |
|---|---|---|
| Maximum epochs | 60 | **80** |
| EarlyStopping patience | 10 | **12** |
| ReduceLROnPlateau patience | 4 | **5** |
| Handling class imbalance | `class_weight` argument | per-sample weights |
| Data pipeline | plain arrays | `tf.data` pipeline |

Each model is allowed to train longer than the baseline because augmentation slows
over-fitting down, so there is more to gain from extra epochs. The switch to per-sample
weights is a technical necessity, not a change of method: a `tf.data` pipeline takes weights
per sample rather than the `class_weight` argument, but the weights themselves are computed
identically.

---

## 5. How many models does this notebook create?

**Five files — plus an ensemble that is not a file at all.**

This is the single most confusing point in the whole project, so it is worth being precise.
The notebook's results table has seven rows, but only five new files are ever written to disk.

| What | Is it a file? | Where it lives |
|---|---|---|
| The 5 trained models | **Yes — 5 files** | `outputs/ensemble/model_seed42.keras` … `model_seed46.keras` |
| The ensemble | **No** | It is the average of the 5 answers, worked out fresh each time a prediction is made. |
| The baseline model | **No — it is read, not created** | `outputs/mi_cnn_model.keras`, produced by the other notebook |

> **The ensemble has no file of its own.** There is no `model_ensemble.keras` anywhere, and
> there never will be. Whenever the ensemble makes a prediction, all five models are asked
> the same question and their five probabilities are averaged. The ensemble exists only as
> that averaging step in the code. This is why the Flask application has to load all five
> files, not one.

The notebook also loads the baseline model from the first notebook in order to compare
against it. It does not create or modify it. If the baseline file is missing from Drive, the
notebook simply skips the comparison and continues.

### 5.1 Counting across both notebooks

| Notebook | Model files created | Names |
|---|---|---|
| `colab_train_eval.ipynb` | **1** | `mi_cnn_model.keras` |
| `colab_train_improved.ipynb` | **5** | `model_seed42` … `model_seed46` `.keras` |
| **Total on disk** | **6** | plus 1 ensemble that is computed, not stored |

---

## 6. Choosing the thresholds again

The ensemble produces a different spread of probabilities from a single model, so its
cut-offs have to be worked out again. They are chosen on the validation fold (fold 9) and
then applied unchanged to the test fold (fold 10).

| Operating point | Threshold | What it is for |
|---|---|---|
| Default | 0.5000 | The naive cut-off. Kept only to show why it is a poor choice in medicine. |
| Best F1 | 0.5285 | The most balanced trade-off between false alarms and misses. |
| **High recall** | **0.2974** | Catches at least 90% of real heart attacks. What a screening tool should use. |

A lower cut-off makes the system shout "MI" more readily, so it misses fewer real heart
attacks but raises more false alarms. Which trade-off is correct is a medical decision, not
a technical one.

---

## 7. Did it actually work?

| Model | Test ROC-AUC |
|---|---|
| Baseline, no augmentation (Model 1) | 0.9017 |
| **Ensemble of 5, with augmentation** | **0.9076** |
| **Difference** | **+0.0059** |

A rise of 0.0059 is small, and on a test fold of only 1,462 recordings a small rise could
easily be noise. So the notebook does not simply report the bigger number. It resamples the
test set 2,000 times and measures the difference between the two models on each resample,
producing a 95% confidence interval for the difference:

| Quantity | Value | Interpretation |
|---|---|---|
| Difference in ROC-AUC | **+0.0059** | The ensemble scores higher. |
| 95% confidence interval | **[+0.0009, +0.0110]** | The interval does not include zero. |
| **Verdict** | **Statistically significant** | The improvement is real, not chance. |

> **Why this test is the important part.** Had the interval included zero, the correct
> conclusion would have been "no significant difference" — and reporting that honestly is
> better science than claiming a win from a 0.005 rise. The notebook is written to reach
> either conclusion; it happened to reach the favourable one.

---

## 8. Where the models are saved and how to access them

### 8.1 Where they are saved

Everything is written to Google Drive, because Colab deletes its own machine when the session
ends. Each model is saved as soon as it finishes, so if Colab disconnects halfway you keep
the models already trained.

```
MyDrive/ecg_project/
    ptbxl_leadI.npz                 <- the input you upload first
    outputs/
        mi_cnn_model.keras          <- the baseline, from the OTHER notebook
        ensemble/
            model_seed42.keras      <- THE FIVE
            model_seed43.keras
            model_seed44.keras
            model_seed45.keras
            model_seed46.keras
            ensemble_config.json    <- thresholds and settings
        improvement_comparison.csv
        ensemble_test_predictions.npz
        baseline_vs_ensemble_roc.png
```

### 8.2 Opening them in Python

Because the ensemble is an averaging step rather than a file, using it means loading all five
models and averaging their answers yourself:

```python
import json, numpy as np, tensorflow as tf

config = json.load(open("ensemble/ensemble_config.json"))

# config['members'] lists the five file names
models = [tf.keras.models.load_model("ensemble/" + name)
          for name in config["members"]]

# ask all five, then average - THIS is the ensemble
probs = [float(m.predict(signal)[0][0]) for m in models]
probability = float(np.mean(probs))
```

### 8.3 How the Flask web application uses them

The application does this automatically. On start-up it checks whether
`ensemble/ensemble_config.json` exists:

- **If it is there** — all five models are loaded and every prediction is the average of five
  answers.
- **If it is missing** — the app quietly falls back to the single baseline model and says so
  in the page header, so you always know which model produced a result.

Because of that fallback, no code has to change when you add or remove the ensemble. You
simply copy the `ensemble` folder down from Drive into the project and restart the app. The
relevant code is near the top of `webapp/app.py`.

Using five models instead of one costs time: server-side prediction takes about 0.35 seconds
with the full ensemble, because each recording is passed through all five networks.

> **Important: these files are not in the project repository.** All six `.keras` files are
> excluded by `.gitignore` because they are large and are regenerated by training. Only the
> small JSON configuration files are stored. If you clone this project onto a new machine the
> `ensemble` folder will contain `ensemble_config.json` and nothing else — you must copy the
> five model files down from Google Drive or re-run this notebook.

---

## 9. Honest limitations

- **The gain is small.** It is statistically real, but +0.0059 ROC-AUC is a modest
  improvement bought with five times the training time and five times the prediction cost.
- **No new information was added.** Augmentation and ensembling reduce variance; they cannot
  reveal something Lead I does not physically contain.
- **Still trained on clean hospital recordings.** Augmentation imitates electrode noise, but
  imitation is not the same as a real ESP32 signal. Live performance will still be lower.
- **It detects patterns, not emergencies.** Only 146 of the 5,469 MI recordings — **2.7%** —
  are labelled acute (`infarction_stadium1 = "Stadium I"`); the rest are prior, healed, or of
  unrecorded stage. (61.7% have an unknown stadium; among those with a recorded stage, 7.0%
  are acute, which is what an earlier draft rounded to "about 6%".)
- **Not a medical device.** Not validated for clinical use.
