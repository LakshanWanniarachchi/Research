"""
EVIDENCE CELL  -  confusion matrices, training curves, and a metrics.txt
==========================================================================
Paste this as a NEW CELL right after cell 21 (the ROC curve) and before cell 22
("Save the ensemble for the Flask app"), then run it in the SAME session that
just finished training - it reads variables already in memory (histories,
test_probs_ens, table, THR_F1, THR_REC, baseline_probs, d/lo/hi, ...) rather
than reloading or retraining anything.

Why this cell exists
---------------------
The notebook already PROVES the ensemble is better (the comparison table and
the bootstrap significance test), but does not yet save three things the
baseline notebook does and a thesis needs as figures/appendix material:

  1. A confusion matrix at each operating point - not just the "Missed MI"
     count buried in a table column.
  2. The 5 models' training curves, so over-fitting can be shown or ruled out.
  3. A metrics.txt in the same layout as the baseline's and
     system/test_results.txt, so all three write-ups can be read side by side
     without re-deriving anything.

Everything here is read-only over variables you already have - nothing is
retrained, and nothing already saved by cells 14/18/20/21 is touched.
"""

from sklearn.metrics import classification_report   # not imported by cell 18

# ---------------------------------------------------------------------------
# 1. Confusion matrices for the ensemble, at all three operating points
# ---------------------------------------------------------------------------
def plot_cm(ax, y_true, probs, threshold, title):
    cm = confusion_matrix(y_true, (probs >= threshold).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "MI"]); ax.set_yticklabels(["Normal", "MI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Truth")
    ax.set_title(title, fontsize=10)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
plot_cm(axes[0], y_test, test_probs_ens, 0.5,     "Default threshold 0.50")
plot_cm(axes[1], y_test, test_probs_ens, THR_F1,  f"Best-F1 threshold {THR_F1:.3f}")
plot_cm(axes[2], y_test, test_probs_ens, THR_REC, f"High-recall threshold {THR_REC:.3f}")
plt.suptitle("Ensemble of 5 - confusion matrices, test fold (fold 10)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_ensemble.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Bottom-left cell of each matrix = missed heart attacks - the one to keep small.\n")


# ---------------------------------------------------------------------------
# 2. Training curves - all 5 seeds, loss and AUC per epoch
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(SEEDS)))

for seed, h, c in zip(SEEDS, histories, colors):
    epochs = range(1, len(h["loss"]) + 1)
    axes[0].plot(epochs, h["loss"],     "--", color=c, alpha=0.5, lw=1)
    axes[0].plot(epochs, h["val_loss"], "-",  color=c, alpha=0.9, lw=1.6, label=f"seed {seed}")
    axes[1].plot(epochs, h["auc"],      "--", color=c, alpha=0.5, lw=1)
    axes[1].plot(epochs, h["val_auc"],  "-",  color=c, alpha=0.9, lw=1.6, label=f"seed {seed}")

axes[0].set_title("Loss (dashed = train, solid = validation)")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[1].set_title("AUC (dashed = train, solid = validation)")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("AUC")
for ax in axes:
    ax.legend(fontsize=8, title="seed", loc="best"); ax.grid(alpha=0.3)
plt.suptitle("Ensemble training curves, all 5 seeds - "
            "a wide train/val gap would flag over-fitting")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves_ensemble.png"), dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 3. The ensemble at ALL THREE operating points
# ---------------------------------------------------------------------------
# The table you already have only shows each model's HIGH-RECALL row. Add
# default and best-F1 for the ensemble specifically, so it can be read the
# same way the single baseline model was in notebook 1.
def evaluate(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Threshold":    round(float(threshold), 4),
        "Accuracy":     accuracy_score(y_true, pred),
        "Precision":    precision_score(y_true, pred, zero_division=0),
        "Recall":       recall_score(y_true, pred),
        "Specificity":  tn / (tn + fp),
        "F1":           f1_score(y_true, pred),
        "ROC-AUC":      roc_auc_score(y_true, probs),
        "PR-AUC":       average_precision_score(y_true, probs),
        "Missed MI":    int(fn),
    }

ensemble_by_threshold = pd.DataFrame({
    "Default (0.50)":         evaluate(y_test, test_probs_ens, 0.5),
    "Best-F1 (from val)":     evaluate(y_test, test_probs_ens, THR_F1),
    "High-recall (from val)": evaluate(y_test, test_probs_ens, THR_REC),
}).T
print(ensemble_by_threshold.round(4))
ensemble_by_threshold.to_csv(os.path.join(OUT_DIR, "metrics_table_ensemble.csv"))


# ---------------------------------------------------------------------------
# 4. Write everything to one metrics.txt, the file the thesis quotes from
# ---------------------------------------------------------------------------
lines = []
lines.append("ENSEMBLE OF 5 - FULL EVALUATION  (PTB-XL fold 10, Lead I, Normal vs MI)")
lines.append("=" * 70)
lines.append("")
lines.append("-- Ensemble at all three operating points --")
lines.append(ensemble_by_threshold.round(4).to_string())
lines.append("")
lines.append("-- Baseline vs each individual seed vs the ensemble (high-recall each) --")
lines.append(table.round(4).to_string())
lines.append("")

if baseline_probs is not None:
    lines.append("-- Is the ensemble improvement real? (2,000-resample bootstrap on ROC-AUC) --")
    lines.append(f"Baseline ROC-AUC : {roc_auc_score(y_test, baseline_probs):.4f}")
    lines.append(f"Ensemble ROC-AUC : {roc_auc_score(y_test, test_probs_ens):.4f}")
    lines.append(f"Difference       : {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    lines.append("VERDICT: " + ("statistically significant improvement" if lo > 0
                                else "not statistically significant on this test fold"))
    lines.append("")

lines.append("-- Classification report, ensemble at best-F1 threshold --")
lines.append(classification_report(y_test, (test_probs_ens >= THR_F1).astype(int),
                                   target_names=["Normal", "MI"], digits=4))

metrics_txt_path = os.path.join(OUT_DIR, "metrics_ensemble.txt")
with open(metrics_txt_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nWritten: {metrics_txt_path}")
print("\nNew files in outputs/:")
for fname in sorted(os.listdir(OUT_DIR)):
    if os.path.isfile(os.path.join(OUT_DIR, fname)):
        mb = os.path.getsize(os.path.join(OUT_DIR, fname)) / (1024 * 1024)
        print(f"  {fname:34s} {mb:6.2f} MB")
