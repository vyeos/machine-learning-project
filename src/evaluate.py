"""
evaluate.py
-----------
Classification metrics and evaluation plots for the intrusion detection task.

Metric rationale (documented in report chapter 7):
* Accuracy alone is misleading under class imbalance and hides which side
  (benign vs malicious) the errors come from.
* Recall  = share of malicious flows actually flagged  (low recall = missed attacks).
* Precision = share of flagged flows that really were malicious
              (low precision = analysts waste time on false alarms).
* F1 balances the two; ROC-AUC and Average Precision summarise behaviour across
  operating thresholds; specificity/FPR/FNR describe the confusion matrix in
  security-friendly language (false alarms per benign flow, missed attacks).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibrationDisplay

from src.data_loading import FIGURES_DIR, RESULTS_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Style + figure helpers
# --------------------------------------------------------------------------- #
def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })


def save_fig(fig, name: str) -> Path:
    """Save into results/figures and return the path (caption used in report)."""
    out = FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: np.ndarray | None = None,
                           threshold: float = 0.5,
                          ) -> dict:
    """
    Full binary-classification report for label 1 = malicious (positive class).
    `threshold` only changes y_pred if y_proba is provided; pass the final
    predictions already produced at the operating threshold otherwise.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "n": len(y_true),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else np.nan,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else np.nan,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["average_precision"] = average_precision_score(y_true, y_proba)
        metrics["brier_score"] = brier_score_loss(y_true, y_proba)
    return metrics


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def confusion_plot(y_true, y_pred, title: str, fname: str, normalize: bool = False):
    cmap = matplotlib.colormaps["Blues"] if normalize else matplotlib.colormaps["Oranges"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pretty = np.array([["Benign", "Malicious"], ["Benign", "Malicious"]])
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".3f"
    else:
        fmt = "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=pretty[0], yticklabels=pretty[1], cbar=normalize, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    return save_fig(fig, fname)


def roc_pr_plots(y_true, proba_by_model: dict[str, np.ndarray], fname_roc: str, fname_pr: str):
    """ROC + precision-recall curves for several models on the same test set."""
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False-positive rate (benign flows wrongly flagged)")
    ax.set_ylabel("True-positive rate (share of attacks caught)")
    ax.set_title("ROC curves - malicious class")
    ax.legend(fontsize=9)
    save_fig(fig, fname_roc)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for name, proba in proba_by_model.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})")
    base = float(np.asarray(y_true).mean())
    ax.axhline(base, color="k", ls="--", lw=1, label=f"Chance (AP={base:.3f})")
    ax.set_xlabel("Recall (share of attacks caught)")
    ax.set_ylabel("Precision (flagged flows that are attacks)")
    ax.set_title("Precision-recall curves - malicious class")
    ax.legend(fontsize=9, loc="lower left")
    save_fig(fig, fname_pr)


def comparison_chart(df_comp: pd.DataFrame, fname: str,
                     metric_cols=("test_f1", "test_recall", "test_precision", "test_roc_auc")):
    """Grouped bar chart of main test metrics per model."""
    d = df_comp.set_index("model")[list(metric_cols)].astype(float)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(d)), 4.8))
    d.plot(kind="bar", ax=ax, width=0.78, edgecolor="white")
    ax.set_ylim(min(0.5, d.min().min() - 0.02), 1.02)
    ax.set_ylabel("Score on held-out test set")
    ax.set_title("Model comparison on the untouched test set")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=9, ncol=len(metric_cols))
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
    return save_fig(fig, fname)


def calibration_plot(y_true, proba: np.ndarray, model_name: str, fname: str, n_bins: int = 8):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    disp = CalibrationDisplay.from_predictions(
        y_true, proba, n_bins=n_bins, ax=ax, strategy="uniform",
        name=model_name,
    )
    disp.ax_.set_title(f"Calibration check - {model_name}")
    save_fig(fig, fname)


def threshold_curve(y_true, proba: np.ndarray, fname: str,
                    grid: np.ndarray | None = None) -> pd.DataFrame:
    """Precision/recall/F1/FPR as a function of the alert threshold (validation only)."""
    y_true = np.asarray(y_true).astype(int)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in grid:
        m = classification_metrics(y_true, (proba >= t).astype(int), threshold=float(t))
        rows.append(m)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(df["threshold"], df["recall"], "-o", ms=3, label="Recall (attacks caught)")
    ax.plot(df["threshold"], df["precision"], "-s", ms=3, label="Precision (alert quality)")
    ax.plot(df["threshold"], df["f1"], "-^", ms=3, label="F1 score")
    ax.plot(df["threshold"], df["false_positive_rate"], "-d", ms=3, label="False-positive rate")
    ax.set_xlabel("Alert threshold (probability of 'malicious')")
    ax.set_ylabel("Metric")
    ax.set_title("Operating-threshold trade-off (validation data only)")
    ax.legend(fontsize=9)
    save_fig(fig, fname)
    return df
