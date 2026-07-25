"""
plots.py
--------
Generates every plot the evaluation criteria and report need: ROC curve,
PR curve, confusion matrix heatmap, risk-score distribution (by true
label, so you can visually see class separation), and the alert-budget
precision/recall/FPR curve. All functions save a PNG to `outdir` and
return the file path, so the report generator (Phase 6) just collects
paths - it never touches matplotlib directly.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .precision import alert_budget_curve


def plot_roc_curve(y_true, y_score, outdir, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="Detector")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pr_curve(y_true, y_score, outdir, filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    baseline = y_true.sum() / len(y_true)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label="Detector")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm_dict, outdir, filename="confusion_matrix.png", title="Confusion Matrix"):
    cm = np.array(cm_dict["matrix"])
    labels = cm_dict["labels"]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.9), max(4, len(labels) * 0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_score_distribution(y_true, y_score, outdir, filename="score_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_score[y_true == 0], bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(y_score[y_true == 1], bins=50, alpha=0.6, label="Anomaly", density=True)
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution by True Label")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_alert_budget_curve(y_true, y_score, outdir, filename="alert_budget_curve.png"):
    df = alert_budget_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["alert_budget_fraction"] * 100, df["precision"], label="Precision")
    ax.plot(df["alert_budget_fraction"] * 100, df["recall"], label="Recall")
    ax.plot(df["alert_budget_fraction"] * 100, df["false_positive_rate"], label="False Positive Rate")
    ax.set_xlabel("Alert Budget (% of events reviewed)")
    ax.set_ylabel("Rate")
    ax.set_title("Precision / Recall / FPR vs Alert Budget")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_plots(y_true, y_score, binary_cm_dict, outdir, multiclass_cm_dict=None):
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "roc_curve": plot_roc_curve(y_true, y_score, outdir),
        "pr_curve": plot_pr_curve(y_true, y_score, outdir),
        "confusion_matrix_binary": plot_confusion_matrix(
            binary_cm_dict, outdir, "confusion_matrix_binary.png", "Binary Detection Confusion Matrix"),
        "score_distribution": plot_score_distribution(y_true, y_score, outdir),
        "alert_budget_curve": plot_alert_budget_curve(y_true, y_score, outdir),
    }
    if multiclass_cm_dict is not None:
        paths["confusion_matrix_multiclass"] = plot_confusion_matrix(
            multiclass_cm_dict, outdir, "confusion_matrix_multiclass.png", "Attack-Type Classification Confusion Matrix")
    return paths
