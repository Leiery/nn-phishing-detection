"""
Metrics and visualization utilities for phishing detection models.

This module provides evaluation metrics, confusion matrix visualization,
precision-recall curves, and training progress plots for binary classification tasks
in phishing detection. Supports both neural network and transformer model evaluation.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics for binary phishing detection.

    Calculates accuracy, macro F1, and per-class precision, recall, and F1 scores
    for both legitimate (class 0) and malicious (class 1) email classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0=legitimate, 1=malicious).
    y_pred : np.ndarray
        Predicted binary labels (0=legitimate, 1=malicious).

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy (float): Overall classification accuracy
        - macro_f1 (float): Macro-averaged F1 score
        - legitimate_f1 (float): F1 score for legitimate emails
        - malicious_f1 (float): F1 score for malicious emails
        - legitimate_precision (float): Precision for legitimate emails
        - malicious_precision (float): Precision for malicious emails
        - legitimate_recall (float): Recall for legitimate emails
        - malicious_recall (float): Recall for malicious emails
        - labels (list): Class label names

    Notes
    -----
    Uses zero_division=0 to handle edge cases where a class has no predictions.
    All metrics are converted to float for JSON serialization compatibility.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_legitimate = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_malicious = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    precision_legitimate = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision_malicious = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    recall_legitimate = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_malicious = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "legitimate_f1": float(f1_legitimate),
        "malicious_f1": float(f1_malicious),
        "legitimate_precision": float(precision_legitimate),
        "malicious_precision": float(precision_malicious),
        "legitimate_recall": float(recall_legitimate),
        "malicious_recall": float(recall_malicious),
        "labels": ["legitimate", "malicious"],
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], outpath: str
) -> None:
    """
    Generate and save a confusion matrix heatmap visualization.

    Creates a seaborn heatmap showing the confusion matrix with annotations
    for binary classification results. Saves the plot as a high-resolution PNG.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    labels : list[str]
        Class label names for axis labels (e.g., ['Legitimate', 'Malicious']).
    outpath : str
        Output file path for saving the confusion matrix plot.

    Returns
    -------
    None
        Saves plot to file and prints confirmation message.

    Notes
    -----
    Uses 'Blues' colormap and saves at 150 DPI with tight bounding box.
    Automatically closes the plot to free memory.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to {outpath}")


def plot_pr_curves(
    y_true: np.ndarray, y_proba: np.ndarray, labels: list[str], outpath_prefix: str
) -> None:
    """
    Generate precision-recall curves for binary phishing classification.

    Creates two plots: a macro PR curve showing overall performance and
    per-class PR curves for both legitimate and malicious email detection.
    Includes average precision (AP) scores and baseline comparisons.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0=legitimate, 1=malicious).
    y_proba : np.ndarray
        Predicted probabilities. Can be 1D array of positive class probabilities
        or 2D array of shape (n_samples, 2) with class probabilities.
    labels : list[str]
        Class label names for plot titles.
    outpath_prefix : str
        Prefix for output file paths. Creates two files:
        {prefix}_macro.png and {prefix}_per_class.png

    Returns
    -------
    None
        Saves plots to files and prints confirmation message.

    Notes
    -----
    For 2D probability arrays, extracts positive class probabilities from column 1.
    Per-class plots show separate PR curves for legitimate and malicious detection.
    """

    # For binary classification, y_proba should be shape (n_samples, 2)
    # If 1D: use directly, if 2D: extract malicious probabilities (column 1)
    y_proba_pos = y_proba if len(y_proba.shape) == 1 else y_proba[:, 1]

    # Compute PR curve for malicious detection
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba_pos)
    ap = average_precision_score(y_true, y_proba_pos)

    # Plot overall PR curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, lw=2, label=f"Malicious (AP = {ap:.3f})")

    # Add baseline
    no_skill = (y_true == 1).sum() / len(y_true)
    plt.axhline(y=no_skill, color="gray", linestyle="--", label="No Skill")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = f"{outpath_prefix}_macro.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot per-class PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Legitimate class (negative class)
    y_true_legitimate = 1 - y_true  # Flip labels
    y_proba_legitimate = 1 - y_proba_pos  # Probability of legitimate
    precision_legitimate, recall_legitimate, _ = precision_recall_curve(
        y_true_legitimate, y_proba_legitimate
    )
    ap_legitimate = average_precision_score(y_true_legitimate, y_proba_legitimate)

    axes[0].plot(recall_legitimate, precision_legitimate, lw=2, color="blue")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"PR Curve - Legitimate (AP = {ap_legitimate:.3f})")
    axes[0].grid(True, alpha=0.3)

    # Malicious class
    axes[1].plot(recall, precision, lw=2, color="orange")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR Curve - Malicious (AP = {ap:.3f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{outpath_prefix}_per_class.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"PR curves saved to {outpath_prefix}_*.png")


def save_metrics_json(metrics: dict, filepath: str) -> None:
    """
    Save evaluation metrics dictionary to a JSON file.

    Serializes the metrics dictionary with proper indentation for readability.
    Used for storing experimental results and model performance records.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics (from compute_metrics()).
    filepath : str
        Output file path for the JSON file.

    Returns
    -------
    None
        Saves JSON file and prints confirmation message.

    Notes
    -----
    Uses 2-space indentation for readable JSON formatting.
    All numeric values should be Python floats for proper serialization.
    """
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")


def plot_training_progress(
    train_losses: list,
    val_losses: list,
    val_metrics: list = None,
    test_loss: float = None,
    outpath_prefix: str = "training_progress",
) -> None:
    """
    Generate training progress visualization plots.

    Creates training progress plots showing loss evolution and
    validation metrics over epochs. Generates both combined and individual plots
    for detailed analysis of model training dynamics.

    Parameters
    ----------
    train_losses : list
        Training loss values for each epoch.
    val_losses : list
        Validation loss values for each epoch.
    val_metrics : list, optional
        Validation macro F1 scores for each epoch. If None, metrics subplot omitted.
    test_loss : float, optional
        Final test loss value to display as horizontal reference line.
    outpath_prefix : str, default="training_progress"
        Prefix for output file paths. Creates:
        {prefix}_combined.png and {prefix}_individual.png

    Returns
    -------
    None
        Saves plot files and prints confirmation messages.

    Notes
    -----
    Combined plot shows training/validation losses and metrics in subplots.
    Individual plot separates train loss, validation loss, and validation F1.
    All plots use high-resolution output with tight bounding boxes.
    """
    epochs = list(range(1, len(train_losses) + 1))

    # Combined loss plot
    plt.figure(figsize=(12, 8))

    # Top subplot: Combined losses
    plt.subplot(2, 1, 1)
    plt.plot(
        epochs,
        train_losses,
        "b-",
        label="Train Loss",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        epochs,
        val_losses,
        "r-",
        label="Val Loss",
        linewidth=2,
        marker="s",
        markersize=4,
    )

    if test_loss is not None:
        plt.axhline(
            y=test_loss,
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Test Loss ({test_loss:.4f})",
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress: Loss Evolution")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Bottom subplot: Validation metrics (if provided)
    if val_metrics:
        plt.subplot(2, 1, 2)
        plt.plot(
            epochs,
            val_metrics,
            "purple",
            label="Val Macro F1",
            linewidth=2,
            marker="^",
            markersize=4,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1 Score")
        plt.title("Validation Performance")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = f"{outpath_prefix}_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined training progress plot saved to {combined_path}")

    # Individual plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train loss only
    axes[0].plot(epochs, train_losses, "b-", linewidth=2, marker="o", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation loss only
    axes[1].plot(epochs, val_losses, "r-", linewidth=2, marker="s", markersize=4)
    if test_loss is not None:
        axes[1].axhline(
            y=test_loss,
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Test Loss ({test_loss:.4f})",
            linewidth=2,
        )
        axes[1].legend(loc="best")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].grid(True, alpha=0.3)

    # Validation F1 (if provided)
    if val_metrics:
        axes[2].plot(
            epochs, val_metrics, "purple", linewidth=2, marker="^", markersize=4
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1 Score")
        axes[2].set_title("Validation Macro F1")
        axes[2].grid(True, alpha=0.3)
    else:
        # Hide third subplot if no metrics
        axes[2].set_visible(False)

    plt.tight_layout()
    individual_path = f"{outpath_prefix}_individual.png"
    plt.savefig(individual_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Individual training progress plots saved to {individual_path}")


def print_metrics_summary(metrics: dict) -> None:
    """
    Print a formatted summary of classification metrics to console.

    Displays key performance metrics in a readable table format with
    proper alignment and separators for easy interpretation.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics with keys:
        'accuracy', 'macro_f1', 'legitimate_f1', 'malicious_f1'.

    Returns
    -------
    None
        Prints formatted metrics summary to stdout.

    Notes
    -----
    Assumes metrics dictionary contains the standard keys from compute_metrics().
    All values are formatted to 4 decimal places for consistency.
    """
    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Legitimate F1: {metrics['legitimate_f1']:.4f}")
    print(f"Malicious F1:  {metrics['malicious_f1']:.4f}")
    print("=" * 50)
