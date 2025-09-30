# ruff: noqa: E501
"""
Base DistilBERT evaluation for phishing detection without fine-tuning.

This module evaluates the performance of pretrained DistilBERT models on phishing
email classification tasks without any fine-tuning. Provides baseline performance
metrics for comparison with fine-tuned models and other approaches.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import logging as transformers_logging

# Add project root to path
sys.path.append("src")

from nn_phishing_detection.data_utils import TransformerDataset
from nn_phishing_detection.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
)

# Suppress transformers warnings
transformers_logging.set_verbosity_error()


def evaluate_model(model, dataloader, device, model_name="DistilBERT"):
    """
    Evaluate DistilBERT model on phishing detection dataset.

    Performs inference on the provided dataset and collects predictions,
    true labels, and class probabilities for evaluation.

    Parameters
    ----------
    model : transformers.DistilBertForSequenceClassification
        Pretrained DistilBERT model for evaluation.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing evaluation samples with tokenized inputs.
    device : torch.device
        Device for model inference (CPU/CUDA/MPS).
    model_name : str, default="DistilBERT"
        Model name for progress display.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - Ground truth labels (n_samples,)
        - Predicted labels (n_samples,)
        - Class probabilities (n_samples, n_classes)

    Notes
    -----
    Sets model to eval mode and disables gradients for efficient inference.
    Uses torch.softmax to convert logits to probabilities.
    Processes batches with progress bar for user feedback.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Evaluating {model_name}...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def load_finetuned_model(model_path, device):
    """Load fine-tuned DistilBERT model."""
    model_path = Path(model_path)

    print(f"Loading fine-tuned model from: {model_path}")

    # Try different model loading strategies
    if (model_path / "pytorch_model.bin").exists() or (
        model_path / "model.safetensors"
    ).exists():
        # Standard transformers format
        tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
    else:
        # Fallback to base model + state dict
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

        # Try different state dict locations
        possible_paths = [
            model_path / "pytorch_model.bin",
            model_path / "model.pt",
            model_path / "best_model.pt",
        ]

        loaded = False
        for path in possible_paths:
            if path.exists():
                print(f"Loading weights from: {path}")
                state_dict = torch.load(path, map_location=device)
                model.load_state_dict(state_dict)
                loaded = True
                break

        if not loaded:
            raise FileNotFoundError(f"No model weights found in {model_path}")

    return model.to(device), tokenizer


def print_comparison(base_metrics, ft_metrics=None):
    """Print detailed comparison between base and fine-tuned models."""
    print("\n" + "=" * 80)
    if ft_metrics:
        print("MODEL COMPARISON: BASE vs FINE-TUNED")
    else:
        print("BASE DISTILBERT EVALUATION RESULTS")
    print("=" * 80)

    if ft_metrics:
        metrics_to_compare = [
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro F1"),
            ("legitimate_f1", "Legitimate F1"),
            ("malicious_f1", "Malicious F1"),
            ("legitimate_precision", "Legitimate Precision"),
            ("malicious_precision", "Malicious Precision"),
            ("legitimate_recall", "Legitimate Recall"),
            ("malicious_recall", "Malicious Recall"),
        ]

        print(
            f"{'Metric':<22} {'Base':<10} {'Fine-tuned':<12} {'Improvement':<12} {'% Change':<10}"
        )
        print("-" * 72)

        for metric_key, metric_name in metrics_to_compare:
            if metric_key in base_metrics and metric_key in ft_metrics:
                base_val = base_metrics[metric_key]
                ft_val = ft_metrics[metric_key]
                improvement = ft_val - base_val
                pct_change = (improvement / base_val) * 100 if base_val > 0 else 0

                print(
                    f"{metric_name:<22} {base_val:<10.4f} {ft_val:<12.4f} "
                    f"{improvement:<+12.4f} {pct_change:<+10.2f}%"
                )
    else:
        # Just base metrics
        print(f"Accuracy: {base_metrics['accuracy']:.4f}")
        print(f"Macro F1: {base_metrics['macro_f1']:.4f}")
        if "legitimate_f1" in base_metrics:
            print(f"Legitimate F1: {base_metrics['legitimate_f1']:.4f}")
            print(f"Malicious F1: {base_metrics['malicious_f1']:.4f}")


def main():
    """
    Main evaluation function with command-line interface.

    Provides complete evaluation for base DistilBERT models including
    data loading, model inference, metrics computation, and results visualization
    with optional comparison to fine-tuned model performance.

    Notes
    -----
    Supports automatic device detection (CUDA/MPS/CPU).
    Generates outputs including confusion matrices and PR curves.
    Can compare base model performance with fine-tuned model if provided.
    Saves detailed evaluation results and performance metrics.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate base DistilBERT vs fine-tuned model on phishing detection"
    )
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default=None,
        help="Path to fine-tuned DistilBERT model for comparison",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/distilbert_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for testing)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization (use raw text)",
    )

    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")

    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    test_df = pd.read_csv(args.test_data)

    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        print(f"Limited to {args.max_samples} samples for testing")

    print(f"Test set size: {len(test_df)} samples")
    print(f"Label distribution: {test_df['label'].value_counts().to_dict()}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base DistilBERT model
    print("\nLoading base DistilBERT model...")
    base_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        problem_type="single_label_classification",
    )
    base_model = base_model.to(device)

    # Text normalization setting
    normalize = not args.no_normalize
    print(f"Text normalization: {'enabled' if normalize else 'disabled'}")

    # Evaluate base model
    base_dataset = TransformerDataset(
        test_df, base_tokenizer, args.max_length, normalize=normalize, label_key="label"
    )
    base_dataloader = DataLoader(
        base_dataset, batch_size=args.batch_size, shuffle=False
    )

    start_time = time.time()
    base_labels, base_preds, base_probs = evaluate_model(
        base_model, base_dataloader, device, "Base DistilBERT"
    )
    base_eval_time = time.time() - start_time

    # Compute base metrics
    base_metrics = compute_metrics(base_labels, base_preds)

    # Load and evaluate fine-tuned model if provided
    ft_metrics = None
    ft_labels = None
    ft_preds = None
    ft_probs = None
    ft_eval_time = 0

    if args.finetuned_model:
        ft_model, ft_tokenizer = load_finetuned_model(args.finetuned_model, device)

        ft_dataset = TransformerDataset(
            test_df,
            ft_tokenizer,
            args.max_length,
            normalize=normalize,
            label_key="label",
        )
        ft_dataloader = DataLoader(
            ft_dataset, batch_size=args.batch_size, shuffle=False
        )

        start_time = time.time()
        ft_labels, ft_preds, ft_probs = evaluate_model(
            ft_model, ft_dataloader, device, "Fine-tuned DistilBERT"
        )
        ft_eval_time = time.time() - start_time

        ft_metrics = compute_metrics(ft_labels, ft_preds)

    total_eval_time = base_eval_time + ft_eval_time
    print(f"\nTotal evaluation completed in {total_eval_time:.1f}s")

    # Print comparison results
    print_comparison(base_metrics, ft_metrics)

    # Print individual classification reports
    print("\nBASE DISTILBERT Classification Report:")
    print(
        classification_report(
            base_labels, base_preds, target_names=["Legitimate", "Malicious"], digits=4
        )
    )

    if ft_metrics:
        print("\nFINE-TUNED DISTILBERT Classification Report:")
        print(
            classification_report(
                ft_labels, ft_preds, target_names=["Legitimate", "Malicious"], digits=4
            )
        )

    # Confusion Matrices
    print("\nBASE DISTILBERT Confusion Matrix:")
    base_cm = confusion_matrix(base_labels, base_preds)
    print(base_cm)
    print("(rows=true labels, cols=predicted)")

    if ft_metrics:
        print("\nFINE-TUNED DISTILBERT Confusion Matrix:")
        ft_cm = confusion_matrix(ft_labels, ft_preds)
        print(ft_cm)
        print("(rows=true labels, cols=predicted)")

    # Error analysis
    print("\nERROR ANALYSIS:")
    test_df_results = test_df.copy()
    test_df_results["base_predicted"] = base_preds
    test_df_results["base_correct"] = base_preds == base_labels
    test_df_results["base_prob_legitimate"] = base_probs[:, 0]
    test_df_results["base_prob_malicious"] = base_probs[:, 1]

    base_errors = test_df_results[~test_df_results["base_correct"]]
    total_samples = len(test_df_results)
    base_error_pct = len(base_errors) / total_samples * 100

    base_fp = test_df_results[
        (test_df_results["label"] == 0) & (test_df_results["base_predicted"] == 1)
    ]
    base_fn = test_df_results[
        (test_df_results["label"] == 1) & (test_df_results["base_predicted"] == 0)
    ]

    print(
        f"Base model errors: {len(base_errors)} / {total_samples} ({base_error_pct:.2f}%)"
    )
    print(
        f"Base False Positives: {len(base_fp)} ({len(base_fp)/total_samples*100:.2f}%)"
    )
    print(
        f"Base False Negatives: {len(base_fn)} ({len(base_fn)/total_samples*100:.2f}%)"
    )

    if ft_metrics:
        test_df_results["ft_predicted"] = ft_preds
        test_df_results["ft_correct"] = ft_preds == ft_labels
        test_df_results["ft_prob_legitimate"] = ft_probs[:, 0]
        test_df_results["ft_prob_malicious"] = ft_probs[:, 1]
        test_df_results["improvement"] = (
            test_df_results["ft_correct"] & ~test_df_results["base_correct"]
        )
        test_df_results["degradation"] = (
            ~test_df_results["ft_correct"] & test_df_results["base_correct"]
        )

        ft_errors = test_df_results[~test_df_results["ft_correct"]]
        ft_error_pct = len(ft_errors) / total_samples * 100

        improved = test_df_results["improvement"].sum()
        degraded = test_df_results["degradation"].sum()
        net_improvement = improved - degraded

        print(
            f"Fine-tuned model errors: {len(ft_errors)} / {total_samples} ({ft_error_pct:.2f}%)"
        )
        print(f"Fine-tuning helped: {improved} cases")
        print(f"Fine-tuning hurt: {degraded} cases")
        print(f"Net improvement: {net_improvement} cases")

    # Save results
    if ft_metrics:
        test_df_results.to_csv(
            output_dir / "distilbert_comparison_results.csv", index=False
        )
    else:
        test_df_results.to_csv(
            output_dir / "base_distilbert_predictions.csv", index=False
        )

    # Save confusion matrix plots
    plot_confusion_matrix(
        base_labels,
        base_preds,
        ["Legitimate", "Malicious"],
        str(output_dir / "base_distilbert_confusion_matrix.png"),
    )

    # Save PR curves
    plot_pr_curves(
        base_labels,
        base_probs,
        ["Legitimate", "Malicious"],
        str(output_dir / "base_distilbert_pr"),
    )

    if ft_metrics:
        plot_confusion_matrix(
            ft_labels,
            ft_preds,
            ["Legitimate", "Malicious"],
            str(output_dir / "finetuned_distilbert_confusion_matrix.png"),
        )
        plot_pr_curves(
            ft_labels,
            ft_probs,
            ["Legitimate", "Malicious"],
            str(output_dir / "finetuned_distilbert_pr"),
        )

    # Save metrics
    results_summary = {
        "base_metrics": base_metrics,
        "evaluation_info": {
            "base_model": "distilbert-base-uncased",
            "test_samples": len(test_df),
            "base_evaluation_time_seconds": base_eval_time,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "text_normalization": normalize,
        },
        "base_error_analysis": {
            "total_errors": len(base_errors),
            "false_positives": len(base_fp),
            "false_negatives": len(base_fn),
            "error_percentage": base_error_pct,
        },
    }

    if ft_metrics:
        results_summary["finetuned_metrics"] = ft_metrics
        results_summary["evaluation_info"]["finetuned_model"] = args.finetuned_model
        results_summary["evaluation_info"][
            "total_evaluation_time_seconds"
        ] = total_eval_time
        # Calculate improvements for numeric metrics only
        numeric_keys = [
            "accuracy",
            "macro_f1",
            "legitimate_f1",
            "malicious_f1",
            "legitimate_precision",
            "malicious_precision",
            "legitimate_recall",
            "malicious_recall",
        ]
        results_summary["improvements"] = {
            k: ft_metrics[k] - base_metrics[k]
            for k in numeric_keys
            if k in base_metrics and k in ft_metrics
        }
        results_summary["comparison"] = {
            "cases_improved": int(improved),
            "cases_degraded": int(degraded),
            "net_improvement": int(net_improvement),
        }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Print file summary
    print(f"\nResults saved to: {output_dir}/")
    if ft_metrics:
        print("- distilbert_comparison_results.csv (detailed predictions)")
        print("- base_distilbert_confusion_matrix.png")
        print("- finetuned_distilbert_confusion_matrix.png")
        print("- base_distilbert_pr_curves.png")
        print("- finetuned_distilbert_pr_curves.png")
    else:
        print("- base_distilbert_predictions.csv (detailed predictions)")
        print("- base_distilbert_confusion_matrix.png")
        print("- base_distilbert_pr_curves.png")
    print("- evaluation_metrics.json (metrics summary)")

    if ft_metrics:
        acc_improvement = ft_metrics["accuracy"] - base_metrics["accuracy"]
        if acc_improvement > 0.2:
            print(" MASSIVE IMPROVEMENT: Fine-tuning dramatically improved performance")
        elif acc_improvement > 0.1:
            print(" MAJOR IMPROVEMENT: Fine-tuning significantly improved performance")
        elif acc_improvement > 0.05:
            print(" MODERATE IMPROVEMENT: Fine-tuning helped performance")
        elif acc_improvement > 0.01:
            print(" MINOR IMPROVEMENT: Fine-tuning slightly improved performance")
        else:
            print(" NO IMPROVEMENT: Fine-tuning didn't help much")

        print(
            f"Accuracy: {base_metrics['accuracy']:.1%} → {ft_metrics['accuracy']:.1%} "
            f"(+{acc_improvement:.1%})"
        )
        print(
            f"F1 Score: {base_metrics['macro_f1']:.1%} → {ft_metrics['macro_f1']:.1%} "
            f"(+{ft_metrics['macro_f1'] - base_metrics['macro_f1']:.1%})"
        )
    else:
        if base_metrics["accuracy"] > 0.9:
            print(
                " EXCELLENT: Base DistilBERT shows strong performance without fine-tuning"
            )
        elif base_metrics["accuracy"] > 0.8:
            print(
                " GOOD: Base DistilBERT shows decent performance, fine-tuning likely beneficial"
            )
        elif base_metrics["accuracy"] > 0.7:
            print(
                " MODERATE: Base DistilBERT shows some capability, fine-tuning needed"
            )
        else:
            print(
                " POOR: Base DistilBERT struggles with this task, fine-tuning essential"
            )

        print(
            f"Base model achieves {base_metrics['accuracy']:.1%} accuracy on phishing detection"
        )


if __name__ == "__main__":
    main()
