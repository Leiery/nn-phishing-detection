# ruff: noqa: E501
"""
DistilBERT fine-tuning script for phishing email classification.

This module provides a complete training pipeline for DistilBERT transformer models
designed to match CNN training setups for fair comparison. Includes data preprocessing,
model fine-tuning, evaluation metrics, and MLflow experiment tracking.

Author: Elizaveta Dovedova <dovedova@cl.uni-heidelberg.de>, David Schatz <schatz@cl.uni-heidelberg.de>
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append("src")
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from nn_phishing_detection.data_utils import TransformerDataset
from nn_phishing_detection.metrics import (
    plot_confusion_matrix,
    plot_pr_curves,
    plot_training_progress,
)
from nn_phishing_detection.tracking import log_metrics, start_run

# Set device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")


def compute_metrics(eval_pred):
    """
    Compute classification metrics matching CNN model evaluation.

    Calculates metrics including accuracy, F1 scores, precision,
    and recall for both classes to enable direct comparison with CNN models.

    Parameters
    ----------
    eval_pred : tuple
        Tuple containing (predictions, labels) where:
        - predictions: Model logits of shape (n_samples, n_classes)
        - labels: Ground truth labels of shape (n_samples,)

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy (float): Overall classification accuracy
        - macro_f1 (float): Macro-averaged F1 score
        - macro_precision (float): Macro-averaged precision
        - macro_recall (float): Macro-averaged recall
        - legitimate_f1 (float): F1 score for legitimate emails
        - malicious_f1 (float): F1 score for malicious emails
        - legitimate_precision (float): Precision for legitimate emails
        - malicious_precision (float): Precision for malicious emails
        - legitimate_recall (float): Recall for legitimate emails
        - malicious_recall (float): Recall for malicious emails

    Notes
    -----
    Uses argmax to convert logits to predictions and computes both macro
    and per-class metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    accuracy = accuracy_score(labels, predictions)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(labels, predictions, average=None)
    )

    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "legitimate_f1": f1_per_class[0],
        "malicious_f1": f1_per_class[1],
        "legitimate_precision": precision_per_class[0],
        "malicious_precision": precision_per_class[1],
        "legitimate_recall": recall_per_class[0],
        "malicious_recall": recall_per_class[1],
    }


def evaluate(model, dataloader, device, return_probs=False):
    """
    Evaluate DistilBERT model on validation or test dataset.

    Performs inference on the provided dataset and computes loss, predictions,
    and optionally class probabilities for evaluation.

    Parameters
    ----------
    model : transformers.AutoModelForSequenceClassification
        Fine-tuned DistilBERT model for evaluation.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing evaluation samples with tokenized inputs.
    device : torch.device
        Device for model inference (CPU/CUDA/MPS).
    return_probs : bool, default=False
        Whether to return class probabilities for PR curve analysis.

    Returns
    -------
    dict or tuple
        If return_probs=False: Dictionary of evaluation metrics.
        If return_probs=True: Tuple of (metrics_dict, predictions_array, labels_array, probabilities_array).

    Notes
    -----
    Sets model to eval mode and disables gradients for efficient inference.
    Uses model's built-in loss computation with CrossEntropyLoss.
    Applies softmax to logits for probability computation.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            total_loss += outputs.loss.item()
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    metrics = compute_metrics((np.eye(2)[all_predictions], all_labels))
    metrics["loss"] = total_loss / len(dataloader)

    if return_probs:
        return (
            metrics,
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probs),
        )
    return metrics, np.array(all_predictions), np.array(all_labels)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train DistilBERT model for one complete epoch.

    Performs forward pass, loss computation, backpropagation, and parameter
    updates for all batches in the training dataset with gradient clipping.

    Parameters
    ----------
    model : transformers.AutoModelForSequenceClassification
        DistilBERT model to train.
    dataloader : torch.utils.data.DataLoader
        Training data loader with tokenized inputs.
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates (typically AdamW).
    scheduler : transformers.SchedulerType
        Learning rate scheduler (typically linear warmup).
    device : torch.device
        Device for training (CPU/CUDA/MPS).

    Returns
    -------
    float
        Average training loss for the epoch.

    Notes
    -----
    Uses gradient clipping (max_norm=1.0).
    Updates learning rate scheduler after each batch.
    Model automatically computes CrossEntropyLoss when labels are provided.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def main():
    """
    Main training function with command-line interface.

    Provides complete DistilBERT fine-tuning pipeline including data loading,
    model initialization, training with early stopping, evaluation, and results
    visualization with MLflow experiment tracking.

    Notes
    -----
    Supports automatic device detection (CUDA/MPS/CPU) and mixed precision training.
    Implements early stopping based on validation F1 score with configurable patience.
    Generates visualizations including confusion matrices and PR curves.
    Saves best model, training history, and detailed evaluation results.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for phishing detection"
    )

    # Data arguments
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--val-data", type=str, required=True, help="Path to validation CSV file"
    )
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to test CSV file"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3, same as CNN)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=500, help="Warmup steps (default: 500)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (use raw text)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/distilbert_phishing",
        help="Output directory (default: runs/distilbert_phishing)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42, same as CNN)"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (optional)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    print("=" * 50)
    print("DISTILBERT FINE-TUNING")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    test_df = pd.read_csv(args.test_data)

    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    # Initialize tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, problem_type="single_label_classification"
    )
    model.to(device)

    # Create datasets with same normalization as CNN
    normalize = not args.no_normalize
    if normalize:
        print("\nUsing text normalization (same as CNN model)")
        print("This includes: URL/email/number replacement, lowercasing, etc.")
    else:
        print("\nUsing raw text (no normalization)")

    train_dataset = TransformerDataset(
        train_df, tokenizer, args.max_length, normalize=normalize
    )
    val_dataset = TransformerDataset(
        val_df, tokenizer, args.max_length, normalize=normalize
    )
    test_dataset = TransformerDataset(
        test_df, tokenizer, args.max_length, normalize=normalize
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Training loop with early stopping
    print("\nStarting training...")
    best_val_f1 = 0
    patience_counter = 0
    training_history = []

    # Start MLflow run
    # Ensure no active run exists
    mlflow.end_run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"distilbert_{timestamp}_e{args.epochs}_bs{args.batch_size}_lr{args.lr}_p{args.patience}"

    params = {
        "model": args.model_name,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "normalize": not args.no_normalize,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "patience": args.patience,
    }

    with start_run(run_name=run_name, params=params) as run:

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

            # Evaluate on validation set
            val_metrics, _, _ = evaluate(model, val_loader, device)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1: {val_metrics['macro_f1']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Log metrics to MLflow
            log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_legitimate_f1": val_metrics["legitimate_f1"],
                    "val_malicious_f1": val_metrics["malicious_f1"],
                    "val_legitimate_precision": val_metrics["legitimate_precision"],
                    "val_malicious_precision": val_metrics["malicious_precision"],
                    "val_legitimate_recall": val_metrics["legitimate_recall"],
                    "val_malicious_recall": val_metrics["malicious_recall"],
                },
                step=epoch,
            )

            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_legitimate_f1": val_metrics["legitimate_f1"],
                    "val_malicious_f1": val_metrics["malicious_f1"],
                    "val_legitimate_precision": val_metrics["legitimate_precision"],
                    "val_malicious_precision": val_metrics["malicious_precision"],
                    "val_legitimate_recall": val_metrics["legitimate_recall"],
                    "val_malicious_recall": val_metrics["malicious_recall"],
                }
            )

            # Early stopping
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                patience_counter = 0

                # Save best model
                best_model_path = output_dir / "best_model"
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                print(f" New best model saved (F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Load best model for final evaluation
        print("\nLoading best model for final evaluation...")
        model = AutoModelForSequenceClassification.from_pretrained(
            output_dir / "best_model"
        )
        model.to(device)

        # Final evaluation on test set with probabilities for PR curves
        print("\nEvaluating on test set...")
        test_metrics, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, device, return_probs=True
        )

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
        print("\nPer-class metrics:")
        print(
            f"  Legitimate - F1: {test_metrics['legitimate_f1']:.4f}, Precision: {test_metrics['legitimate_precision']:.4f}, Recall: {test_metrics['legitimate_recall']:.4f}"
        )
        print(
            f"  Malicious  - F1: {test_metrics['malicious_f1']:.4f}, Precision: {test_metrics['malicious_precision']:.4f}, Recall: {test_metrics['malicious_recall']:.4f}"
        )

        # Log final test metrics to MLflow
        log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
                "test_legitimate_f1": test_metrics["legitimate_f1"],
                "test_malicious_f1": test_metrics["malicious_f1"],
                "test_legitimate_precision": test_metrics["legitimate_precision"],
                "test_malicious_precision": test_metrics["malicious_precision"],
                "test_legitimate_recall": test_metrics["legitimate_recall"],
                "test_malicious_recall": test_metrics["malicious_recall"],
            }
        )

        # Save results
        results = {
            "model": args.model_name,
            "test_metrics": test_metrics,
            "best_val_f1": best_val_f1,
            "training_history": training_history,
            "args": vars(args),
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_dir}/results.json")

        # Save predictions
        test_df["predicted"] = test_preds
        test_df.to_csv(output_dir / "test_predictions.csv", index=False)
        print(f"Predictions saved to {output_dir}/test_predictions.csv")

        # Generate and save visualizations
        print("\nGenerating visualizations...")

        # Confusion matrix
        plot_confusion_matrix(
            test_labels,
            test_preds,
            labels=["Legitimate", "Malicious"],
            outpath=str(output_dir / "confusion_matrix.png"),
        )
        print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

        # PR curves (macro and per-class)
        plot_pr_curves(
            test_labels,
            test_probs,
            labels=["Legitimate", "Malicious"],
            outpath_prefix=str(output_dir / "pr_curves"),
        )
        print(f"PR curves saved to {output_dir}/pr_curves_*.png")

        # Training progress plots
        train_losses = [h["train_loss"] for h in training_history]
        val_losses = [h["val_loss"] for h in training_history]
        val_f1s = [h["val_f1"] for h in training_history]

        plot_training_progress(
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_f1s,
            test_loss=test_metrics["loss"],
            outpath_prefix=str(output_dir / "training_progress"),
        )
        print(f"Training progress plots saved to {output_dir}/training_progress_*.png")

        # All artifacts saved to output directory
        print(f"\nAll files saved to: {output_dir}/")
        print(f"  - Model: {output_dir}/best_model/")
        print(f"  - Results: {output_dir}/results.json")
        print(f"  - Predictions: {output_dir}/test_predictions.csv")
        print(f"\nMLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
