"""
Model evaluation script for trained phishing detection neural networks.

This module provides evaluation capabilities for trained WordCNNBiGRU models,
including model loading, vocabulary handling, test data recreation, and detailed
performance analysis with error breakdown and comparison with training metrics.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append("src")

from nn_phishing_detection.data_utils import load_data, remove_duplicates, split_data
from nn_phishing_detection.metrics import compute_metrics, plot_confusion_matrix
from nn_phishing_detection.models.word_cnn_bigru import WordCNNBiGRU
from nn_phishing_detection.training.train_word_bigru import (
    WordPhishingDataset,
    collate_fn,
)


def evaluate_model(model, dataloader, device):
    """
    Evaluate neural network model on test dataset.

    Performs inference on the provided dataloader and collects predictions,
    true labels, and class probabilities for evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model (WordCNNBiGRU).
    dataloader : torch.utils.data.DataLoader
        DataLoader containing test samples.
    device : torch.device
        Device for model inference (CPU/CUDA/MPS).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - Ground truth labels (n_samples,)
        - Predicted labels (n_samples,)
        - Class probabilities (n_samples, n_classes)

    Notes
    -----
    Sets model to eval mode and disables gradients for inference.
    Uses torch.softmax to convert logits to probabilities.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for texts, labels, lengths in tqdm(dataloader, desc="Evaluating"):
            texts = texts.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(texts, lengths)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def load_model_and_config(model_dir, device):
    """
    Load trained model with configuration from checkpoint or fallback files.

    Attempts to load model configuration and weights from checkpoint first,
    falling back to individual JSON/pickle files if checkpoint unavailable.
    Also loads training metrics for comparison purposes.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing model files (checkpoint.pt, best_model.pt, etc.).
    device : torch.device
        Target device for model loading.

    Returns
    -------
    tuple[torch.nn.Module, dict, dict]
        Tuple containing:
        - Loaded model instance moved to device
        - Training metrics dictionary (if available)
        - Training arguments dictionary (if available)

    Raises
    ------
    FileNotFoundError
        If required model files (vocabulary, weights) are not found.

    Notes
    -----
    Prefers checkpoint.pt for configuration, falls back to vocab.json.
    Loads best_model.pt weights if available, otherwise uses checkpoint weights.
    Handles both new checkpoint format and legacy individual file format.
    """
    model_dir = Path(model_dir)

    # Try to load training metrics from metrics.json
    metrics_path = model_dir / "metrics.json"
    training_metrics = {}
    if metrics_path.exists():
        print(f"Loading training metrics from: {metrics_path}")
        with open(metrics_path) as f:
            training_metrics = json.load(f)

    # Try to load from checkpoint for config
    checkpoint_path = model_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        print(f"Loading config from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = checkpoint["model_config"]
        training_args = checkpoint.get("args", {})

        # Debug: Print loaded config
        print(f"Loaded config: {config}")
        print(f"Training args: {training_args}")

        # Create model with config from checkpoint
        model = WordCNNBiGRU(**config)

        # Load weights from best model
        best_model_path = model_dir / "best_model.pt"
        if best_model_path.exists():
            print(f"Loading best model weights from: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print("Best model not found, using checkpoint weights")
            model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(device), training_metrics, training_args

    # Fallback to individual files
    print("Checkpoint not found, loading individual files...")

    # Load vocab to get vocab_size
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with open(vocab_path) as f:
        vocab_data = json.load(f)

    model = WordCNNBiGRU(
        vocab_size=vocab_data["vocab_size"],
        embedding_dim=300,
        num_classes=2,
        cnn_channels=300,
        kernel_size=3,
        conv_layers=3,
        gru_hidden=128,
        dropout=0.313,
        pretrained_embeddings=None,
    )

    # Load weights
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        model_path = model_dir / "model.pt"

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model.to(device), {}, {}


def load_vocabulary(model_dir):
    """
    Load vocabulary object from saved pickle file.

    Loads the WordVocab object that was saved during training for text tokenization 
    and encoding during evaluation.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing vocab.pkl file.

    Returns
    -------
    WordVocab
        Loaded vocabulary object with encode/decode capabilities.

    Raises
    ------
    FileNotFoundError
        If vocab.pkl file is not found in the model directory.

    Notes
    -----
    Requires vocab.pkl generated during training.
    """
    vocab_pkl_path = Path(model_dir) / "vocab.pkl"

    if not vocab_pkl_path.exists():
        raise FileNotFoundError(
            f"No vocab.pkl found in {model_dir}. "
            f"Please retrain your model to generate the vocabulary pickle file."
        )

    print(f"Loading WordVocab object from: {vocab_pkl_path}")
    with open(vocab_pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    """
    Main evaluation function with command-line interface.

    Provides complete evaluation pipeline including model loading,
    test data preparation, inference, metrics computation, error analysis,
    and results visualization with comparison to training performance.

    Notes
    -----
    Generates evaluation predictions CSV and confusion matrix visualization.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained phishing detection model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runs/cnn_bigru_word_tuned_patience_20",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test CSV file (optional - will recreate from raw data if not provided)",
    )  # noqa: E501
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_len", type=int, default=600, help="Maximum sequence length"
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

    # Load model and config
    print("\n=== Loading Model ===")
    model, training_metrics, training_args = load_model_and_config(
        args.model_dir, device
    )
    print("Model loaded successfully")

    if training_metrics:
        print(
            f"Original training test accuracy: {training_metrics.get('accuracy', 'N/A'):.4f}"
        )
        print(
            f"Original training test F1: {training_metrics.get('macro_f1', 'N/A'):.4f}"
        )

    # Load vocabulary
    print("\n=== Loading Vocabulary ===")
    vocab = load_vocabulary(args.model_dir)
    if hasattr(vocab, "vocab_size"):
        print(f"Vocabulary size: {vocab.vocab_size}")
    else:
        print(f"Vocabulary size: {len(vocab)}")  # For fallback word2idx dict

    # Load test data
    print("\n=== Loading Test Data ===")
    if args.test_data:
        # Use provided test file
        test_df = pd.read_csv(args.test_data)
        print(f"Loaded test data from {args.test_data}: {len(test_df)} samples")
    else:
        # Recreate test split from raw data (using same seed as training)
        seed = training_args.get("seed", 42)
        print(f"Recreating test split from raw data (seed={seed})")

        df = load_data("data/raw", normalize=False)
        df = remove_duplicates(df, exact=True)

        _, _, test_df = split_data(df, test_size=0.1, val_size=0.1, random_state=seed)
        print(f"Recreated test set: {len(test_df)} samples")

    print(f"Label distribution: {test_df['label'].value_counts().to_dict()}")

    # Create dataset and dataloader (using same interface as training)
    test_dataset = WordPhishingDataset(test_df, vocab, args.max_len)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Evaluate
    print("\n=== Evaluating Model ===")
    labels, preds, probs = evaluate_model(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(labels, preds)

    # Results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    if "legitimate_f1" in metrics:
        print(f"Legitimate F1: {metrics['legitimate_f1']:.4f}")
        print(f"Malicious F1: {metrics['malicious_f1']:.4f}")
        print(f"Legitimate Precision: {metrics['legitimate_precision']:.4f}")
        print(f"Malicious Precision: {metrics['malicious_precision']:.4f}")
        print(f"Legitimate Recall: {metrics['legitimate_recall']:.4f}")
        print(f"Malicious Recall: {metrics['malicious_recall']:.4f}")

    # Detailed report
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    print("(rows=true labels, cols=predicted)")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            labels, preds, target_names=["Legitimate", "Malicious"], digits=4
        )
    )

    # Error analysis
    test_df = test_df.copy()
    test_df["predicted"] = preds
    test_df["correct"] = preds == labels
    test_df["prob_legitimate"] = probs[:, 0]
    test_df["prob_malicious"] = probs[:, 1]

    errors = test_df[~test_df["correct"]]
    print("\nError Analysis:")
    print(
        f"Total errors: {len(errors)} / {len(test_df)} ({len(errors)/len(test_df)*100:.2f}%)"
    )

    false_positives = test_df[(test_df["label"] == 0) & (test_df["predicted"] == 1)]
    false_negatives = test_df[(test_df["label"] == 1) & (test_df["predicted"] == 0)]

    print(
        f"False Positives: {len(false_positives)} ({len(false_positives)/len(test_df)*100:.2f}%)"
    )
    print(
        f"False Negatives: {len(false_negatives)} ({len(false_negatives)/len(test_df)*100:.2f}%)"
    )

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Save predictions
    test_df.to_csv(output_dir / "evaluation_predictions.csv", index=False)

    # Save confusion matrix plot
    cm_path = output_dir / "evaluation_confusion_matrix.png"
    plot_confusion_matrix(labels, preds, ["Legitimate", "Malicious"], str(cm_path))

    print(f"\nResults saved to {output_dir}/")
    print("- evaluation_predictions.csv")
    print("- evaluation_confusion_matrix.png")

    # Comparison with training
    if training_metrics:
        print("\n" + "=" * 50)
        print("COMPARISON WITH TRAINING")
        print("=" * 50)
        train_acc = training_metrics.get("accuracy", 0)
        train_f1 = training_metrics.get("macro_f1", 0)

        print(f"Training Test Accuracy: {train_acc:.4f}")
        print(f"Current Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Difference:             {metrics['accuracy'] - train_acc:+.4f}")
        print()
        print(f"Training Test F1:       {train_f1:.4f}")
        print(f"Current F1:             {metrics['macro_f1']:.4f}")
        print(f"Difference:             {metrics['macro_f1'] - train_f1:+.4f}")

        if abs(metrics["accuracy"] - train_acc) < 0.01:
            print("\n✓ Results match training performance!")
        else:
            print(
                f"\n⚠ Results differ from training by {abs(metrics['accuracy'] - train_acc):.4f}"
            )


if __name__ == "__main__":
    main()
