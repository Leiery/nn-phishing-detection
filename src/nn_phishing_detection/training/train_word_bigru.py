"""
Training script for word-level CNN-BiGRU phishing detection models.

This module provides a complete training pipeline for WordCNNBiGRU models including
data loading, vocabulary building, model training with early stopping, hyperparameter
optimization integration, and evaluation with MLflow experiment tracking.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

import argparse
import datetime
import json
import pickle
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.append("src")

from nn_phishing_detection import (
    WordCNNBiGRU,
    WordPhishingDataset,
    compute_metrics,
    load_data,
    log_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_training_progress,
    print_metrics_summary,
    remove_duplicates,
    save_metrics_json,
    setup_mlflow,
    split_data,
    start_run,
)
from nn_phishing_detection.analysis.error_analysis import (
    save_error_analysis,
)
from nn_phishing_detection.vocab.word_vocab import WordVocab


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible experiments.

    Configures random number generators for Python, NumPy, PyTorch, and CUDA
    to ensure reproducible training runs across different hardware.

    Parameters
    ----------
    seed : int, default=42
        Random seed value for all generators.

    Returns
    -------
    None
        Configures global random state.

    Notes
    -----
    Also sets torch.backends.cudnn.deterministic=True for full reproducibility,
    which may slightly reduce performance on CUDA devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    """
    Custom collate function for variable-length sequence batching.

    Pads sequences in the batch to the same length and stacks them into
    tensors suitable for CNN-BiGRU processing. Handles text sequences,
    labels, and sequence lengths.

    Parameters
    ----------
    batch : list of tuples
        List of (text_tensor, label_tensor, length_tensor) tuples from dataset.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing:
        - Padded text sequences (batch_size, max_seq_len)
        - Labels (batch_size,)
        - Original sequence lengths (batch_size,)

    Notes
    -----
    Pads sequences with zeros to match the longest sequence in the batch.
    Preserves original lengths for proper packed sequence handling in BiGRU.
    """
    texts, labels, lengths = zip(*batch, strict=False)

    # Pad sequences to the same length
    max_len = max(len(text) for text in texts)
    padded_texts = []
    for text in texts:
        padded = torch.cat([text, torch.zeros(max_len - len(text), dtype=torch.long)])
        padded_texts.append(padded)

    texts = torch.stack(padded_texts)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return texts, labels, lengths


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model performance on validation or test dataset.

    Performs evaluation on the provided dataset and computes loss, predictions,
    and class probabilities for metric calculation and analysis.

    Parameters
    ----------
    model : torch.nn.Module
        Trained WordCNNBiGRU model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing evaluation samples.
    criterion : torch.nn.Module
        Loss function for evaluation (typically CrossEntropyLoss).
    device : torch.device
        Device for model inference.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - Average loss across all batches
        - Ground truth labels (n_samples,)
        - Predicted labels (n_samples,)
        - Class probabilities (n_samples, n_classes)

    Notes
    -----
    Sets model to eval mode and disables gradients.
    Uses torch.softmax to convert logits to probabilities.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for texts, labels, lengths in tqdm(dataloader, desc="Evaluating"):
            texts = texts.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def train_epoch(
    model, dataloader, criterion, optimizer, scheduler, device, scaler=None
):
    """
    Train model for one complete epoch through the dataset.

    Performs forward pass, loss computation, backpropagation, and parameter
    updates for all batches in the training dataset. Supports mixed precision
    training with gradient scaling.

    Parameters
    ----------
    model : torch.nn.Module
        WordCNNBiGRU model to train.
    dataloader : torch.utils.data.DataLoader
        Training data loader.
    criterion : torch.nn.Module
        Loss function (typically CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    device : torch.device
        Device for training.
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training.

    Returns
    -------
    float
        Average training loss for the epoch.

    Notes
    -----
    Uses gradient clipping to prevent exploding gradients.
    Supports automatic mixed precision (AMP) when scaler is provided.
    Updates learning rate scheduler after each batch.
    """
    model.train()
    total_loss = 0.0

    for texts, labels, lengths in tqdm(dataloader, desc="Training"):
        texts = texts.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with autocast(device_type=device.type):
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            model.clip_gradients(max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            model.clip_gradients(max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    if scheduler:
        scheduler.step()

    return total_loss / len(dataloader)


def main():
    """
    Main training function with command-line interface.

    Provides complete WordCNNBiGRU training pipeline including data loading,
    vocabulary building with GloVe embeddings, model training with early stopping,
    evaluation, and results visualization with MLflow experiment tracking.

    Notes
    -----
    Supports automatic device detection (CUDA/MPS/CPU) and mixed precision training.
    Implements early stopping based on validation F1 score with configurable patience.
    Builds word vocabulary with spaCy tokenization and GloVe embedding integration.
    Generates visualizations including confusion matrices and PR curves.
    Saves best model, vocabulary, training history, and detailed evaluation results.
    Includes tokenization caching for improved performance on repeated runs.
    """
    parser = argparse.ArgumentParser(
        description="Train word-level CNN-BiGRU for binary phishing detection"
    )  # noqa: E501
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument(
        "--glove", type=str, default=None, help="Path to GloVe embeddings file"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (paper used 32)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--max_len", type=int, default=500, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_vocab", type=int, default=30000, help="Maximum vocabulary size"
    )
    parser.add_argument(
        "--min_freq", type=int, default=3, help="Minimum word frequency"
    )
    parser.add_argument(
        "--out_dir", type=str, default="runs/word_bigru", help="Output directory"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--cnn_channels", type=int, default=300, help="CNN channels")
    parser.add_argument(
        "--conv_layers", type=int, default=1, help="Number of conv layers"
    )
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size")
    parser.add_argument("--gru_hidden", type=int, default=256, help="GRU hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="Embedding dimension (100 for GloVe)",
    )  # noqa: E501

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Setup MLflow
    setup_mlflow()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Load data
    print("\n=== Loading Data ===")
    df = load_data(args.data, normalize=False)
    print(f"Loaded {len(df)} samples")

    # Remove exact duplicates
    df = remove_duplicates(df, exact=True)

    # Split data
    train_df, val_df, test_df = split_data(
        df, test_size=0.1, val_size=0.1, random_state=args.seed
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save dataset splits as artifacts
    print("\n=== Saving Dataset Splits ===")
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    val_df.to_csv(splits_dir / "val_split.csv", index=False)
    test_df.to_csv(splits_dir / "test_split.csv", index=False)

    print(f"Saved dataset splits to {splits_dir}")

    # Log dataset splits to MLflow immediately
    print(f"Data splits saved to {splits_dir}/")
    print("Logged dataset splits to MLflow")

    # Build word vocabulary
    print("\n=== Building Word Vocabulary ===")

    # Try to load existing tokenization cache
    cache_file = out_dir / "train_set_tokenization_cache.pkl"

    vocab = WordVocab(
        train_df["text"],
        max_vocab_size=args.max_vocab,
        min_freq=args.min_freq,
        glove_path=args.glove,
        embedding_dim=args.embedding_dim,
        cache_tokenization=True,
        cache_file=str(cache_file) if cache_file.exists() else None,
    )

    # Save the updated cache for next time
    print(f"Saving tokenization cache to {cache_file}")
    vocab.save_tokenization_cache(str(cache_file))

    # Create datasets
    train_dataset = WordPhishingDataset(train_df, vocab, args.max_len)
    val_dataset = WordPhishingDataset(val_df, vocab, args.max_len)
    test_dataset = WordPhishingDataset(test_df, vocab, args.max_len)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Compute class weights (though data is fairly balanced)
    train_labels = torch.tensor(train_dataset.labels)
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")

    # Create model
    print("\n=== Creating Word-level CNN-BiGRU Model ===")
    model = WordCNNBiGRU(
        vocab_size=vocab.vocab_size,
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
        conv_layers=args.conv_layers,
        kernel_size=args.kernel_size,
        gru_hidden=args.gru_hidden,
        num_classes=2,  # Binary classification
        pretrained_embeddings=vocab.embeddings,  # Use GloVe if available
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2, eta_min=1e-5
    )

    # Mixed precision
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # Training parameters for MLflow
    params = {
        "model": "word_cnn_bigru",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "cnn_channels": args.cnn_channels,
        "conv_layers": args.conv_layers,
        "kernel_size": args.kernel_size,
        "gru_hidden": args.gru_hidden,
        "vocab_size": vocab.vocab_size,
        "max_vocab": args.max_vocab,
        "min_freq": args.min_freq,
        "embedding_dim": args.embedding_dim,
        "glove": args.glove is not None,
        "seed": args.seed,
        "total_params": total_params,
        "classes": 2,  # Binary
    }

    # Start MLflow run
    # Ensure no active run exists
    mlflow.end_run()

    # Create descriptive run name with timestamp and key hyperparameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"word_cnn_bigru_{timestamp}_c{args.cnn_channels}_l{args.conv_layers}_k{args.kernel_size}_h{args.gru_hidden}_e{args.epochs}"  # noqa: E501

    with start_run(run_name=run_name, params=params):
        best_val_f1 = 0.0
        patience_counter = 0

        # Track training progress
        train_losses = []
        val_losses = []
        val_f1_scores = []

        print("\n=== Training ===")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            # Train
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, scaler
            )

            # Validate
            val_loss, val_labels, val_preds, val_probs = evaluate(
                model, val_loader, criterion, device
            )

            # Compute metrics
            val_metrics = compute_metrics(val_labels, val_preds)
            val_f1 = val_metrics["macro_f1"]

            # Store for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)

            # Log to MLflow
            log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_metrics["accuracy"],
                },
                step=epoch,
            )

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val Macro F1: {val_f1:.4f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                best_model_path = out_dir / "best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model (Val F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Evaluate on test set
        print("\n=== Evaluating on Test Set ===")
        test_loss, test_labels, test_preds, test_probs = evaluate(
            model, test_loader, criterion, device
        )

        # Compute test metrics
        test_metrics = compute_metrics(test_labels, test_preds)
        print_metrics_summary(test_metrics)

        # Log test metrics
        log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_legitimate_f1": test_metrics["legitimate_f1"],
                "test_malicious_f1": test_metrics["malicious_f1"],
                "test_legitimate_precision": test_metrics["legitimate_precision"],
                "test_malicious_precision": test_metrics["malicious_precision"],
                "test_legitimate_recall": test_metrics["legitimate_recall"],
                "test_malicious_recall": test_metrics["malicious_recall"],
            }
        )

        # Plot training progress
        print("\n=== Plotting Training Progress ===")
        progress_path_prefix = str(out_dir / "training_progress")
        plot_training_progress(
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_f1_scores,
            test_loss=test_loss,
            outpath_prefix=progress_path_prefix,
        )

        # Log training progress plots
        print(f"Training progress plots saved to {progress_path_prefix}_*.png")

        # Save artifacts
        print("\n=== Saving Artifacts ===")

        # Confusion matrix (binary: legitimate vs malicious)
        cm_path = out_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            test_labels, test_preds, ["legitimate", "malicious"], str(cm_path)
        )
        print(f"Confusion matrix saved to {cm_path}")

        # Error analysis
        print("\n=== Performing Error Analysis ===")
        try:
            # Reload test data to get original texts
            save_error_analysis(
                model=model,
                dataloader=test_loader,
                dataset_df=test_df,
                vocab=vocab,
                device=device,
                output_dir=out_dir,
                max_samples=100,  # Save up to 100 examples of each error type
            )

            # Log error analysis artifacts
            error_dir = out_dir / "error_analysis"
            if error_dir.exists():
                print(f"Error analysis saved to {error_dir}/")

        except Exception as e:
            print(f"Warning: Error analysis failed: {e}")

        # PR curves
        pr_path_prefix = str(out_dir / "pr_curves")
        plot_pr_curves(
            test_labels, test_probs, ["legitimate", "malicious"], pr_path_prefix
        )
        print(f"PR curves saved to {pr_path_prefix}_*.png")

        # Save metrics JSON
        metrics_path = out_dir / "metrics.json"
        save_metrics_json(test_metrics, str(metrics_path))
        print(f"Metrics saved to {metrics_path}")

        # Save final model (multiple formats for debugging)
        model_path = out_dir / "model_state.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save entire model object
        full_model_path = out_dir / "full_model.pt"
        torch.save(model, full_model_path)

        # Save vocabulary as JSON
        vocab_path = out_dir / "vocab.json"

        vocab_data = {
            "word2idx": vocab.word2idx,
            "idx2word": {str(k): v for k, v in vocab.idx2word.items()},
            "vocab_size": vocab.vocab_size,
        }
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, indent=2)

        # Save full WordVocab object (for proper evaluation)
        vocab_pkl_path = out_dir / "vocab.pkl"
        
        with open(vocab_pkl_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Saved vocabulary to {vocab_path} and {vocab_pkl_path}")

        # Save model + vocab + args as checkpoint
        checkpoint_path = out_dir / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "vocab_size": vocab.vocab_size,
                    "embedding_dim": args.embedding_dim,
                    "num_classes": 2,
                    "cnn_channels": args.cnn_channels,
                    "kernel_size": args.kernel_size,
                    "conv_layers": args.conv_layers,
                    "gru_hidden": args.gru_hidden,
                    "dropout": args.dropout,
                },
                "vocab_size": vocab.vocab_size,
                "args": vars(args),
                "test_metrics": test_metrics,
            },
            checkpoint_path,
        )

        # Log remaining artifacts to MLflow
        print("\n=== Logging Final Artifacts to MLflow ===")
        print(f"Full model saved to {full_model_path}")
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Checkpoint saved to {checkpoint_path}")

        print(f"\nArtifacts saved to {out_dir} and logged to MLflow")
        print("\nFinal Test Performance:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
