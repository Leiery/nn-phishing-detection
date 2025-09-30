# ruff: noqa: E501
"""
Optuna hyperparameter optimization for word-level phishing detection models.

This module provides hyperparameter optimization using Optuna for
WordCNNBiGRU models. Includes pruning, parallel optimization,
and results visualization for finding optimal model configurations.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna.visualization as vis
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

import optuna

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader

# Add project root to path
sys.path.append("src")

from nn_phishing_detection import (
    WordCNNBiGRU,
    WordPhishingDataset,
    compute_metrics,
    load_data,
    log_metrics,
    remove_duplicates,
    setup_mlflow,
    split_data,
    start_run,
)
from nn_phishing_detection.vocab.word_vocab import WordVocab


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Ensures deterministic behavior by setting seeds for Python's random module,
    NumPy, PyTorch CPU and CUDA operations.

    Parameters
    ----------
    seed : int, default=42
        Random seed value to set across all libraries.

    Returns
    -------
    None

    Notes
    -----
    This function sets torch.backends.cudnn.deterministic=True, which may
    reduce performance but ensures reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    """
    Custom collate function for batching phishing detection samples.

    Stacks text encodings, labels, and sequence lengths into batch tensors
    for efficient model processing.

    Parameters
    ----------
    batch : list of tuples
        List of (encoded_text, label, length) tuples from Dataset.__getitem__.

    Returns
    -------
    tuple of torch.Tensor
        Batched tensors (texts, labels, lengths) where:
        - texts: (batch_size, max_len) encoded text sequences
        - labels: (batch_size,) classification labels
        - lengths: (batch_size,) actual sequence lengths for packing

    Notes
    -----
    Uses zip with strict=False for backward compatibility with Python <3.10.
    """
    texts, labels, lengths = zip(*batch, strict=False)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return texts, labels, lengths


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Train model for one epoch with mixed precision support.

    Performs one complete training epoch over the dataset with gradient clipping,
    optional automatic mixed precision (AMP) for CUDA devices, and loss aggregation.

    Parameters
    ----------
    model : WordCNNBiGRU
        Neural network model to train.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of training data.
    criterion : torch.nn.Module
        Loss function (e.g., CrossEntropyLoss with class weights).
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device
        Device to run training on (cuda/mps/cpu).
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training on CUDA.

    Returns
    -------
    float
        Average training loss across all batches.

    Notes
    -----
    Uses gradient clipping with max_norm=1.0 to prevent exploding gradients.
    Automatically applies AMP when scaler is provided and device is CUDA.
    """
    model.train()
    total_loss = 0.0

    for texts, labels, lengths in dataloader:
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

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation or test dataset.

    Computes loss and predictions without gradient computation for efficient
    evaluation during training or final testing.

    Parameters
    ----------
    model : WordCNNBiGRU
        Neural network model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of evaluation data.
    criterion : torch.nn.Module
        Loss function for computing evaluation loss.
    device : torch.device
        Device to run evaluation on (cuda/mps/cpu).

    Returns
    -------
    avg_loss : float
        Average loss across all evaluation batches.
    all_labels : np.ndarray
        Ground truth labels (shape: (n_samples,)).
    all_preds : np.ndarray
        Predicted labels (shape: (n_samples,)).

    Notes
    -----
    Sets model to eval mode and disables gradient computation for efficiency.
    Predictions are obtained via argmax over the output logits.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_labels), np.array(all_preds)


class OptunaObjective:
    """
    Optuna objective function for WordCNNBiGRU hyperparameter optimization.

    Defines the search space, training procedure, and optimization objective
    for finding optimal model hyperparameters using Optuna with pruning.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing data paths and configuration.
    train_df : pd.DataFrame
        Training dataset with 'text' and 'label' columns.
    val_df : pd.DataFrame
        Validation dataset for hyperparameter evaluation.
    device : torch.device
        Device for model training (cuda/mps/cpu).

    Attributes
    ----------
    vocab : WordVocab or None
        Word vocabulary built once and reused across trials.

    Notes
    -----
    Builds vocabulary once on first trial for efficiency.
    Returns best validation macro F1 score as optimization objective.
    Supports Optuna pruning to terminate unpromising trials early.
    """

    def __init__(self, args, train_df, val_df, device):
        self.args = args
        self.train_df = train_df
        self.val_df = val_df
        self.device = device
        self.vocab = None

    def __call__(self, trial):
        """
        Execute single Optuna trial with sampled hyperparameters.

        Samples hyperparameters from the search space, trains a WordCNNBiGRU
        model, and returns the best validation F1 score for optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for suggesting hyperparameters.

        Returns
        -------
        float
            Best validation macro F1 score achieved during training.

        Raises
        ------
        optuna.TrialPruned
            If the trial is pruned based on intermediate results.

        Notes
        -----
        Search space includes learning rate, dropout, CNN architecture,
        GRU hidden size, batch size, max sequence length, and regularization.
        Implements early stopping with patience=3 on validation F1 score.
        Logs all metrics to MLflow for tracking and visualization.
        """

        # Common hyperparameters
        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = trial.suggest_int("epochs", 6, 100)

        # Model architecture hyperparameters (word-level specific ranges based on paper)
        cnn_channels = trial.suggest_categorical("cnn_channels", [100, 256, 300, 600])
        conv_layers = trial.suggest_int("conv_layers", 1, 6)
        kernel_size = trial.suggest_categorical("kernel_size", [1, 3, 5, 7, 10])
        max_len = trial.suggest_int("max_len", 300, 800, step=100)

        gru_hidden = trial.suggest_categorical("gru_hidden", [128, 192, 256, 320])

        # Build vocabulary if not already done
        if self.vocab is None:
            print("Building word vocabulary...")
            self.vocab = WordVocab(
                self.train_df["text"],
                max_vocab_size=30000,
                min_freq=3,
                glove_path=self.args.glove,
                embedding_dim=self.args.embedding_dim,
            )

        pretrained_embeddings = self.vocab.embeddings

        # Create datasets
        train_dataset = WordPhishingDataset(self.train_df, self.vocab, max_len)
        val_dataset = WordPhishingDataset(self.val_df, self.vocab, max_len)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )

        # Create model
        model = WordCNNBiGRU(
            vocab_size=self.vocab.vocab_size,
            dropout=dropout,
            cnn_channels=cnn_channels,
            conv_layers=conv_layers,
            kernel_size=kernel_size,
            gru_hidden=gru_hidden,
            num_classes=2,  # Binary classification
            pretrained_embeddings=pretrained_embeddings,
        ).to(self.device)

        # Compute class weights
        train_labels = torch.tensor(train_dataset.labels)
        class_counts = torch.bincount(train_labels)
        if len(class_counts) < 2:
            class_counts = torch.cat([class_counts, torch.zeros(2 - len(class_counts))])
        class_weights = 1.0 / class_counts.float().clamp(min=1)
        class_weights = class_weights / class_weights.sum() * 2
        class_weights = class_weights.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=2, eta_min=1e-5
        )

        # Mixed precision
        scaler = GradScaler("cuda") if self.device.type == "cuda" else None

        # Training parameters for MLflow
        params = {
            "trial_number": trial.number,
            "lr": lr,
            "dropout": dropout,
            "cnn_channels": cnn_channels,
            "conv_layers": conv_layers,
            "kernel_size": kernel_size,
            "gru_hidden": gru_hidden,
            "weight_decay": weight_decay,
            "max_len": max_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "vocab_size": self.vocab.vocab_size,
        }

        # Start MLflow run
        run_name = f"optuna_word_trial_{trial.number}"
        with start_run(run_name=run_name, params=params):

            best_val_f1 = 0.0
            patience = 3
            patience_counter = 0

            # Training loop
            for epoch in range(epochs):
                # Train
                train_loss = train_epoch(
                    model, train_loader, criterion, optimizer, self.device, scaler
                )

                # Validate
                val_loss, val_labels, val_preds = evaluate(
                    model, val_loader, criterion, self.device
                )

                # Compute metrics
                val_metrics = compute_metrics(val_labels, val_preds)
                val_f1 = val_metrics["macro_f1"]

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

                # Update best score and early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience and epoch >= 3:
                        log_metrics({"early_stopped": 1, "stopped_epoch": epoch})
                        break

                # Report to Optuna for pruning
                trial.report(val_f1, epoch)

                # Prune trial if needed
                if trial.should_prune():
                    log_metrics({"pruned": 1})
                    raise optuna.TrialPruned()

                # Update scheduler
                scheduler.step()

            # Log final best score
            log_metrics({"best_val_macro_f1": best_val_f1})

        return best_val_f1


def main():
    """
    Main entry point for Optuna hyperparameter optimization.

    Configures and executes complete hyperparameter search for WordCNNBiGRU
    models with visualization and results reporting.

    Notes
    -----
    Provides command-line interface for configuring:
    - Data path and GloVe embeddings
    - Number of optimization trials and pruning strategy
    - MLflow tracking URI and study persistence
    - Random seed for reproducibility

    Generates visualization artifacts including:
    - Optimization history plots
    - Parameter importance analysis
    - Contour and slice plots
    - Trial timeline and F1 distribution

    Results are saved to JSON file and SQLite database for persistence.
    All visualizations are exported as interactive HTML files.
    """
    parser = argparse.ArgumentParser(
        description="Optuna optimization for word-level CNN-BiGRU"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument(
        "--glove", type=str, default=None, help="Path to GloVe embeddings"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=300, help="Embedding dimension"
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["median", "asha"],
        default="median",
        help="Pruner type",
    )
    parser.add_argument("--mlflow", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name (default: auto-generated based on vocab type)",
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Auto-generate study name if not provided
    if args.study_name is None:
        args.study_name = "word_bigru_optimization"

    # Setup MLflow
    if args.mlflow:
        setup_mlflow(args.mlflow)
    else:
        setup_mlflow()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load and preprocess data
    print("\n=== Loading Data ===")
    df = load_data(args.data, normalize=False)

    # Binary classification (0=legitimate, 1=malicious/phishing)
    print(
        f"Binary classification: 0={(df['label']==0).sum()}, 1={(df['label']==1).sum()}"
    )

    # Remove duplicates
    df = remove_duplicates(df, exact=True)

    # Split data
    train_df, val_df, test_df = split_data(
        df, test_size=0.1, val_size=0.1, random_state=args.seed
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Select pruner
    if args.pruner == "median":
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    else:
        pruner = SuccessiveHalvingPruner()

    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=pruner,
        storage=f"sqlite:///{args.study_name}.db",
        load_if_exists=True,
    )

    # Create objective
    objective = OptunaObjective(args, train_df, val_df, device)

    # Optimize
    print(f"\n=== Starting Optuna Optimization ({args.trials} trials) ===")
    print(f"Study name: {args.study_name}")
    if args.glove:
        print(f"Using GloVe embeddings from: {args.glove}")

    study.optimize(objective, n_trials=args.trials)

    # Print results
    print("\n=== Optimization Results ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Val Macro F1): {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study results
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }

    results_file = "optuna_word_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Generate Optuna visualizations
    print("\n=== Generating Optuna Visualizations ===")
    try:

        # Create visualization directory
        viz_dir = Path("optuna_word_viz")
        viz_dir.mkdir(exist_ok=True)

        # 1. Optimization History
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(viz_dir / "optimization_history.html"))
        print(f" Optimization history saved to {viz_dir}/optimization_history.html")

        # 2. Parallel Coordinate Plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(viz_dir / "parallel_coordinate.html"))
        print(f" Parallel coordinate plot saved to {viz_dir}/parallel_coordinate.html")

        # 3. Parameter Importances
        try:
            fig = vis.plot_param_importances(study)
            fig.write_html(str(viz_dir / "param_importances.html"))
            print(f" Parameter importances saved to {viz_dir}/param_importances.html")
        except Exception as e:
            print(f" Could not generate parameter importances: {e}")

        # 4. Contour Plot (for top 2 most important params)
        try:
            importances = optuna.importance.get_param_importances(study)
            if len(importances) >= 2:
                top_params = list(importances.keys())[:2]
                fig = vis.plot_contour(study, params=top_params)
                fig.write_html(
                    str(viz_dir / f"contour_{top_params[0]}_{top_params[1]}.html")
                )
                print(
                    f" Contour plot saved to {viz_dir}/contour_{top_params[0]}_{top_params[1]}.html"
                )
        except Exception as e:
            print(f" Could not generate contour plot: {e}")

        # 5. Slice Plot (individual parameter effects)
        try:
            fig = vis.plot_slice(study)
            fig.write_html(str(viz_dir / "slice_plot.html"))
            print(f" Slice plot saved to {viz_dir}/slice_plot.html")
        except Exception as e:
            print(f" Could not generate slice plot: {e}")

        # 6. Timeline Plot (trial durations)
        try:
            fig = vis.plot_timeline(study)
            fig.write_html(str(viz_dir / "timeline.html"))
            print(f" Timeline plot saved to {viz_dir}/timeline.html")
        except Exception as e:
            print(f" Could not generate timeline plot: {e}")

        # 7. Custom summary plot
        try:
            # Create a summary plot with key metrics
            trials_df = study.trials_dataframe()

            # F1 score distribution
            fig = px.histogram(
                trials_df,
                x="value",
                nbins=20,
                title="F1 Score Distribution (word-level)",
                labels={"value": "Validation F1 Score", "count": "Number of Trials"},
            )
            fig.write_html(str(viz_dir / "f1_distribution.html"))

            # Parameter vs F1 scatter plots for numeric parameters
            numeric_params = ["lr", "dropout", "weight_decay", "epochs"]
            for param in numeric_params:
                param_col = f"params_{param}"
                if param_col in trials_df.columns:
                    fig = px.scatter(
                        trials_df,
                        x=param_col,
                        y="value",
                        title=f"{param.title()} vs F1 Score",
                        labels={
                            "value": "Validation F1 Score",
                            param_col: param.title(),
                        },
                    )
                    fig.write_html(str(viz_dir / f"{param}_vs_f1.html"))

            print(f" Custom analysis plots saved to {viz_dir}/")

        except Exception as e:
            print(f" Could not generate custom plots: {e}")

        # Generate summary report
        report_path = viz_dir / "optimization_report.txt"
        with open(report_path, "w") as f:
            f.write("Optuna Optimization Report\n")
            f.write("=" * 30 + "\n\n")
            f.write("Vocabulary Type: word\n")
            f.write(f"Total Trials: {len(study.trials)}\n")
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best F1 Score: {study.best_value:.4f}\n\n")

            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nStudy Statistics:\n")
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            pruned_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ]
            f.write(f"  Completed trials: {len(completed_trials)}\n")
            f.write(f"  Pruned trials: {len(pruned_trials)}\n")

            if completed_trials:
                values = [t.value for t in completed_trials]
                f.write(f"  Mean F1: {np.mean(values):.4f}\n")
                f.write(f"  Std F1: {np.std(values):.4f}\n")
                f.write(f"  Min F1: {np.min(values):.4f}\n")
                f.write(f"  Max F1: {np.max(values):.4f}\n")

        print(f" Optimization report saved to {report_path}")
        print(f"\n All visualizations saved in: {viz_dir}/")
        print("Open the .html files in a browser to view interactive plots!")

    except ImportError:
        print("Optuna visualization requires plotly")
        print("Install with: pip install optuna plotly")
    except Exception as e:
        print(f" Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
