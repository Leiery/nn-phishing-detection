"""
MLflow experiment tracking utilities for phishing detection models.

This module provides convenient wrapper functions and context managers for MLflow
experiment tracking, including run management, metric logging, and artifact handling.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""

from contextlib import contextmanager
from typing import Any

import mlflow
import mlflow.pytorch


def setup_mlflow(tracking_uri: str | None = None) -> None:
    """
    Initialize MLflow tracking with specified or default URI.

    Sets up MLflow experiment tracking backend. If no URI is provided,
    defaults to local file store in ./mlruns directory.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking server URI. If None, uses local file store.

    Returns
    -------
    None
        Configures MLflow global tracking URI.

    Notes
    -----
    Common tracking URIs:
    - Local: "file:./mlruns" (default)
    - Remote server: "http://localhost:5000"
    - SQLite: "sqlite:///mlflow.db"
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Use local file store by default
        mlflow.set_tracking_uri("file:./mlruns")


@contextmanager
def start_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    experiment_name: str = "nn-phishing-detection",
):
    """
    Context manager for MLflow experiment runs with automatic setup.

    Creates or accesses an MLflow experiment and starts a new run with
    the specified parameters. Automatically logs parameters and run metadata.

    Parameters
    ----------
    run_name : str
        Descriptive name for the experiment run.
    params : dict[str, Any], optional
        Dictionary of hyperparameters and configuration to log.
    experiment_name : str, default="nn-phishing-detection"
        Name of the MLflow experiment to use or create.

    Yields
    ------
    mlflow.ActiveRun
        Active MLflow run object for logging metrics and artifacts.

    Returns
    -------
    None
        Context manager that handles run lifecycle automatically.

    Notes
    -----
    Automatically creates the experiment if it doesn't exist.
    Filters out None values from parameters before logging.
    Prints run ID and name for tracking purposes.
    """
    # Ensure experiment exists
    experiment_id = get_experiment_id(experiment_name)
    if experiment_id is None:
        print(f"Creating new experiment: {experiment_name}")
        experiment_id = create_experiment(experiment_name)

    mlflow.set_experiment(experiment_id)

    # Start run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters if provided
        if params:
            for key, value in params.items():
                if value is not None:
                    mlflow.log_param(key, value)

        # Log run info
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow Run Name: {run_name}")

        yield run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log performance metrics to the active MLflow run.

    Logs multiple metrics to MLflow, filtering out invalid values
    like None, lists, or dictionaries that cannot be logged as scalars.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary of metric names and values to log.
    step : int, optional
        Step number for time-series metrics (e.g., epoch number).

    Returns
    -------
    None
        Logs metrics to active MLflow run.

    Notes
    -----
    Only logs scalar numeric values. Skips None, list, or dict values.
    Converts all values to float for MLflow storage.
    Must be called within an active MLflow run context.
    """
    for key, value in metrics.items():
        if value is not None and not isinstance(value, (list, dict)):
            mlflow.log_metric(key, float(value), step=step)


def get_experiment_id(experiment_name: str) -> str | None:
    """
    Retrieve MLflow experiment ID by name.

    Looks up an existing MLflow experiment by name and returns its ID.
    Used to check if an experiment exists before creating a new one.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to find.

    Returns
    -------
    str or None
        Experiment ID if found, None otherwise.

    Notes
    -----
    Handles exceptions gracefully and returns None for any lookup errors.
    Used internally by start_run() to manage experiment lifecycle.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return experiment.experiment_id if experiment else None
    except Exception:
        return None


def create_experiment(experiment_name: str) -> str:
    """
    Create a new MLflow experiment with the specified name.

    Creates a new experiment in the MLflow tracking backend.
    Used when an experiment doesn't exist and needs to be created.

    Parameters
    ----------
    experiment_name : str
        Name for the new MLflow experiment.

    Returns
    -------
    str
        ID of the newly created experiment.

    Notes
    -----
    Used internally by start_run() when an experiment doesn't exist.
    Experiment names must be unique within the MLflow tracking backend.
    """
    return mlflow.create_experiment(experiment_name)
