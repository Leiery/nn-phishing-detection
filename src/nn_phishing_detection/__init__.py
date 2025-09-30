"""Neural Network Phishing Detection Package."""

from .data_utils import (
    TransformerDataset,
    WordPhishingDataset,
    compute_text_hash,
    load_data,
    normalize_text,
    remove_duplicates,
    split_data,
)
from .metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_training_progress,
    print_metrics_summary,
    save_metrics_json,
)
from .models.word_cnn_bigru import WordCNNBiGRU
from .tracking import (
    create_experiment,
    get_experiment_id,
    log_metrics,
    setup_mlflow,
    start_run,
)

__all__ = [
    # data_utils
    'load_data',
    'remove_duplicates',
    'split_data',
    'WordPhishingDataset',
    'TransformerDataset',
    'normalize_text',
    'compute_text_hash',

    # metrics
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_pr_curves',
    'plot_training_progress',
    'save_metrics_json',
    'print_metrics_summary',

    # tracking
    'setup_mlflow',
    'start_run',
    'log_metrics',
    'create_experiment',
    'get_experiment_id',

    # models
    'WordCNNBiGRU'
]