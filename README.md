# Phishing Email Detection: A Comparative Study of Neural Networks and LLMs

A phishing email detection system using neural networks (CNN-BiGRU, DistilBERT) and zero-shot classification with large language models.

## Table of Contents

- [Phishing Email Detection: A Comparative Study of Neural Networks and LLMs](#phishing-email-detection-a-comparative-study-of-neural-networks-and-llms)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Install dependencies](#2-install-dependencies)
    - [3. Download GloVe embeddings (for CNN-BiGRU)](#3-download-glove-embeddings-for-cnn-bigru)
    - [4. Set up API keys (for zero-shot classification)](#4-set-up-api-keys-for-zero-shot-classification)
  - [Quick Start](#quick-start)
    - [Training DistilBERT](#training-distilbert)
    - [Training CNN-BiGRU](#training-cnn-bigru)
    - [Zero-Shot Classification](#zero-shot-classification)
      - [Using the convenience script (recommended)](#using-the-convenience-script-recommended)
      - [Direct Python usage](#direct-python-usage)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Data Preparation](#data-preparation)
      - [View dataset statistics](#view-dataset-statistics)
      - [Analyze text similarity](#analyze-text-similarity)
    - [Model Evaluation](#model-evaluation)
      - [Evaluate trained CNN-BiGRU](#evaluate-trained-cnn-bigru)
      - [Compare with fine-tuned model](#compare-with-fine-tuned-model)
    - [Hyperparameter Optimization](#hyperparameter-optimization)
    - [Dataset Analysis](#dataset-analysis)
      - [Show statistics](#show-statistics)
      - [Analyze text similarity](#analyze-text-similarity-1)
  - [Model Architectures](#model-architectures)
    - [WordCNNBiGRU](#wordcnnbigru)
    - [DistilBERT](#distilbert)
    - [Zero-Shot Classifier](#zero-shot-classifier)
  - [Requirements](#requirements)
  - [Contributors](#contributors)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd nn-phishing-detection
```

### 2. Install dependencies

Using poetry:

```bash
poetry install
```

### 3. Download GloVe embeddings (for CNN-BiGRU)

```bash
bash download_glove.sh
```

This downloads GloVe 840B 300d embeddings to `data/glove/glove.840B.300d.txt`.

### 4. Set up API keys (for zero-shot classification)

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Mistral
export MISTRAL_API_KEY="your-api-key-here"
```

## Quick Start

### Training DistilBERT

Fine-tune a DistilBERT model for phishing detection:

```bash
python src/nn_phishing_detection/training/train_distilbert.py \
    --train-data data/train.csv \
    --val-data data/val.csv \
    --test-data data/test.csv \
    --epochs 3 \
    --output-dir runs/distilbert_phishing
```

**Key arguments**:
- `--train-data`, `--val-data`, `--test-data`: Paths to CSV files with `text` and `label` columns
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 2e-5)
- `--patience`: Early stopping patience (default: 3)
- `--model-name`: Pretrained model (default: `distilbert-base-uncased`)

### Training CNN-BiGRU

Train a word-level CNN-BiGRU model with Optuna-optimized hyperparameters:

```bash
python src/nn_phishing_detection/training/train_word_bigru.py \
    --data data/raw \
    --glove data/glove/glove.840B.300d.txt \
    --embedding_dim 100 \
    --epochs 98 \
    --batch_size 128 \
    --lr 0.000961 \
    --weight_decay 1.29e-06 \
    --max_len 600 \
    --dropout 0.313 \
    --cnn_channels 300 \
    --conv_layers 3 \
    --kernel_size 3 \
    --gru_hidden 128 \
    --patience 3 \
    --out_dir runs/cnn_bigru_word_tuned \
    --seed 42
```

**Key arguments**:
- `--data`: Path to data directory or CSV file (required)
- `--glove`: Path to GloVe embeddings file (optional)
- `--embedding_dim`: Embedding dimension (default: 100)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization (default: 1e-6)
- `--max_len`: Maximum sequence length (default: 500)
- `--max_vocab`: Maximum vocabulary size (default: 30000)
- `--min_freq`: Minimum word frequency (default: 3)
- `--dropout`: Dropout rate (default: 0.3)
- `--cnn_channels`: CNN filter channels (default: 300)
- `--conv_layers`: Number of CNN layers (default: 1)
- `--kernel_size`: CNN kernel size (default: 5)
- `--gru_hidden`: GRU hidden size (default: 256)
- `--patience`: Early stopping patience (default: 3)
- `--seed`: Random seed (default: 42)
- `--out_dir`: Output directory (default: runs/word_bigru)

### Zero-Shot Classification

Classify emails using large language models without training.

#### Using the convenience script (recommended)

The easiest way to run zero-shot classification is using the `run_zero_shot.sh` script:

```bash
# Quick test with 10 samples
bash run_zero_shot.sh --quick

# OpenAI with batch API (50% cheaper, up to 24h processing)
bash run_zero_shot.sh --openai --batch

# OpenAI sequential API (faster, more expensive)
bash run_zero_shot.sh --openai

# Mistral with batch API
bash run_zero_shot.sh --mistral --batch

# Custom model
bash run_zero_shot.sh --provider openai --model gpt-4o

# Without text normalization
bash run_zero_shot.sh --openai --no-normalize
```

**Script options**:
- `--provider PROVIDER`: AI provider (`openai` or `mistral`, default: `openai`)
- `--model MODEL`: Specific model to use (default: `gpt-4o-mini` for OpenAI, `mistral-medium-latest` for Mistral)
- `--batch`: Use Batch API for 50% cost reduction (processing up to 24h)
- `--quick`: Test with only 10 samples
- `--openai`: Shorthand for `--provider openai`
- `--mistral`: Shorthand for `--provider mistral`
- `--normalize`: Apply text normalization (default: enabled)
- `--no-normalize`: Disable text normalization
- `--help`: Show help message

**Prerequisites**:
1. Set up API keys:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or
   export MISTRAL_API_KEY="your-key-here"
   ```

2. Run `train_optimal.sh` first to generate test data splits, or ensure test data exists.

#### Direct Python usage

Alternatively, call the Python script directly:

**Sequential API (faster, rate-limited)**:
```bash
python src/nn_phishing_detection/zero_shot/zs_classifier.py \
    --test-set data/test.csv \
    --provider openai \
    --model gpt-4o-mini \
    --batch-size 20 \
    --rpm 500 \
    --output-dir results/zero_shot
```

**Batch API (50% cheaper, slower)**:
```bash
python src/nn_phishing_detection/zero_shot/zs_classifier.py \
    --test-set data/test.csv \
    --provider openai \
    --model gpt-4o-mini \
    --use-batch \
    --output-dir results/zero_shot_batch
```

**Key arguments**:
- `--test-set`: Path to test CSV file (required)
- `--provider`: AI provider (`openai` or `mistral`, default: `openai`)
- `--model`: Model name (default: `gpt-4o-mini`)
- `--use-batch`: Use Batch API for 50% cost reduction
- `--rpm`: Requests per minute limit (default: 500)
- `--max-samples`: Limit samples for testing (optional)
- `--no-normalize`: Disable text normalization
- `--verbose`: Show detailed progress

## Project Structure

```
nn-phishing-detection/
├── src/nn_phishing_detection/
│   ├── models/
│   │   └── word_cnn_bigru.py         # CNN-BiGRU architecture
│   ├── training/
│   │   ├── train_distilbert.py       # DistilBERT training
│   │   └── train_word_bigru.py       # CNN-BiGRU training
│   ├── zero_shot/
│   │   └── zs_classifier.py          # Zero-shot classification
│   ├── hyperparameters/
│   │   └── optuna_word.py            # Hyperparameter optimization
│   ├── tools/
│   │   ├── data_stats.py             # Dataset statistics
│   │   └── similarity.py             # Similarity analysis
│   ├── vocab/
│   │   └── word_vocab.py             # Vocabulary building
│   ├── data_utils.py                 # Data loading utilities
│   ├── metrics.py                    # Evaluation metrics
│   ├── tracking.py                   # MLflow tracking
│   ├── evaluate_model.py             # Model evaluation
│   └── evaluate_base_distilbert.py   # Base DistilBERT eval
├── data/                             # Data directory
├── pyproject.toml                    # Project dependencies
├── download_glove.sh                 # Script to download Glove embeddings
├── finetune_distilbert.sh            # Script for finetuning of distilbert
├── run_zero_shot.sh                  # Script for running zero-shot experiments
├── train_optimal                     # Run model training with optimized parameters
└── README.md                         # This file
```

## Usage

### Data Preparation

Your CSV files should have at minimum:
- `text` column: Email text content
- `label` column: Binary labels (0=legitimate, 1=malicious)
- `source`: Dataset source identifier

#### View dataset statistics

```bash
python src/nn_phishing_detection/tools/data_stats.py \
    --data data/raw \
    --normalize
```

#### Analyze text similarity

```bash
python src/nn_phishing_detection/tools/similarity.py \
    --data data/raw \
    --sample-fraction 0.01 \
    --out-dir results/similarity
```

### Model Evaluation

#### Evaluate trained CNN-BiGRU

```bash
python src/nn_phishing_detection/evaluate_model.py \
    --model_dir runs/word_bigru/ \
    --test-data data/test.csv \
```

#### Compare with fine-tuned model

```bash
python src/nn_phishing_detection/evaluate_base_distilbert.py \
    --test-data data/test.csv \
    --finetuned-model runs/distilbert_phishing/best_model \
    --output-dir evaluation_results/comparison
```

### Hyperparameter Optimization

Optimize CNN-BiGRU hyperparameters with Optuna:

```bash
python src/nn_phishing_detection/hyperparameters/optuna_word.py \
    --data data/raw \
    --glove data/glove/glove.840B.300d.txt \
    --trials 50 \
    --pruner median \
    --embedding_dim 300 \
    --study_name word_bigru_optimization
```

This generates:
- SQLite database with trial history
- JSON file with best parameters
- Interactive HTML visualizations in `optuna_word_viz/`
- Text report with optimization summary

**Visualization outputs**:
- `optimization_history.html`: Trial progression
- `parallel_coordinate.html`: Parameter relationships
- `param_importances.html`: Most important parameters
- `slice_plot.html`: Individual parameter effects
- `f1_distribution.html`: Score distribution

### Dataset Analysis

#### Show statistics

```bash
python src/nn_phishing_detection/tools/data_stats.py \
    --data data/raw \
    --normalize
```

**Output includes**:
- Total sample count
- Label distribution (legitimate vs malicious)
- Source distribution
- Text length statistics
- Cross-tabulation by source

#### Analyze text similarity

```bash
python src/nn_phishing_detection/tools/similarity.py \
    --data data/raw \
    --sample-fraction 0.5 \
    --out-dir results/similarity
```

**Output includes**:
- High similarity pairs (≥0.9)
- Similarity distribution statistics
- Cross-label similarity counts
- JSON file with top 100 similar pairs

## Model Architectures

### WordCNNBiGRU

Word-level CNN-BiGRU combining convolutional layers for local feature extraction with bidirectional GRU for sequence modeling.

**Architecture**:
1. Word embeddings (300d GloVe or learned)
2. Spatial dropout
3. Multiple 1D CNN layers with residual connections
4. Bidirectional GRU
5. Global max + mean pooling
6. Classification head

### DistilBERT

Fine-tuned DistilBERT transformer for phishing classification.

### Zero-Shot Classifier

LLM-based classification without training using OpenAI or Mistral APIs.

**Supported models**:
- OpenAI: gpt-4o-mini
- Mistral: mistral-medium-latest

## Requirements

See `pyproject.toml` for complete dependency list.

## Contributors

- David Schatz <schatz@cl.uni-heidelberg.de>
- Elizaveta Dovedova <dovedova@cl.uni-heidelberg.de>