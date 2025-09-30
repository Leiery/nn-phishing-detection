#!/bin/bash

# Fine-tune DistilBERT for phishing detection

echo "Fine-tuning DistilBERT for phishing detection..."
echo "============================================================================"
echo ""

# Check if splits exist from CNN training
if [ ! -f "runs/cnn_bigru_word_tuned_patience_3_with_vocab_pkl/splits/train_split.csv" ]; then
    echo "Error: Training splits not found!"
    echo "Please run ./train_optimal.sh first to generate the data splits"
    exit 1
fi

echo "Finetuning DistilBERT model..."
echo "  - Model: distilbert-base-uncased"
echo "  - Batch size: 32"
echo "  - Learning rate: 2e-5"
echo "  - Weight decay: 0.01"
echo "  - Max length: 512 tokens"
echo "  - Epochs: 10 (with early stopping)"
echo "  - Patience: 3"
echo ""

# Use the same splits as CNN for fair comparison
TRAIN_DATA="runs/cnn_bigru_word_tuned_patience_3/splits/train_split.csv"
VAL_DATA="runs/cnn_bigru_word_tuned_patience_3/splits/val_split.csv"
TEST_DATA="runs/cnn_bigru_word_tuned_patience_3/splits/test_split.csv"

# Train DistilBERT
python src/nn_phishing_detection/training/train_distilbert.py \
    --train-data $TRAIN_DATA \
    --val-data $VAL_DATA \
    --test-data $TEST_DATA \
    --model-name distilbert-base-uncased \
    --max-length 512 \
    --epochs 10 \
    --batch-size 32 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --patience 3 \
    --warmup-steps 500 \
    --output-dir runs/finetuned_distilbert_phishing_e10_p3 \
    --seed 42 \
    $MLFLOW_URI

echo ""
echo "================================================"
echo "DistilBERT finetuning complete!"