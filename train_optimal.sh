#!/bin/bash

# Train CNN-BiGRU with optimal hyperparameters from Optuna

echo "Training with optimal hyperparameters from Optuna..."
echo "============================================================================"
echo ""

# Check if GloVe embeddings exist
if [ ! -f "data/glove/glove.840B.300d.txt" ] && [ ! -f "data/glove/glove.6B.100d.txt" ]; then
    echo "Warning: GloVe embeddings not found!"
    echo "Please run: ./download_glove.sh or ./download_glove.sh 840B"
    echo ""
fi

# Train with best word-level parameters
echo "Training word-level CNN-BiGRU model..."
echo "  - Expected F1 Score: 98.03%"
echo "  - Conv layers: 3"
echo "  - CNN channels: 300"
echo "  - Kernel size: 3"
echo "  - GRU hidden: 128"
echo "  - Batch size: 128"
echo "  - Learning rate: 0.000961"
echo "  - Dropout: 0.313"
echo "  - Max length: 600 words"
echo ""

# Use 840B if available, otherwise 6B
if [ -f "data/glove/glove.840B.300d.txt" ]; then
    GLOVE_PATH="data/glove/glove.840B.300d.txt"
    EMBED_DIM=300
    echo "Using GloVe 840B embeddings (300d)"
else
    GLOVE_PATH="data/glove/glove.6B.100d.txt"
    EMBED_DIM=100
    echo "Using GloVe 6B embeddings (100d)"
fi

poetry run python3.12 src/nn_phishing_detection/training/train_word_bigru.py \
    --data data/raw \
    --glove $GLOVE_PATH \
    --embedding_dim $EMBED_DIM \
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
    --out_dir runs/cnn_bigru_word_tuned_patience_3 \
    --seed 42

echo ""
echo "================================================"
echo "Training complete!"