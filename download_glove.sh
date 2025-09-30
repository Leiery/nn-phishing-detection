#!/bin/bash

# Download GloVe embeddings
echo "GloVe Embedding Downloader"
echo "=========================="

mkdir -p data/glove

# Check command line argument
if [ "$1" = "840B" ] || [ "$1" = "840b" ]; then
    # Download 840B tokens, 300d vectors (2.2GB)
    if [ ! -f data/glove/glove.840B.300d.txt ]; then
        echo "Downloading GloVe 840B.300d (2.2GB - highest quality)..."
        echo "This will take a while..."
        wget -P data/glove/ http://nlp.stanford.edu/data/glove.840B.300d.zip
        cd data/glove/
        unzip glove.840B.300d.zip
        rm glove.840B.300d.zip
        cd ../..
        echo "GloVe 840B embeddings downloaded to data/glove/"
        GLOVE_FILE="data/glove/glove.840B.300d.txt"
        EMBEDDING_DIM=300
    else
        echo "GloVe 840B embeddings already exist"
        GLOVE_FILE="data/glove/glove.840B.300d.txt"
        EMBEDDING_DIM=300
    fi
else
    # Default: Download 6B tokens, 100d vectors (171MB)
    if [ ! -f data/glove/glove.6B.100d.txt ]; then
        echo "Downloading GloVe 6B.100d (171MB - standard)..."
        wget -P data/glove/ http://nlp.stanford.edu/data/glove.6B.zip
        cd data/glove/
        unzip glove.6B.zip
        rm glove.6B.zip
        # Keep only 100d version to save space
        rm glove.6B.50d.txt glove.6B.200d.txt glove.6B.300d.txt 2>/dev/null || true
        cd ../..
        echo "GloVe 6B embeddings downloaded to data/glove/"
        GLOVE_FILE="data/glove/glove.6B.100d.txt"
        EMBEDDING_DIM=100
    else
        echo "GloVe 6B embeddings already exist"
        GLOVE_FILE="data/glove/glove.6B.100d.txt"
        EMBEDDING_DIM=100
    fi
fi

echo ""
echo "Ready to train with:"
echo "python src/nn_phishing_detection/scripts/train_word_bigru.py \\"
echo "    --data data/raw \\"
echo "    --glove $GLOVE_FILE \\"
echo "    --embedding_dim $EMBEDDING_DIM \\"
echo "    --epochs 10 \\"
echo "    --batch_size 32 \\"
echo "    --lr 0.001"
echo ""
echo "Usage:"
echo "  ./download_glove.sh        # Downloads 6B.100d"
echo "  ./download_glove.sh 840B   # Downloads 840B.300d"