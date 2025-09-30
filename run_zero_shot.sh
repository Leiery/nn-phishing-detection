#!/bin/bash

# Zero-shot classification for phishing detection (OpenAI & Mistral)

echo "Zero-shot AI Classification for Phishing Detection"
echo "============================================================================"
echo ""

# Check if splits exist from CNN training
if [ ! -f "runs/cnn_bigru_word_tuned_patience_3_with_vocab_pkl/splits/test_split.csv" ]; then
    echo "Error: Test split not found!"
    echo "Looking for: $(pwd)/runs/cnn_bigru_word_tuned_patience_3_with_vocab_pkl/splits/test_split.csv"
    echo ""
    echo "Available files in runs/:"
    ls -la runs/ 2>/dev/null || echo "  runs/ directory not found"
    echo ""
    echo "Please run ./train_optimal.sh first to generate the data splits"
    echo "Or ensure you're running from the project root directory"
    exit 1
fi

# Default settings
PROVIDER="openai"
MODEL=""
USE_BATCH=false
MAX_SAMPLES=""
MODE="test"
NORMALIZE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch)
            USE_BATCH=true
            shift
            ;;
        --quick)
            MAX_SAMPLES="--max-samples 10"
            echo "Quick mode: Testing with 10 samples only"
            shift
            ;;
        --openai)
            PROVIDER="openai"
            shift
            ;;
        --mistral)
            PROVIDER="mistral"
            shift
            ;;
        --normalize)
            NORMALIZE=true
            shift
            ;;
        --no-normalize)
            NORMALIZE=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --provider PROVIDER    AI provider: openai or mistral (default: openai)"
            echo "  --model MODEL         Specific model to use"
            echo "  --batch               Use Batch API (50% cheaper, up to 24h processing)"
            echo "  --quick               Test with 10 samples only"
            echo "  --openai              Shorthand for --provider openai"
            echo "  --mistral             Shorthand for --provider mistral"
            echo "  --normalize           Apply text normalization (default: enabled)"
            echo "  --no-normalize        Disable text normalization"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --openai --batch                    # OpenAI batch API with normalization"
            echo "  $0 --mistral --model mistral-medium-latest    # Mistral with specific model"
            echo "  $0 --provider mistral --quick          # Quick test with Mistral"
            echo "  $0 --mistral --no-normalize            # Mistral without text normalization"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default models if not specified
if [ -z "$MODEL" ]; then
    if [ "$PROVIDER" = "openai" ]; then
        MODEL="gpt-4o-mini"
    elif [ "$PROVIDER" = "mistral" ]; then
        MODEL="mistral-medium-latest"
    fi
fi

# Check API key based on provider
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable not set!"
        echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    API_KEY_VAR="OPENAI_API_KEY"
elif [ "$PROVIDER" = "mistral" ]; then
    if [ -z "$MISTRAL_API_KEY" ]; then
        echo "Error: MISTRAL_API_KEY environment variable not set!"
        echo "Please set it with: export MISTRAL_API_KEY='your-key-here'"
        exit 1
    fi
    API_KEY_VAR="MISTRAL_API_KEY"
else
    echo "Error: Unsupported provider '$PROVIDER'"
    echo "Supported providers: openai, mistral"
    exit 1
fi

# Set data path and output directory
TEST_DATA="runs/cnn_bigru_word_tuned_patience_3_with_vocab_pkl/splits/test_split.csv"

# Build output directory name based on settings
NORM_SUFFIX=""
if [ "$NORMALIZE" = true ]; then
    NORM_SUFFIX="_normalized"
else
    NORM_SUFFIX="_raw"
fi

if [ "$USE_BATCH" = true ]; then
    OUTPUT_DIR="runs/zero_shot_${PROVIDER}_batch${NORM_SUFFIX}"
else
    OUTPUT_DIR="runs/zero_shot_${PROVIDER}_sequential${NORM_SUFFIX}"
fi

echo ""
echo "Configuration:"
echo "  - Provider: $PROVIDER"
echo "  - Model: $MODEL"
echo "  - Data: Test split"
echo "  - Text Normalization: $NORMALIZE"
if [ "$USE_BATCH" = true ]; then
    echo "  - API Mode: Batch API (50% cheaper, up to 24h processing)"
else
    echo "  - API Mode: Sequential"
fi
echo "  - Output: $OUTPUT_DIR"
echo "  - API Key: $API_KEY_VAR is set"
echo ""

# Confirm expensive operation (skip for quick mode or batch)
if [ "$USE_BATCH" = false ] && [ -z "$MAX_SAMPLES" ]; then
    read -p "Continue with full sequential processing? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborted."
        echo "Try --batch for cheaper processing or --quick for testing"
        exit 0
    fi
fi

# Build command arguments
CMD_ARGS="--test-set $TEST_DATA --provider $PROVIDER --model $MODEL --output-dir $OUTPUT_DIR"
if [ "$USE_BATCH" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-batch"
fi
if [ "$NORMALIZE" = false ]; then
    CMD_ARGS="$CMD_ARGS --no-normalize"
fi
if [ -n "$MAX_SAMPLES" ]; then
    CMD_ARGS="$CMD_ARGS $MAX_SAMPLES"
fi

# Run zero-shot classifier
if [ "$USE_BATCH" = true ]; then
    echo "Starting $PROVIDER Batch API job..."
    python src/nn_phishing_detection/zero_shot/zs_classifier.py $CMD_ARGS

    echo ""
    echo "================================================"
    echo "Batch job submitted to $PROVIDER!"
    echo "Processing time: Up to 24 hours"
    echo "Check results at: $OUTPUT_DIR/"
    echo ""
    if [ "$PROVIDER" = "openai" ]; then
        echo "Monitor status with OpenAI API calls or wait for completion"
    elif [ "$PROVIDER" = "mistral" ]; then
        echo "Monitor status with Mistral API calls or wait for completion"
    fi
else
    echo "Running $PROVIDER sequential classification..."
    python src/nn_phishing_detection/zero_shot/zs_classifier.py $CMD_ARGS

    echo ""
    echo "================================================"
    echo "Zero-shot classification complete!"
    echo "Results saved to: $OUTPUT_DIR/"
fi