#!/bin/bash

# Parse command line arguments
DEBUG=false
CONFIG_FILE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true ;;
        *) CONFIG_FILE="$1" ;;
    esac
    shift
done

# Function to execute or debug a command
execute_command() {
    local cmd="$1"
    if [ "$DEBUG" = true ]; then
        echo "Would execute: $cmd"
    else
        eval "$cmd"
    fi
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 [--debug] <config_file.json>"
    exit 1
fi

MODEL_CONFIG_FILE="../config/model_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: Model config file $MODEL_CONFIG_FILE not found"
    exit 1
fi

# Read base configuration
MODEL_KEY=$(jq -r '.model_name' "$CONFIG_FILE")
BASE_DIR=$(jq -r '.base_directory' "$CONFIG_FILE")
PRETRAIN_DATASET=$(jq -r '.pretrain_dataset' "$CONFIG_FILE")
SYNTHETIC_DATASET=$(jq -r '.synthetic_dataset' "$CONFIG_FILE")
SAMPLE_COUNT=$(jq -r '.sample_count' "$CONFIG_FILE")

if [ "$DEBUG" = true ]; then
    echo "=== Debug Information ==="
    echo "MODEL_KEY: $MODEL_KEY"
    echo "BASE_DIR: $BASE_DIR"
    echo "PRETRAIN_DATASET: $PRETRAIN_DATASET"
    echo "SYNTHETIC_DATASET: $SYNTHETIC_DATASET"
    echo "SAMPLE_COUNT: $SAMPLE_COUNT"
    echo "======================="
    echo ""
fi

# Run memorization benchmark
echo "Running memorization benchmark..."
echo "================================"

MEMORIZE_DIR="$BASE_DIR/$MODEL_KEY"
execute_command "cd .."
execute_command "python ./src/unmemorizerun.py \
    --model_name \"$MODEL_KEY\" \
    --logging_folder \"$MEMORIZE_DIR\" \
    --dataset \"$SYNTHETIC_DATASET\" \
    --benchmark-memorized \
    --unmemorize_sample_count \"$SAMPLE_COUNT\""

# Run pretrained benchmark
echo ""
echo "Running pretrained synthetic benchmark..."
echo "=================================="

SYNTHETIC_PRETRAIN_DIR="$BASE_DIR/$MODEL_KEY/pretrained-synthetic"
execute_command "python ./src/unmemorizerun.py \
    --model_name \"$MODEL_KEY\" \
    --logging_folder \"$SYNTHETIC_PRETRAIN_DIR\" \
    --dataset \"$SYNTHETIC_DATASET\" \
    --benchmark-pretrained \
    --unmemorize_sample_count \"$SAMPLE_COUNT\""

# Run pretrained benchmark
echo ""
echo "Running pretrained pretrain test..."
echo "=================================="

PRETRAIN_DIR="$BASE_DIR/$MODEL_KEY/pretrained-pretrained"
execute_command "python ./src/unmemorizerun.py \
    --model_name \"$MODEL_KEY\" \
    --logging_folder \"$PRETRAIN_DIR\" \
    --dataset \"$PRETRAIN_DATASET\" \
    --test-pretrained \
    --unmemorize_sample_count \"$SAMPLE_COUNT\""

execute_command "cp experiments/$CONFIG_FILE $MEMORIZE_DIR"

echo ""
echo "Memorization completed:"
echo "- Memorization results in:         $MEMORIZE_DIR"
echo "- Pretrained synthetic results in: $SYNTHETIC_PRETRAIN_DIR"
echo "- Pretrained pretrain results in:  $PRETRAIN_DIR"