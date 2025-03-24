#!/bin/bash

# Store the script directory (experiments) and root directory (unmemorize)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
ORIGINAL_DIR="$(pwd)"
CONFIG_ROOT="$ROOT_DIR/config"
DEBUG=false

# datasets
SYNTHETIC_DATA="data/synthetic"
SYNTHETIC_DATA100="data/synthetic100"
PRETRAIN_DATA="data/organic/dataset"
PRETRAIN_LABELS="data/organic/labels.txt"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true ;;
        *) CONFIG_FILE="$ORIGINAL_DIR/$1" ;;
    esac
    shift
done

if [ "$DEBUG" = true ]; then
    echo "=== Debug Information ==="
    echo "SCRIPT_DIR: $SCRIPT_DIR"
    echo "ROOT_DIR: $ROOT_DIR"
    echo "ORIGINAL_DIR: $ORIGINAL_DIR"
    echo "CONFIG_ROOT: $CONFIG_ROOT"
    echo "CONFIG_FILE: $CONFIG_FILE"
    echo "======================="
    echo ""
fi

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

MODEL_CONFIG_FILE="$ROOT_DIR/config/model_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: Model config file $MODEL_CONFIG_FILE not found"
    exit 1
fi

# Function to get dataset path based on type
get_dataset_path() {
    local dataset=$1
    if [ "$dataset" = "synthetic" ]; then
        echo "$SYNTHETIC_DATA"
    elif [ "$dataset" = "synthetic100" ]; then
       echo "$SYNTHETIC_DATA100"
    else
        echo "$PRETRAIN_DATA"
    fi
}

# Function to execute or debug a command
execute_command() {
    local cmd="$1"
    if [ "$DEBUG" = true ]; then
        echo "Would execute: $cmd"
    else
        eval "$cmd"
    fi
}

# Function to run experiment
run_experiment() {
    local model_name=$1
    local config=$2
    local sample_count=$3
    local exp_type=$4
    local smart_flag=$5
    local dataset=$6
    local base_dir=$7
    local top_k=$8 

    local dataset_path=$(get_dataset_path "$dataset")
    
    echo "Running experiment with following parameters:"
    echo "- Model: $model_name"
    echo "- Model Folder: $python_model_folder"
    echo "- Config: $config"
    echo "- Sample Count: $sample_count"
    echo "- Experiment Type: $exp_type"
    echo "- Dataset: $dataset"
    echo "- Top K: $top_k"
    echo "=================================================="
    
    local run_dir="$base_dir/$exp_type/$dataset/$model_name/$top_k/$config/0"
    if [ "$DEBUG" = true ]; then
        echo "Would create directory: $run_dir"
    else
        mkdir -p "$run_dir"
    fi
    
    # Logging folder is without the config and run number
    local logging_folder="$base_dir/$exp_type/$dataset/$model_name/$top_k"
    
    local python_model_name="$model_name"  # Always use short name
    local python_model_folder
    
    if [[ "$dataset" =~ "synthetic" ]]; then
        # For synthetic, model_folder is path to memorized model
        local parent_dir=$(dirname "$base_dir")
        python_model_folder="$parent_dir/$model_name/memorized/model"
    else
        # For pretrained, model_folder is the expanded model name from config
        python_model_folder=$(jq -r --arg model "$model_name" '.[$model].model_name' "$MODEL_CONFIG_FILE")
    fi
    
    local cmd="cd $ROOT_DIR && python ./src/unmemorizerun.py \
        --model_name \"$python_model_name\" \
        --logging_folder \"$logging_folder\" \
        --model_folder \"$python_model_folder\" \
        --dataset \"$dataset_path\" \
        --run_config \"$CONFIG_ROOT/runs/$config.json\" \
        --num_runs 1 \
        --unmemorize_sample_count \"$sample_count\" \
        $([[ "$smart_flag" == "--smart_select" ]] && echo "--smart_select") \
        $top_k_param"       
    
    execute_command "$cmd"
    execute_command "cd $ORIGINAL_DIR"
}

# Function to generate plots
generate_plots() {
    local model_name=$1
    local config=$2
    local exp_type=$3
    local dataset=$4
    local base_dir=$5
    local sample_count=$6
    
    local suffix=""
    if [ "$exp_type" == "smart" ]; then
        suffix=" (Smart)"
    fi
    
    echo "Generating plots..."
    local flags=""
    
    # Set up flags for both single and multi-sample cases
    if [[ "$dataset" =~ "synthetic" ]]; then
        local parent_dir=$(dirname "$base_dir")
        local memorized_log="$parent_dir/$model_name/memorized/test.log"
        local memorized_greedy_log="$parent_dir/$model_name/memorized/test_greedy.log"
        
        flags="--memorized \"$memorized_log\""
        if [ -f "$memorized_greedy_log" ]; then
            flags="$flags --memorized_greedy \"$memorized_greedy_log\""
        fi
    else
        # For pretrained dataset
        local parent_dir=$(dirname "$base_dir")
        local pretrained_log="$parent_dir/$model_name/pretrained-pretrained/test.log"
        local pretrained_greedy_log="$parent_dir/$model_name/pretrained-pretrained/test_greedy.log"
        
        flags="--pretrained \"$pretrained_log\" --sample-labels \"$PRETRAIN_LABELS\""
        if [ -f "$pretrained_greedy_log" ]; then
            flags="$flags --pretrained_greedy \"$pretrained_greedy_log\""
        fi
    fi
    
    # For single sample experiments, use the runs_folder parameter
    if [ "$sample_count" == "1" ]; then
        local exp_dir="$base_dir/$exp_type/$dataset/$model_name/$top_k"
        local cmd="cd $ROOT_DIR && python ./src/create_multisample_plots.py \
            --runs_folder \"$exp_dir\" \
            --output \"$exp_dir\" \
            --title \"$model_name $dataset$suffix\" \
            $flags"
        
        execute_command "$cmd"
        execute_command "cd $ORIGINAL_DIR"
        return
    fi
    
    # Multi-sample plot generation code
    # Add input file and its greedy version if it exists
    local input_log="$base_dir/$exp_type/$dataset/$model_name/$top_k/$config/0/test.log"
    local input_greedy_log="${input_log%.*}_greedy.log"
    
    local cmd="cd $ROOT_DIR && python ./src/create_multisample_plots.py \
        --input \"$input_log\" \
        --input_greedy \"$input_greedy_log\" \
        --output \"$base_dir/$exp_type/$dataset/$model_name/$top_k\" \
        --title \"$model_name $dataset$suffix\" \
        $flags"
    
    execute_command "$cmd"
    execute_command "cd $ORIGINAL_DIR"
}

# Main execution
if [ "$DEBUG" = true ]; then
    echo "Reading config file contents:"
    cat "$CONFIG_FILE"
    echo ""
    echo "Attempting to parse base_directory:"
    jq -r '.base_directory' "$CONFIG_FILE" || echo "Failed to parse base_directory"
    echo ""
fi

# Read base directory from config file
BASE_DIR=$(jq -r '.base_directory' "$CONFIG_FILE")
if [ -z "$BASE_DIR" ]; then
    echo "Error: Could not read base_directory from config file"
    exit 1
fi

if [ "$DEBUG" = true ]; then
    echo "Base directory: $BASE_DIR"
    echo ""
    echo "Config file structure:"
    jq '.' "$CONFIG_FILE"
    echo ""
fi

# Iterate through each model in the config
jq -c '.experiments[]' "$CONFIG_FILE" | while read -r model; do
    MODEL_NAME=$(echo "$model" | jq -r '.model_name')
    
    # Iterate through each configuration for the model
    echo "$model" | jq -c '.configurations[]' | while read -r config; do
        CONFIG=$(echo "$config" | jq -r '.config')
        SAMPLE_COUNT=$(echo "$config" | jq -r '.sample_count')
        
        # Iterate through experiment types
        echo "$config" | jq -c '.experiment_types[]' | while read -r exp_type; do
            EXP_NAME=$(echo "$exp_type" | jq -r '.name')
            SMART_SELECT=$(echo "$exp_type" | jq -r '.smart_select')
            DATASET_TYPE=$(echo "$exp_type" | jq -r '.data')
            TOP_K=$(echo "$exp_type" | jq -r '.top_k // empty')

            # Set smart flag based on configuration
            SMART_FLAG=""
            if [ "$SMART_SELECT" = "true" ]; then
                SMART_FLAG="--smart_select"
            fi
            
            run_experiment \
                "$MODEL_NAME" \
                "$CONFIG" \
                "$SAMPLE_COUNT" \
                "$EXP_NAME" \
                "$SMART_FLAG" \
                "$DATASET_TYPE" \
                "$BASE_DIR" \
                "$TOP_K"
                
            generate_plots \
                            "$MODEL_NAME" \
                            "$CONFIG" \
                            "$EXP_NAME" \
                            "$DATASET_TYPE" \
                            "$BASE_DIR" \
                            "$SAMPLE_COUNT" 
                
            echo ""
        done
    done
done