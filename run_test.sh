#!/bin/bash

# --- Argument Parsing ---
SETTING_FILE_PATH=""

# --- Configuration (Define defaults or allow overrides via arguments) ---
# Defaults should ideally match common training parameters, but can be overridden.
MODEL_ID="custom_small" # Default from original script was "custom_model"
MODEL="iTransformer"    # Default matches original script
DATA="custom"           # Default matches original script
ROOT_PATH="./data/"     # Default updated from original './data/sample_data'
TEST_DATA="lg_test.csv" # Default test dataset, original had sample_data_test.csv
FEATURES="MS"           # Default matches original script
TARGET="Voltage"        # Default matches original script
SEQ_LEN=200             # Default updated from original 60
LABEL_LEN=0             # Default matches original script
PRED_LEN=1              # Default matches original script
ENC_IN=3                # Default matches original script
DEC_IN=3                # Default matches original script
C_OUT=1                 # Default matches original script
D_MODEL=128             # Default added
N_HEADS=2               # Default added
E_LAYERS=2              # Default added
D_LAYERS=1              # Default added
D_FF=512                # Default added
MOVING_AVG=25           # Default added
FACTOR=1                # Default added
DEVICES="0,1"           # Default matches original script
BATCH_SIZE=200          # Default added
DROPOUT=0.35            # Default added

# Array to hold arguments not explicitly handled here but needed by run.py
OTHER_ARGS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --setting_file_path)
        SETTING_FILE_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        --model_id)
        MODEL_ID="$2"
        shift; shift ;;
        --model)
        MODEL="$2"
        shift; shift ;;
        --data_path) # Allow overriding test data path
        TEST_DATA="$2"
        shift; shift ;;
        --root_path)
        ROOT_PATH="$2"
        shift; shift ;;
        --seq_len)
        SEQ_LEN="$2"
        shift; shift ;;
        --pred_len)
        PRED_LEN="$2"
        shift; shift ;;
        --enc_in)
        ENC_IN="$2"
        shift; shift ;;
        --dec_in)
        DEC_IN="$2"
        shift; shift ;;
        --c_out)
        C_OUT="$2"
        shift; shift ;;
        --d_model)
        D_MODEL="$2"
        shift; shift ;;
        --n_heads)
        N_HEADS="$2"
        shift; shift ;;
        --e_layers)
        E_LAYERS="$2"
        shift; shift ;;
        --d_layers)
        D_LAYERS="$2"
        shift; shift ;;
        --d_ff)
        D_FF="$2"
        shift; shift ;;
        --devices)
        DEVICES="$2"
        shift; shift ;;
        --batch_size)
        BATCH_SIZE="$2"
        shift; shift ;;
        --dropout)
        DROPOUT="$2"
        shift; shift ;;
        # Pass any other unrecognized arguments directly to run.py
        *)
        OTHER_ARGS+=("$1")
        if [[ "$2" != --* ]] && [[ ! -z "$2" ]]; then # Check if next is a value
          OTHER_ARGS+=("$2")
          shift
        fi
        shift
        ;;
    esac
done

# --- Validation ---
if [ -z "$SETTING_FILE_PATH" ]; then
    echo "Error: --setting_file_path argument is required." >&2
    echo "Usage: ./run_test.sh --setting_file_path <path_to_setting_file> [other_options...]" >&2
    exit 1
fi

if [ ! -f "$SETTING_FILE_PATH" ]; then
    echo "Error: Setting file not found at $SETTING_FILE_PATH" >&2
    exit 1
fi

echo "Using setting file: $SETTING_FILE_PATH"
echo "Testing with Data Path: $TEST_DATA"

# --- SHM setup (Optional, uncomment if needed) ---
# SHM_DIR=/tmp/shm_dehuryb
# mkdir -p "$SHM_DIR"
# find "$SHM_DIR" -maxdepth 1 -type f -name "torch_*" -exec rm -f {} \;
# find "$SHM_DIR" -maxdepth 1 -type f -name "nccl-*" -exec rm -f {} \;

# --- Execute Testing ---
echo "Starting testing on GPUs $DEVICES using setting from $SETTING_FILE_PATH..."
python run.py --is_training 0 \\
               --setting_file_path "$SETTING_FILE_PATH" \\
               --model_id "$MODEL_ID" \\
               --model "$MODEL" \\
               --data "$DATA" \\
               --root_path "$ROOT_PATH" \\
               --data_path "$TEST_DATA" \\
               --features "$FEATURES" \\
               --target "$TARGET" \\
               --seq_len "$SEQ_LEN" \\
               --label_len "$LABEL_LEN" \\
               --pred_len "$PRED_LEN" \\
               --enc_in "$ENC_IN" \\
               --dec_in "$DEC_IN" \\
               --c_out "$C_OUT" \\
               --d_model "$D_MODEL" \\
               --n_heads "$N_HEADS" \\
               --e_layers "$E_LAYERS" \\
               --d_layers "$D_LAYERS" \\
               --d_ff "$D_FF" \\
               --moving_avg "$MOVING_AVG" \\
               --factor "$FACTOR" \\
               --devices "$DEVICES" \\
               --batch_size "$BATCH_SIZE" \\
               --dropout "$DROPOUT" \\
               --inverse \\
               "${OTHER_ARGS[@]}" # Pass any other args captured

echo "Testing script finished."
