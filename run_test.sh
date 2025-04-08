#!/bin/bash

# --- Argument Parsing ---
# SETTING_FILE_PATH="Logs/setting_20250403_172133.txt" # Replaced by --output_path argument
OUTPUT_PATH="" # Will be set by argument
K_FOLDS=0    # Default for non-CV runs
FOLD=0       # Default for non-CV runs

# --- Configuration (Define defaults or allow overrides via arguments) ---
# Defaults set to match the specific training run: setting_20250329_094517
MODEL_ID="custom_small_2"
MODEL="iTransformer"
DATA="custom"
ROOT_PATH="./data/"
TEST_DATA="lg_test.csv" # Test data path used in the run
FEATURES="MS"
TARGET="Voltage" # Define the target variable
SEQ_LEN=200
LABEL_LEN=0
PRED_LEN=1
ENC_IN=3
DEC_IN=3
C_OUT=1
D_MODEL=32
N_HEADS=2
E_LAYERS=2
D_LAYERS=1
D_FF=128
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS=1
BATCH_SIZE=200
PATIENCE=20
LEARNING_RATE=0.0008
DROPOUT=0.35
# Other relevant parameters from the run (can be added to OTHER_ARGS if needed by run.py test mode)
# embed='timeF', activation='gelu', use_norm=True

# Array to hold arguments not explicitly handled here but needed by run.py
OTHER_ARGS=()
# Pass use_norm 0 to disable instance normalization, consistent with training setup
OTHER_ARGS+=(--use_norm 0)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # --setting_file_path) # Removed
        # SETTING_FILE_PATH="$2"
        # shift # past argument
        # shift # past value
        # ;;
        --output_path)
        OUTPUT_PATH="$2"
        shift; shift ;;
        --k_folds)
        K_FOLDS="$2"
        shift; shift ;;
        --fold)
        FOLD="$2"
        shift; shift ;;
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
if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --output_path argument is required." >&2
    echo "Usage: ./run_test.sh --output_path <path_to_job_or_fold_dir> [other_options...]" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_PATH" ]; then
    echo "Error: Output directory not found at $OUTPUT_PATH" >&2
    exit 1
fi

echo "Using output directory: $OUTPUT_PATH"
echo "Testing with Data Path: $TEST_DATA"

# --- SHM setup (Optional, uncomment if needed) ---
# SHM_DIR=/tmp/shm_dehuryb
# mkdir -p "$SHM_DIR"
# find "$SHM_DIR" -maxdepth 1 -type f -name "torch_*" -exec rm -f {} \;
# find "$SHM_DIR" -maxdepth 1 -type f -name "nccl-*" -exec rm -f {} \;

# --- Execute Testing ---
echo "Starting testing on GPUs $DEVICES using model from $OUTPUT_PATH..."
python run.py --is_training 0 \
               --output_path "$OUTPUT_PATH" \
               --k_folds "$K_FOLDS" \
               --fold "$FOLD" \
               --model_id "$MODEL_ID" \
               --model "$MODEL" \
               --data "$DATA" \
               --root_path "$ROOT_PATH" \
               --data_path "$TEST_DATA" \
               --features "$FEATURES" \
               --target "$TARGET" \
               --seq_len "$SEQ_LEN" \
               --label_len "$LABEL_LEN" \
               --pred_len "$PRED_LEN" \
               --enc_in "$ENC_IN" \
               --dec_in "$DEC_IN" \
               --c_out "$C_OUT" \
               --d_model "$D_MODEL" \
               --n_heads "$N_HEADS" \
               --e_layers "$E_LAYERS" \
               --d_layers "$D_LAYERS" \
               --d_ff "$D_FF" \
               --moving_avg "$MOVING_AVG" \
               --factor "$FACTOR" \
               --devices "$DEVICES" \
               --batch_size "$BATCH_SIZE" \
               --dropout "$DROPOUT" \
               "${OTHER_ARGS[@]}" # Pass any other args captured

echo "Testing script finished."
