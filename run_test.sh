#!/bin/bash

# --- Configuration for a Specific SWA Test Run ---
# Setting file corresponding to the completed SWA training run
SETTING_FILE_PATH="Logs/setting_custom_small_swa_20250405_152727.txt"

# --- Parameters from the specific SWA training run --- 
# Derived from setting: custom_small_swa_iTransformer_custom_sl200_dm64_nh2_df16_ts20250405_152727
MODEL_ID="custom_small_swa" # Must match the training run's model_id
MODEL="iTransformer"
DATA="custom"
ROOT_PATH="./data/"
TEST_DATA="lg_test.csv" # The dataset to test on
FEATURES="MS"
TARGET="Voltage"
SEQ_LEN=200
LABEL_LEN=0 
PRED_LEN=1
ENC_IN=3    # Based on previous context
DEC_IN=3    # Based on previous context
C_OUT=1     # Based on previous context
D_MODEL=64  # From setting string
N_HEADS=2   # From setting string
E_LAYERS=2  # Assumed from previous context
D_LAYERS=1  # Assumed from previous context
D_FF=16     # From setting string
MOVING_AVG=25 # Assumed default
FACTOR=1    # Assumed default
DROPOUT=0.35 # Assumed from previous context
DEVICES="0,1" # Match training setup
TEST_BATCH_SIZE=200 # Batch size for testing
USE_AMP=true  # Match training setup (important if model saved with AMP)
USE_NORM=0    # Match training setup
INVERSE=true  # Match training setup

# --- Validation ---
if [ -z "$SETTING_FILE_PATH" ]; then
    echo "Error: SETTING_FILE_PATH is not set." >&2
    exit 1
fi

if [ ! -f "$SETTING_FILE_PATH" ]; then
    echo "Error: Setting file not found at $SETTING_FILE_PATH" >&2
    exit 1
fi

echo "Attempting to test model specified in: $SETTING_FILE_PATH"
echo "Testing with Data Path: $TEST_DATA"

# --- Execute Testing --- 
# The python script will read the setting name from the file,
# construct the path to the output directory, and load 'final_model.pth'
echo "Starting testing on GPUs $DEVICES..."

TEST_CMD=(
    python run.py --is_training 0 \
                   --setting_file_path "$SETTING_FILE_PATH" \
                   # --- Model Definition Args (must match training) --- \
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
                   --dropout "$DROPOUT" \
                   # <<< Other Execution Args >>> \
                   --devices "$DEVICES" \
                   --batch_size "$TEST_BATCH_SIZE" \
                   --use_norm $USE_NORM \
                   $( [[ "$INVERSE" == true ]] && echo "--inverse" ) \
                   # SWA args are NOT needed for testing
)

# Conditionally add AMP argument for testing if used in training
if [ "$USE_AMP" = true ]; then
    TEST_CMD+=(--use_amp)
fi

# Execute testing
"${TEST_CMD[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Testing failed." >&2
    exit 1
fi

echo "Testing script finished."
