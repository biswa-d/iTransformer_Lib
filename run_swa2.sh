#!/bin/bash

# Create Logs directory if it doesn't exist
mkdir -p Logs
mkdir -p ./run_outputs # Ensure base output dir exists

# Create timestamp and unique setting file path
RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
# Give a distinct model ID for the SWA run
MODEL_ID="custom_small_swa"
# Construct setting name here for logging and directory naming consistency
SETTING_NAME="${MODEL_ID}_${RUN_TIMESTAMP}" # Simplified name for single run
SETTING_FILE_PATH="Logs/setting_${SETTING_NAME}.txt"

# SHM setup (Optional, keep if needed)
SHM_DIR=/tmp/shm_dehuryb
mkdir -p "$SHM_DIR"
find "$SHM_DIR" -maxdepth 1 -type f -name "torch_*" -exec rm -f {} \;
find "$SHM_DIR" -maxdepth 1 -type f -name "nccl-*" -exec rm -f {} \;

# Set hyperparameters (Based on run_experiment_3_input_cosine.sh)
MODEL="iTransformer"
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="lg_train_noisy.csv"
TEST_DATA="lg_test.csv"
FEATURES="MS"
TARGET="Voltage"
SEQ_LEN=200
LABEL_LEN=0
PRED_LEN=1
ENC_IN=3
DEC_IN=3
C_OUT=1
D_MODEL=128
N_HEADS=4
E_LAYERS=2
D_LAYERS=1
D_FF=32 # Using the smaller D_FF from cosine script
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS=600 # SWA benefits from longer training
BATCH_SIZE=200
TEST_BATCH_SIZE=200 # Define test batch size
PATIENCE=300 # Early stopping based on pre-SWA validation loss
LEARNING_RATE=0.0008
DROPOUT=0.35
WEIGHT_DECAY=5e-5
USE_AMP=true # Set based on cosine script
USE_NORM=0

# --- Learning Rate Schedule Options ---
SCHEDULER='cosine'
LRADJ='none'
COSINE_T_MAX=$TRAIN_EPOCHS # Main scheduler T_max (will be adjusted internally if SWA starts early)
COSINE_ETA_MIN=0.000001
LR_DECAY_FACTOR=0.8 # Dummy
LR_DECAY_PERIOD=20 # Dummy

# --- SWA Parameters ---
USE_SWA=true # <<< Enable SWA >>>
SWA_START_FRAC=0.15 # Start SWA after 75% of epochs
SWA_LR=0.0004 # Specify SWA learning rate (can be None to use LEARNING_RATE)
SWA_ANNEAL_EPOCHS=30 # SWA LR annealing period

# --------------------------------------

# Log start time and parameters
echo "===== SWA Training Started at $(date) ====="
echo "Timestamp: $RUN_TIMESTAMP"
echo "Model ID: $MODEL_ID"
echo "Setting Name: $SETTING_NAME"
echo "Setting File: $SETTING_FILE_PATH"
echo "Model: $MODEL"
echo "Devices: $DEVICES"
echo "Epochs: $TRAIN_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "SWA Enabled: $USE_SWA (Start Frac: $SWA_START_FRAC, LR: $SWA_LR)"

# --- Train the model with SWA ---
echo "Starting training on GPUs $DEVICES..."
# Build the training command array
TRAIN_CMD=(
    python run.py --is_training 1
                   --run_timestamp "$RUN_TIMESTAMP" # Pass timestamp
                   --setting_file_path "$SETTING_FILE_PATH" # Path to log file for setting name
                   --model_id "$MODEL_ID" # Use the SWA model ID
                   --model "$MODEL"
                   --data "$DATA"
                   --root_path "$ROOT_PATH"
                   --data_path "$TRAIN_DATA"
                   --features "$FEATURES"
                   --target "$TARGET"
                   --seq_len "$SEQ_LEN"
                   --label_len "$LABEL_LEN"
                   --pred_len "$PRED_LEN"
                   --enc_in "$ENC_IN"
                   --dec_in "$DEC_IN"
                   --c_out "$C_OUT"
                   --d_model "$D_MODEL"
                   --n_heads "$N_HEADS"
                   --e_layers "$E_LAYERS"
                   --d_layers "$D_LAYERS"
                   --d_ff "$D_FF"
                   --moving_avg "$MOVING_AVG"
                   --factor "$FACTOR"
                   --devices "$DEVICES"
                   --train_epochs "$TRAIN_EPOCHS"
                   --batch_size "$BATCH_SIZE"
                   --patience "$PATIENCE"
                   --learning_rate "$LEARNING_RATE"
                   --dropout "$DROPOUT"
                   --weight_decay $WEIGHT_DECAY
                   --lradj $LRADJ
                   --lr_decay_factor $LR_DECAY_FACTOR
                   --lr_decay_period $LR_DECAY_PERIOD
                   --scheduler $SCHEDULER
                   --cosine_T_max $COSINE_T_MAX
                   --cosine_eta_min $COSINE_ETA_MIN
                   --use_norm $USE_NORM
                   --inverse
)

# Conditionally add AMP argument
if [ "$USE_AMP" = true ]; then
    TRAIN_CMD+=(--use_amp)
fi

# Conditionally add SWA arguments
if [ "$USE_SWA" = true ]; then
  TRAIN_CMD+=(--use_swa --swa_start_frac $SWA_START_FRAC --swa_anneal_epochs $SWA_ANNEAL_EPOCHS)
  # Only add --swa_lr if it's set (otherwise python default is None)
  if [ ! -z "$SWA_LR" ]; then
      TRAIN_CMD+=(--swa_lr $SWA_LR)
  fi
fi

# Execute training
"${TRAIN_CMD[@]}"

# Check training exit status
if [ $? -ne 0 ]; then
    echo "Error: SWA Training failed. Exiting." >&2
    exit 1
fi

echo "SWA Training finished."

# --- Test the final model (which should be SWA model if enabled) ---
echo "Starting testing on GPUs $DEVICES using setting from $SETTING_FILE_PATH..."
# Build the testing command array
TEST_CMD=(
    python run.py --is_training 0
                   --setting_file_path "$SETTING_FILE_PATH" # Use the same setting file to load correct model/setting
                   # <<< Model Definition Args (must match training) >>>
                   --model_id "$MODEL_ID"
                   --model "$MODEL"
                   --data "$DATA"
                   --root_path "$ROOT_PATH"
                   --data_path "$TEST_DATA" # Use test data
                   --features "$FEATURES"
                   --target "$TARGET"
                   --seq_len "$SEQ_LEN"
                   --label_len "$LABEL_LEN"
                   --pred_len "$PRED_LEN"
                   --enc_in "$ENC_IN"
                   --dec_in "$DEC_IN"
                   --c_out "$C_OUT"
                   --d_model "$D_MODEL"
                   --n_heads "$N_HEADS"
                   --e_layers "$E_LAYERS"
                   --d_layers "$D_LAYERS"
                   --d_ff "$D_FF"
                   --moving_avg "$MOVING_AVG"
                   --factor "$FACTOR"
                   --dropout "$DROPOUT"
                   # <<< Other Execution Args >>>
                   --devices "$DEVICES"
                   --batch_size "$TEST_BATCH_SIZE"
                   --use_norm $USE_NORM
                   --inverse
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

echo "Script finished."
