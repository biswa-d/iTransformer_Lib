#!/bin/bash

# Create Logs directory if it doesn't exist
mkdir -p Logs

# Create timestamp and unique setting file path
RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SETTING_FILE_PATH="Logs/setting_${RUN_TIMESTAMP}.txt" # Unique setting file per run

# SHM setup (Optional, keep if needed)
SHM_DIR=/tmp/shm_dehuryb
mkdir -p "$SHM_DIR"
find "$SHM_DIR" -maxdepth 1 -type f -name "torch_*" -exec rm -f {} \;
find "$SHM_DIR" -maxdepth 1 -type f -name "nccl-*" -exec rm -f {} \;

# Set hyperparameters for training
MODEL_ID="custom_small"
MODEL="iTransformer" #test
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
WEIGHT_DECAY=1e-4 # Define weight decay variable
LR_DECAY_FACTOR=0.8 # LR decay factor for type1 schedule
LR_DECAY_PERIOD=20  # LR decay period (epochs) for type1 schedule

# Log start time and parameters
echo "===== Training Started at $(date) ====="
echo "Timestamp: $RUN_TIMESTAMP"
echo "Setting File: $SETTING_FILE_PATH"
echo "Model: $MODEL"
echo "Devices: $DEVICES"
echo "Epochs: $TRAIN_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"

# Train the model
echo "Starting training on GPUs $DEVICES..."
python run.py --is_training 1 \
               --run_timestamp "$RUN_TIMESTAMP" \
               --setting_file_path "$SETTING_FILE_PATH" \
               --model_id "$MODEL_ID" \
               --model "$MODEL" \
               --data "$DATA" \
               --root_path "$ROOT_PATH" \
               --data_path "$TRAIN_DATA" \
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
               --train_epochs "$TRAIN_EPOCHS" \
               --batch_size "$BATCH_SIZE" \
               --patience "$PATIENCE" \
               --learning_rate "$LEARNING_RATE" \
               --dropout "$DROPOUT" \
               --weight_decay $WEIGHT_DECAY \
               --lr_decay_factor $LR_DECAY_FACTOR \
               --lr_decay_period $LR_DECAY_PERIOD \
               --use_norm 0 \
               --inverse

echo "Training finished."

# Test the model
echo "Starting testing on GPUs $DEVICES using setting from $SETTING_FILE_PATH..."
python run.py --is_training 0 \
               --setting_file_path "$SETTING_FILE_PATH" \
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
               --train_epochs "$TRAIN_EPOCHS" \
               --batch_size "$BATCH_SIZE" \
               --patience "$PATIENCE" \
               --learning_rate "$LEARNING_RATE" \
               --dropout "$DROPOUT" \
               --weight_decay $WEIGHT_DECAY \
               --lr_decay_factor $LR_DECAY_FACTOR \
               --lr_decay_period $LR_DECAY_PERIOD \
               --use_norm 0 \
               --inverse

echo "Script finished."