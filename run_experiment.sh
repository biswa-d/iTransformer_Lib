#!/bin/bash

# Create Logs directory if it doesn't exist
mkdir -p Logs

# Create timestamped log file and run timestamp variable
RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="Logs/training_${RUN_TIMESTAMP}.log"

# SHM setup
SHM_DIR=/tmp/shm_dehuryb
mkdir -p "$SHM_DIR"
find "$SHM_DIR" -maxdepth 1 -type f -name "torch_*" -exec rm -f {} \;
find "$SHM_DIR" -maxdepth 1 -type f -name "nccl-*" -exec rm -f {} \;

# Hyperparameters (unchanged)
MODEL_ID="custom_model"
MODEL="iTransformer" #test
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="lg_train.csv"
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
D_FF=1024
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS=200
BATCH_SIZE=64
PATIENCE=20
LEARNING_RATE=0.0005
DROPOUT=0.35

# Log start time and parameters
echo "===== Training Started at $(date) =====" >> "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "Devices: $DEVICES" >> "$LOG_FILE"
echo "Epochs: $TRAIN_EPOCHS" >> "$LOG_FILE"
echo "Batch Size: $BATCH_SIZE" >> "$LOG_FILE"
echo "Learning Rate: $LEARNING_RATE" >> "$LOG_FILE"

# Train the model
echo "Starting training on GPUs $DEVICES..." >> "$LOG_FILE"
python run.py --is_training 1 \
               --run_timestamp $RUN_TIMESTAMP \
               --model_id $MODEL_ID \
               --model $MODEL \
               --data $DATA \
               --root_path $ROOT_PATH \
               --data_path $TRAIN_DATA \
               --features $FEATURES \
               --target $TARGET \
               --seq_len $SEQ_LEN \
               --label_len $LABEL_LEN \
               --pred_len $PRED_LEN \
               --enc_in $ENC_IN \
               --dec_in $DEC_IN \
               --c_out $C_OUT \
               --d_model $D_MODEL \
               --n_heads $N_HEADS \
               --e_layers $E_LAYERS \
               --d_layers $D_LAYERS \
               --d_ff $D_FF \
               --moving_avg $MOVING_AVG \
               --factor $FACTOR \
               --devices $DEVICES \
               --train_epochs $TRAIN_EPOCHS \
               --batch_size $BATCH_SIZE \
               --patience $PATIENCE \
               --learning_rate $LEARNING_RATE \
               --dropout $DROPOUT \
               --inverse >> "$LOG_FILE" 2>&1

# Log training completion
echo "===== Training Completed at $(date) =====" >> "$LOG_FILE"

# Test the model
echo "Starting testing on GPUs $DEVICES..." >> "$LOG_FILE"
python run.py --is_training 0 \
               --run_timestamp $RUN_TIMESTAMP \
               --model_id $MODEL_ID \
               --model $MODEL \
               --data $DATA \
               --root_path $ROOT_PATH \
               --data_path $TEST_DATA \
               --features $FEATURES \
               --target $TARGET \
               --seq_len $SEQ_LEN \
               --label_len $LABEL_LEN \
               --pred_len $PRED_LEN \
               --enc_in $ENC_IN \
               --dec_in $DEC_IN \
               --c_out $C_OUT \
               --d_model $D_MODEL \
               --n_heads $N_HEADS \
               --e_layers $E_LAYERS \
               --d_layers $D_LAYERS \
               --d_ff $D_FF \
               --moving_avg $MOVING_AVG \
               --factor $FACTOR \
               --devices $DEVICES \
               --train_epochs $TRAIN_EPOCHS \
               --batch_size $BATCH_SIZE \
               --patience $PATIENCE \
               --learning_rate $LEARNING_RATE \
               --dropout $DROPOUT \
               --inverse >> "$LOG_FILE" 2>&1

# Log final completion
echo "===== Testing Completed at $(date) =====" >> "$LOG_FILE"
echo "All outputs logged to $LOG_FILE"