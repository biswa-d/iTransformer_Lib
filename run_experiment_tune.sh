#!/bin/bash

# Set hyperparameters for training
MODEL_ID="custom_model"
MODEL="iTransformer"  # Change model to LSTM for comparison
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="train_sorted_tesla.csv"
TEST_DATA="test_sorted_tesla.csv"
FEATURES="MS"
TARGET="Voltage"
SEQ_LEN=200
LABEL_LEN=0
PRED_LEN=1
ENC_IN=3
DEC_IN=3
C_OUT=1
D_MODEL=32
N_HEADS=16
E_LAYERS=2
D_LAYERS=1
D_FF=2048
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS=5000
BATCH_SIZE=500
PATIENCE=25
LEARNING_RATE=0.001

# Hyperparameter tuning and training
echo "Starting hyperparameter tuning and training on GPUs $DEVICES..."
python run_tune.py --is_training 1 \  # <-- This line is added
                   --tune_and_train 1 \
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
                   --learning_rate $LEARNING_RATE

# Test the best model with the best hyperparameters found
echo "Starting testing on GPUs $DEVICES..."
python run_tune.py --test_best 1 \
                   --is_training 0 \  # <-- This line is added
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
                   --devices $DEVICES