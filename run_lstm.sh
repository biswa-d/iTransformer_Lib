#!/bin/bash

# Set hyperparameters for training
MODEL_ID="custom_model"
MODEL="mLSTM"  # Change model to LSTM for comparison
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="train_scaled_tesla.csv"
TEST_DATA="test_scaled_tesla.csv"
FEATURES="MS"
TARGET="Voltage"
SEQ_LEN=400
LABEL_LEN=0
PRED_LEN=1
ENC_IN=3  # Number of input features (Voltage, SOC, etc.)
D_MODEL=16  # Size of hidden layer in LSTM
C_OUT=1  # Output size (prediction of Voltage)
E_LAYERS=2  # Number of LSTM layers
DEVICES="0,1"
TRAIN_EPOCHS=5000
BATCH_SIZE=300
PATIENCE=100
LEARNING_RATE=0.0001

# Train the model
echo "Starting training on GPUs $DEVICES..."
python run.py --is_training 1 \
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
               --d_model $D_MODEL \
               --c_out $C_OUT \
               --e_layers $E_LAYERS \
               --devices $DEVICES \
               --train_epochs $TRAIN_EPOCHS \
               --batch_size $BATCH_SIZE \
               --patience $PATIENCE \
               --learning_rate $LEARNING_RATE

# Test the model
echo "Starting testing on GPUs $DEVICES..."
python run.py --is_training 0 \
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
               --d_model $D_MODEL \
               --c_out $C_OUT \
               --e_layers $E_LAYERS \
               --devices $DEVICES
