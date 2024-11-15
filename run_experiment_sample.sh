#!/bin/bash

# Set hyperparameters for training
MODEL_ID="custom_model"
MODEL="iTransformer"
DATA="custom"
ROOT_PATH="./data/sample_data"
TRAIN_DATA="sample_data_train.csv"
TEST_DATA="sample_data_test.csv"
FEATURES="S"
TARGET="Voltage"
SEQ_LEN=60
LABEL_LEN=0
PRED_LEN=1
ENC_IN=3
DEC_IN=3
C_OUT=1
DEVICES="0,1"
TRAIN_EPOCHS=5
BATCH_SIZE=32
PATIENCE=3
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
               --label_len $LABEL_LEN \
               --pred_len $PRED_LEN \
               --enc_in $ENC_IN \
               --dec_in $DEC_IN \
               --c_out $C_OUT \
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
               --seq_len 1 \
               --label_len $LABEL_LEN \
               --pred_len $PRED_LEN \
               --enc_in $ENC_IN \
               --dec_in $DEC_IN \
               --c_out $C_OUT \
               --devices $DEVICES

