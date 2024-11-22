#!/bin/bash

# Set hyperparameters for training
MODEL_ID="custom_model_mlstm"
MODEL="mLSTM"  # Change model to mLSTM
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="train_sorted_tesla.csv"
TEST_DATA="test_sorted_tesla.csv"
FEATURES="MS"
TARGET="Voltage"
SEQ_LEN=200  # Reduced sequence length for compatibility
LABEL_LEN=0
PRED_LEN=1
ENC_IN=3  # Number of input features (Voltage, SOC, etc.)
C_OUT=1  # Output size (prediction of Voltage)
D_MODEL=16  # Hidden size in mLSTM
N_HEADS=4  # Number of heads for mLSTM
E_LAYERS=2  # Number of mLSTM layers
D_LAYERS=1
D_FF=512  # Reduced feedforward dimension for mLSTM
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS=5000
BATCH_SIZE=300  # Adjusted batch size for mLSTM
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
               --c_out $C_OUT \
               --d_model $D_MODEL \
               --n_heads $N_HEADS \
               --e_layers $E_LAYERS \
               --d_layers $D_LAYERS \
               --d_ff $D_FF \
               --moving_avg $MOVING_AVG \
               --factor $FACTOR \
               --devices $DEVICES
