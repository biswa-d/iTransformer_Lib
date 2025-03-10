#!/bin/bash

# Set hyperparameters for training
MODEL_ID="custom"
MODEL="iTransformer" #test
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="lg_train.csv"
TEST_DATA="lg_valid.csv"
FEATURES="MS"
TARGET="Voltage"
SEQ_LEN=200
LABEL_LEN=0
PRED_LEN=1
ENC_IN=4
DEC_IN=4
C_OUT=1
D_MODEL=512
N_HEADS=4
E_LAYERS=3
D_LAYERS=1
D_FF=1024
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS=1000
BATCH_SIZE=200
PATIENCE=100
LEARNING_RATE=0.0001
DROPOUT=0.35


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
               --dropout $DROPOUT

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
               --dropout $DROPOUT
