#!/bin/bash

# Train the model using GPUs 1 and 2
echo "Starting training on GPUs 1 and 2..."
python run.py --is_training 1 \
               --model_id "custom_model" \
               --model "iTransformer" \
               --data custom \
               --root_path ./data/ \
               --data_path itransormer_train.csv \
               --features MS \
               --target Voltage \
               --seq_len 200 \
               --label_len 0 \
               --pred_len 1 \
               --enc_in 3 \
               --dec_in 3 \
               --c_out 1 \
               --use_gpu True \
               --use_multi_gpu True \
               --devices "1,2" \
               --gpu 1

# Test the model on the same configuration
echo "Starting testing on GPUs 1 and 2..."
python run.py --is_training 0 \
               --model_id "custom_model" \
               --model "iTransformer" \
               --data custom \
               --root_path ./data/ \
               --data_path itransormer_test.csv \
               --features MS \
               --target Voltage \
               --seq_len 200 \
               --label_len 0 \
               --pred_len 1 \
               --enc_in 3 \
               --dec_in 3 \
               --c_out 1 \
               --use_gpu True \
               --use_multi_gpu True \
               --devices "1,2" \
               --gpu 1
