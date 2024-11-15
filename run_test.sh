#!/bin/bash

# Test the trained model on new data
echo "Starting testing on GPUs 1 and 2..."
python run.py --is_training 0 \
               --model_id "custom_model" \
               --model "iTransformer" \
               --data custom \
               --root_path ./data/sample_data \
               --data_path sample_data_test.csv \  # Specify your new test data file
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
