#!/bin/bash

# --- Base Configuration (from run_experiment_3_input.sh) ---
K_FOLDS=5 # Number of folds
BASE_MODEL_ID="cv5_3input" # Base name for this CV run
MODEL="iTransformer"
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="sample_data_train.csv" # Use the training data file for CV
TEST_DATA="sample_data_test.csv" # Use the test data file for CV
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
D_FF=16 # NOTE: Using the small d_ff from your last runs
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS_PER_FOLD=1 # Epochs to train for each fold (adjust as needed)
BATCH_SIZE=200
NUM_WORKERS=10
PATIENCE=20 # Patience applies within each fold's training
LEARNING_RATE=0.0008
DROPOUT=0.35
WEIGHT_DECAY=1e-4
USE_AMP=True # Set based on previous discussion (using A100)
USE_NORM=0 # Set based on previous discussion

# --- Learning Rate Schedule Option (Set ONE block) ---

# Option 1: Cosine Annealing
SCHEDULER='cosine'
LRADJ='none'
# T_max should be epochs per fold for cosine schedule
COSINE_T_MAX=$TRAIN_EPOCHS_PER_FOLD
COSINE_ETA_MIN=0.0
LR_DECAY_FACTOR=0.8 # Dummy
LR_DECAY_PERIOD=20 # Dummy

# # Option 2: Periodic Exponential Decay (type1)
# SCHEDULER='none'
# LRADJ='type1'
# LR_DECAY_FACTOR=0.8
# LR_DECAY_PERIOD=20
# COSINE_T_MAX=$TRAIN_EPOCHS_PER_FOLD # Dummy
# COSINE_ETA_MIN=0.0 # Dummy

# # Option 3: Custom Step Decay (type2)
# SCHEDULER='none'
# LRADJ='type2'
# LR_DECAY_FACTOR=0.8 # Dummy
# LR_DECAY_PERIOD=20 # Dummy
# COSINE_T_MAX=$TRAIN_EPOCHS_PER_FOLD # Dummy
# COSINE_ETA_MIN=0.0 # Dummy

# --------------------------------------

# Create Logs directory if it doesn't exist
mkdir -p Logs

# --- K-Fold Loop ---
echo "Starting K-Fold Cross-Validation with K=$K_FOLDS"
echo "Base Model ID: $BASE_MODEL_ID"
echo "Epochs per Fold: $TRAIN_EPOCHS_PER_FOLD"

for (( k=0; k<$K_FOLDS; k++ ))
do
    echo ""
    echo "===== Starting Fold $k / $K_FOLDS ====="
    CURRENT_MODEL_ID="${BASE_MODEL_ID}_fold${k}"
    RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
    # Setting file name is specific to this fold's training run
    SETTING_FILE_PATH="Logs/setting_${CURRENT_MODEL_ID}_${RUN_TIMESTAMP}.txt"

    # Construct args for python call
    PYTHON_ARGS=(
        --is_training 1
        --k_folds $K_FOLDS
        --fold $k
        --model_id "$CURRENT_MODEL_ID"
        --run_timestamp "$RUN_TIMESTAMP"
        --setting_file_path "$SETTING_FILE_PATH"
        --model "$MODEL"
        --data "$DATA"
        --root_path "$ROOT_PATH"
        --data_path "$TRAIN_DATA" # Always use training data for CV
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
        --train_epochs "$TRAIN_EPOCHS_PER_FOLD" # Use epochs per fold
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
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

    # Conditionally add --use_amp if enabled
    if [ "$USE_AMP" = True ]; then
      PYTHON_ARGS+=(--use_amp)
    fi

    # Run training for this fold
    echo "Running command: python run.py ${PYTHON_ARGS[@]}"
    python run.py "${PYTHON_ARGS[@]}"

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: Fold $k failed. Exiting." >&2
        exit 1
    fi
done

echo ""
echo "===== K-Fold Training Completed ====="
echo "Results for each fold saved in ./run_outputs/${BASE_MODEL_ID}_fold*/"
echo "Check metrics_summary.txt in each fold directory."
echo "Remember to run testing separately on the best fold or a model retrained on all data."
