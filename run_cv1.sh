#!/bin/bash

# --- Base Configuration (from run_experiment_3_input.sh) ---
K_FOLDS=5 # Number of folds
BASE_MODEL_ID="cv5_3input" # Base name for this CV run
MODEL="iTransformer"
DATA="custom"
ROOT_PATH="./data/"
TRAIN_DATA="lg_train.csv" # Use the training data file for CV
TEST_DATA="lg_test.csv" # Use the actual test dataset for final evaluation
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
E_LAYERS=1
D_LAYERS=1
D_FF=16
MOVING_AVG=25
FACTOR=1
DEVICES="0,1"
TRAIN_EPOCHS_PER_FOLD=1 # <<< Set back to 1 for quick test
BATCH_SIZE=200
TEST_BATCH_SIZE=200
NUM_WORKERS=10
PATIENCE=20
LEARNING_RATE=0.0008
DROPOUT=0.35
WEIGHT_DECAY=1e-4
USE_AMP=True
USE_NORM=0
INVERSE=true

# <<< Warmup Option >>>
LR_WARMUP_EPOCHS=5 # Set to 0 to disable

# <<< Custom Multi-Phase Schedule Params >>>
MAIN_DECAY_EPOCHS=100 # Number of epochs for initial cosine decay after warmup (Set > 0 to enable custom)
EXPLOIT_LR=0.0002     # Starting LR for exploitation cycles (defaults to min_lr if None/empty)
EXPLOIT_CYCLE_EPOCHS=20 # Length of each exploitation cycle

# --- Learning Rate Schedule Option (Set ONE block) ---
SCHEDULER='cosine'
LRADJ='none'
COSINE_T_MAX=$TRAIN_EPOCHS_PER_FOLD
COSINE_ETA_MIN=0.0
LR_DECAY_FACTOR=0.8 # Dummy
LR_DECAY_PERIOD=20 # Dummy

# --------------------------------------

# <<< Generate a timestamp for the entire CV run >>>
CV_RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# <<< Create the main output directory for this CV run >>>
CV_RUN_DIR="./run_cv/run_${CV_RUN_TIMESTAMP}_${BASE_MODEL_ID}"
mkdir -p "$CV_RUN_DIR"
mkdir -p Logs # Ensure Logs dir exists for setting files

echo "Starting K-Fold Cross-Validation with K=$K_FOLDS"
echo "Main Output Directory: $CV_RUN_DIR"
echo "Base Model ID: $BASE_MODEL_ID"
echo "Epochs per Fold: $TRAIN_EPOCHS_PER_FOLD"
echo "Test Data: $TEST_DATA"

FOLD_DIRS=() # Array to store fold output directories relative to CV_RUN_DIR

for (( k=0; k<$K_FOLDS; k++ ))
do
    echo ""
    echo "===== Starting Fold $k / $K_FOLDS ====="
    CURRENT_FOLD_ID="${BASE_MODEL_ID}_fold${k}"
    # Timestamp for this specific fold's execution (can be useful for logs)
    FOLD_RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

    # Setting name incorporates fold ID and fold timestamp
    SETTING_NAME="${CURRENT_FOLD_ID}_${MODEL}_${DATA}_sl${SEQ_LEN}_dm${D_MODEL}_nh${N_HEADS}_df${D_FF}_ts${FOLD_RUN_TIMESTAMP}"

    # Log file path (remains in the central Logs directory)
    SETTING_FILE_PATH="Logs/setting_${SETTING_NAME}.txt"

    # <<< Output directory for this specific fold run (inside the main CV_RUN_DIR) >>>
    FOLD_OUTPUT_DIR="${CV_RUN_DIR}/${SETTING_NAME}"
    FOLD_DIRS+=("$SETTING_NAME") # Store relative name for summary later

    # --- Training Phase ---
    echo "--- Training Fold $k ---"
    # Construct training args
    TRAIN_ARGS=(
        --is_training 1
        --k_folds $K_FOLDS
        --fold $k
        --model_id "$CURRENT_FOLD_ID" # Use fold-specific ID
        --run_timestamp "$FOLD_RUN_TIMESTAMP" # Pass fold-specific timestamp
        --setting_file_path "$SETTING_FILE_PATH" # Path to log file
        --model "$MODEL"
        --data "$DATA"
        --root_path "$ROOT_PATH"
        --data_path "$TRAIN_DATA" # Use training data
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
        --train_epochs "$TRAIN_EPOCHS_PER_FOLD"
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
        $( [[ "$INVERSE" == true ]] && echo "--inverse" )
        --optimizer adamw
        --lr_warmup_epochs "$LR_WARMUP_EPOCHS"
        --main_decay_epochs "$MAIN_DECAY_EPOCHS"
        $( [ ! -z "$EXPLOIT_LR" ] && echo "--exploit_lr $EXPLOIT_LR" )
        --exploit_cycle_epochs "$EXPLOIT_CYCLE_EPOCHS"
    )
    if [ "$USE_AMP" = True ]; then TRAIN_ARGS+=(--use_amp); fi

    echo "Running training command (Output Dir: $FOLD_OUTPUT_DIR)..."
    # NOTE: The python script determines the final output path based on the setting name
    python run.py "${TRAIN_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "Error: Training Fold $k failed. Exiting." >&2
        exit 1
    fi
    # Verify the directory was created where expected by the python script
    # The python script uses os.path.join(base_output_dir, setting)
    # base_output_dir is ./run_cv/ because args.k_folds > 0
    # setting is SETTING_NAME
    ACTUAL_FOLD_OUTPUT_DIR="./run_cv/${SETTING_NAME}" # Path constructed by python script
    if [ -d "$ACTUAL_FOLD_OUTPUT_DIR" ]; then
        echo "Training Fold $k finished. Checkpoint saved in $ACTUAL_FOLD_OUTPUT_DIR"
    else
        echo "Error: Expected output directory $ACTUAL_FOLD_OUTPUT_DIR not found after training Fold $k." >&2
        exit 1
    fi


    # --- Testing Phase (using the checkpoint from this fold) ---
    echo "--- Testing Fold $k Model on $TEST_DATA ---"
    # Construct testing args (use the same setting file path to load the correct setting)
    TEST_ARGS=(
        --is_training 0
        --setting_file_path "$SETTING_FILE_PATH" # Use the setting file saved by training
        --k_folds 0 # Testing doesn't use k-fold data splitting
        --fold 0    # Irrelevant for testing
        --model_id "$CURRENT_FOLD_ID" # Keep model ID consistent
        --model "$MODEL"
        --data "$DATA"
        --root_path "$ROOT_PATH"
        --data_path "$TEST_DATA" # <<< Use TEST data path
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
        --batch_size "$TEST_BATCH_SIZE" # Use test batch size
        --num_workers "$NUM_WORKERS"
        --patience "$PATIENCE"
        --learning_rate "$LEARNING_RATE"
        --dropout "$DROPOUT"
        --weight_decay $WEIGHT_DECAY
        --lradj 'none'
        --scheduler 'none'
        --use_norm $USE_NORM
        --inverse
        --optimizer adamw
    )
     if [ "$USE_AMP" = True ]; then TEST_ARGS+=(--use_amp); fi

    echo "Running testing command (Output Dir: $ACTUAL_FOLD_OUTPUT_DIR)..."
    # The python script will reuse the same output dir based on the setting name read from SETTING_FILE_PATH
    python run.py "${TEST_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "Warning: Testing Fold $k failed." >&2
        # Continue to next fold, but log the failure
    else
       echo "Testing Fold $k finished. Results saved in $ACTUAL_FOLD_OUTPUT_DIR"
    fi

done

# --- Summary Generation ---
echo ""
echo "===== Generating Cross-Validation Summary ====="
# <<< Save summary inside the main CV run directory >>>
SUMMARY_FILE="${CV_RUN_DIR}/cv_summary.txt"
echo "Saving summary to: $SUMMARY_FILE"
echo "# K-Fold Cross-Validation Summary" > "$SUMMARY_FILE"
echo "# Run Directory: $CV_RUN_DIR" >> "$SUMMARY_FILE"
echo "# Base Model ID: $BASE_MODEL_ID" >> "$SUMMARY_FILE"
echo "# K = $K_FOLDS" >> "$SUMMARY_FILE"
echo "# Timestamp: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Fold | Best Validation Loss | Fold Output Directory Name" >> "$SUMMARY_FILE"
echo "-----|----------------------|---------------------------" >> "$SUMMARY_FILE"

BEST_FOLD_LOSS=inf
BEST_FOLD_DIR_NAME=""
BEST_FOLD_IDX=-1

# Loop through the recorded fold directory *names*
for i in "${!FOLD_DIRS[@]}"; do
    fold_dir_name="${FOLD_DIRS[$i]}"
    # Construct full path to the fold directory created by python script
    actual_fold_dir_path="./run_cv/${fold_dir_name}"
    loss_file="${actual_fold_dir_path}/best_vali_loss.txt"

    if [ -f "$loss_file" ]; then
        loss=$(cat "$loss_file")
        echo "  $i  | $loss             | $fold_dir_name" >> "$SUMMARY_FILE"
        # Check if this is the best loss so far
         if (( $(echo "$loss < $BEST_FOLD_LOSS" | bc -l) )); then
              BEST_FOLD_LOSS=$loss
              BEST_FOLD_DIR_NAME=$fold_dir_name
              BEST_FOLD_IDX=$i
         fi
    else
        echo "  $i  | --- Not Found ---    | $fold_dir_name" >> "$SUMMARY_FILE"
        echo "Warning: best_vali_loss.txt not found for fold $i in $actual_fold_dir_path"
    fi
done

echo "" >> "$SUMMARY_FILE"
if [ $BEST_FOLD_IDX -ne -1 ]; then
    echo "Best Fold based on Validation Loss: Fold $BEST_FOLD_IDX" >> "$SUMMARY_FILE"
    echo "Best Validation Loss: $BEST_FOLD_LOSS" >> "$SUMMARY_FILE"
    echo "Best Fold Directory Name: $BEST_FOLD_DIR_NAME" >> "$SUMMARY_FILE"
    echo "-> Check testing results (metrics_summary.txt, results_*.csv) in ./run_cv/$BEST_FOLD_DIR_NAME" >> "$SUMMARY_FILE"
else
     echo "Could not determine the best fold (no validation loss files found?)." >> "$SUMMARY_FILE"
fi

echo ""
echo "===== K-Fold Run Finished ====="
echo "Main Output Directory: $CV_RUN_DIR"
echo "Summary saved to $SUMMARY_FILE"
echo "Best validation fold: $BEST_FOLD_IDX (Loss: $BEST_FOLD_LOSS)"
echo "Check detailed results and test metrics in the respective fold directories within $CV_RUN_DIR"
