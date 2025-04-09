# run_cv.py
import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
# Import other Exp classes if needed, or keep it specific for now
# from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import os
import time
import json # For saving args

def create_setting_string(args, fold_num):
    """Helper function to create a consistent setting string for logging/identification."""
    setting_components = [
        f"{args.model_id}_fold{fold_num}", # Include fold in the identifier
        args.model,
        args.data,
        f'sl{args.seq_len}',
        f'dm{args.d_model}',
        f'nh{args.n_heads}',
        f'df{args.d_ff}',
        # Add timestamp if needed for uniqueness within fold, though output_path handles it
        # f'ts{time.strftime("%Y%m%d_%H%M%S")}'
    ]
    return '_'.join(setting_components)

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer K-Fold CV Runner')

    # --- Arguments needed by run_cv.py ---
    parser.add_argument('--k_folds', type=int, required=True, help='Number of folds for K-Fold CV (must be > 1)')
    parser.add_argument('--cv_base_path', type=str, required=True, help='Base output directory for the entire CV run (e.g., ./run_cv/job_name_tsXXX)')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data file for final evaluation')

    # --- Copy MOST arguments from run.py (excluding is_training, fold, output_path, run_timestamp) ---
    # basic config
    parser.add_argument('--model_id', type=str, required=True, default='cv_run', help='Base model id for the CV run')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, help='Train/Val data csv file') # This is TRAIN data path
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints (less relevant now)')

    # noise injection parameters
    parser.add_argument('--noise_voltage', type=float, default=0.005, help='noise percentage for voltage (0.5% = 0.005)')
    parser.add_argument('--noise_current', type=float, default=0.003, help='noise percentage for current (0.3% = 0.003)')
    parser.add_argument('--noise_temp', type=float, default=0.001, help='noise percentage for temperature (0.1% = 0.001)')
    parser.add_argument('--noise_soc', type=float, default=0.0002, help='noise percentage for SOC (0.02% = 0.0002)')
    parser.add_argument('--use_noise', action='store_true', default=False, help='whether to inject noise during training')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    # parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data') # Can add if needed
    # parser.add_argument('--do_simulate', action='store_true', help='whether to run autoregressive simulation') # Can add if needed

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times (usually 1 for CV fold)')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=None, help='batch size for testing (defaults to train batch_size)')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='cv_run', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate type [type1, type2]')
    parser.add_argument('--lr_decay_factor', type=float, default=0.8, help='factor for learning rate decay (used by type1 lradj)')
    parser.add_argument('--lr_decay_period', type=int, default=20, help='period for learning rate decay (used by type1 lradj)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (e.g., 1e-4)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer specific (if needed)
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF', help='experiemnt name')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    # parser.add_argument('--partial_start_index', type=int, default=0, help='partial training start index') # Add if using partial exp

    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='none', help='Learning rate scheduler type [none, cosine]')
    parser.add_argument('--cosine_T_max', type=int, default=None, help='T_max for CosineAnnealingLR (default: train_epochs)')
    parser.add_argument('--cosine_eta_min', type=float, default=0.0, help='Minimum learning rate for CosineAnnealingLR')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='Number of epochs for linear learning rate warmup (0 to disable)')

    # SWA Arguments
    parser.add_argument('--use_swa', action='store_true', help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa_start_frac', type=float, default=0.75, help='Fraction of epochs to complete before starting SWA')
    parser.add_argument('--swa_lr', type=float, default=None, help='SWA learning rate. If None, uses the base learning_rate.')
    parser.add_argument('--swa_anneal_epochs', type=int, default=10, help='Number of epochs in the SWA annealing strategy')

    # Optimizer Choice
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer to use (adam or adamw)')

    # Custom Multi-Phase LR Schedule
    parser.add_argument('--main_decay_epochs', type=int, default=0, help='Epochs for main LR decay phase after warmup (0 disables custom schedule)')
    parser.add_argument('--exploit_lr', type=float, default=None, help='Starting LR for exploitation cycles (defaults to cosine_eta_min)')
    parser.add_argument('--exploit_cycle_epochs', type=int, default=10, help='Length of each exploitation cycle')
    # --- End Copied Arguments ---

    # Add fold argument internally, not from command line for this script
    parser.add_argument('--fold', type=int, default=0, help='Current fold index (managed internally)')
    parser.add_argument('--output_path', type=str, default='', help='Path for fold outputs (managed internally)')


    args = parser.parse_args()

    # --- Post-processing and Validation ---
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.k_folds <= 1:
        raise ValueError("--k_folds must be greater than 1 for cross-validation.")

    if not os.path.isdir(args.cv_base_path):
        # Or create it? Let's assume shell script creates it.
        raise ValueError(f"--cv_base_path directory does not exist: {args.cv_base_path}")

    # Check full paths for data files
    full_train_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(full_train_path):
         raise FileNotFoundError(f"Train/Val data file not found at expected path: {full_train_path}")
    full_test_path = os.path.join(args.root_path, args.test_data_path)
    if not os.path.exists(full_test_path):
         raise FileNotFoundError(f"Test data file not found at expected path: {full_test_path}")

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size # Default test batch size

    print('Arguments for CV Run:')
    print(args)
    # --- End Post-processing ---

    # --- Select Experiment Class ---
    # Assuming Exp_Long_Term_Forecast for now
    Exp = Exp_Long_Term_Forecast
    # Add logic here if different Exp classes are needed based on args.exp_name etc.
    # if args.exp_name == 'partial_train':
    #     Exp = Exp_Long_Term_Forecast_Partial
    # else:
    #     Exp = Exp_Long_Term_Forecast
    # --- End Experiment Class Selection ---


    # --- K-Fold Loop ---
    print(f"\n===== Starting K-Fold Cross-Validation (K={args.k_folds}) =====")
    print(f"Base Output Directory: {args.cv_base_path}")

    all_fold_metrics = [] # Store metrics from each fold

    for k in range(args.k_folds):
        print(f"\n----- Running Fold {k}/{args.k_folds-1} -----")

        # --- Prepare Args for this Fold ---
        fold_args = argparse.Namespace(**vars(args)) # Create a copy of args
        fold_args.fold = k
        fold_args.output_path = os.path.join(args.cv_base_path, f'fold_{k}')
        # Ensure fold directory exists (should be created by shell, but double-check)
        os.makedirs(fold_args.output_path, exist_ok=True)

        # Create a setting string for logging within this fold (optional)
        setting_str = create_setting_string(fold_args, k)

        # Save fold-specific args (optional, but good practice)
        try:
            with open(os.path.join(fold_args.output_path, 'args_fold.json'), 'w') as f:
                 json.dump(vars(fold_args), f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save fold-specific args: {e}")
        # --- End Arg Prep ---


        # --- Instantiate and Run Experiment for Fold ---
        exp = Exp(fold_args) # Pass fold-specific args

        # 1. Train
        print(f">>>>>>> Training Fold {k}: {setting_str} >>>>>>>")
        exp.train(setting_str) # Pass setting string for internal logging if needed

        # 2. Test
        print(f"\n>>>>>>> Testing Fold {k}: {setting_str} >>>>>>>")
        # Pass the explicit test data path
        exp.test(setting=setting_str, test_data_path=args.test_data_path, test=1)

        # --- Collect Metrics (Optional) ---
        try:
            metrics_path = os.path.join(fold_args.output_path, 'metrics_summary.txt')
            if os.path.exists(metrics_path):
                 with open(metrics_path, 'r') as f:
                      # Basic parsing, adjust if format changes
                      metrics = {}
                      for line in f:
                           if ':' in line:
                                key, value = line.split(':', 1)
                                try:
                                     metrics[key.strip()] = float(value.strip())
                                except ValueError:
                                     metrics[key.strip()] = value.strip() # Keep as string if not float
                      all_fold_metrics.append(metrics)
                      print(f"Fold {k} Metrics collected.")
            else:
                 print(f"Warning: Metrics file not found for fold {k} at {metrics_path}")
                 all_fold_metrics.append(None) # Placeholder for missing metrics
        except Exception as e:
            print(f"Warning: Error collecting metrics for fold {k}: {e}")
            all_fold_metrics.append(None)
        # --- End Metric Collection ---

        torch.cuda.empty_cache()
        # --- End Fold ---

    # --- Aggregate and Summarize Results ---
    print(f"\n===== K-Fold Cross-Validation Summary =====")
    print(f"Base Output Directory: {args.cv_base_path}")
    summary_path = os.path.join(args.cv_base_path, 'cv_summary_results.txt')
    print(f"Saving summary to: {summary_path}")

    avg_metrics = {}
    valid_fold_count = 0
    with open(summary_path, 'w') as f:
        f.write(f"# K-Fold CV Summary (K={args.k_folds})\n")
        f.write(f"# Base Path: {args.cv_base_path}\n")
        f.write(f"# Timestamp: {time.strftime('%Y%m%d_%H%M%S')}\n\n")
        f.write("Fold | MSE       | MAE       | ... (Add other metrics)\n")
        f.write("-----|-----------|-----------|------------------------\n")

        for i, metrics in enumerate(all_fold_metrics):
            if metrics and 'Voltage MSE' in metrics and 'Voltage MAE' in metrics:
                 mse = metrics['Voltage MSE']
                 mae = metrics['Voltage MAE']
                 f.write(f"{i:<4} | {mse:<9.7f} | {mae:<9.7f} | ...\n")
                 # Accumulate for averaging
                 for key, value in metrics.items():
                      if isinstance(value, (int, float)): # Only average numeric metrics
                           avg_metrics[key] = avg_metrics.get(key, 0) + value
                 valid_fold_count += 1
            else:
                 f.write(f"{i:<4} | ---       | ---       | Error or Missing Metrics\n")

        f.write("-----|-----------|-----------|------------------------\n")
        if valid_fold_count > 0:
            f.write("Avg  |")
            for key in avg_metrics:
                 avg_value = avg_metrics[key] / valid_fold_count
                 # Basic formatting, adjust width as needed
                 if 'MSE' in key or 'MAE' in key:
                      f.write(f" {avg_value:<9.7f} |")
                 else:
                      f.write(f" {avg_value:<9.3f} |") # Example for other metrics
            f.write(" ...\n")
            print("\nAverage Metrics Across Valid Folds:")
            for key, value in avg_metrics.items():
                 print(f"  {key}: {value / valid_fold_count:.7f}")
        else:
            f.write("Avg  | ---       | ---       | No valid folds found for averaging.\n")
            print("\nCould not calculate average metrics (no valid folds found).")

    print("\n===== K-Fold Run Finished =====")