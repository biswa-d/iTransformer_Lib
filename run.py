import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import os
import time # Ensure time is imported

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # noise injection parameters
    parser.add_argument('--noise_voltage', type=float, default=0.005, help='noise percentage for voltage (0.5% = 0.005)')
    parser.add_argument('--noise_current', type=float, default=0.003, help='noise percentage for current (0.3% = 0.003)')
    parser.add_argument('--noise_temp', type=float, default=0.001, help='noise percentage for temperature (0.1% = 0.001)')
    parser.add_argument('--noise_soc', type=float, default=0.0002, help='noise percentage for SOC (0.02% = 0.0002)')
    parser.add_argument('--use_noise', action='store_true', default=False, help='whether to inject noise during training')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--do_simulate', action='store_true', help='whether to run autoregressive simulation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate type [type1, type2]')
    parser.add_argument('--lr_decay_factor', type=float, default=0.8, help='factor for learning rate decay (used by type1 lradj)')
    parser.add_argument('--lr_decay_period', type=int, default=20, help='period for learning rate decay (used by type1 lradj)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (e.g., 1e-4)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True) # changed to use multiple gpus by default
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # Argument for the timestamp generated by the shell script
    parser.add_argument('--run_timestamp', type=str, required=False, help='Timestamp for training run (required if is_training=1)') # Optional now, required only for training
    # Argument for the unique setting file path
    parser.add_argument('--setting_file_path', type=str, required=True, help='Path to the unique file storing/reading the setting name for this run')

    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='none', 
                        help='Learning rate scheduler type [none, cosine]')
    parser.add_argument('--cosine_T_max', type=int, default=None, 
                        help='T_max for CosineAnnealingLR (default: train_epochs)')
    parser.add_argument('--cosine_eta_min', type=float, default=0.0, 
                        help='Minimum learning rate for CosineAnnealingLR')

    # K-Fold Cross-Validation Arguments
    parser.add_argument('--k_folds', type=int, default=0, help='Number of folds for K-Fold CV (0 means disabled)')
    parser.add_argument('--fold', type=int, default=0, help='Current fold index (0 to k_folds-1) for K-Fold CV')
    parser.add_argument('--cv_run_dir', type=str, default=None, help='Base output directory for the entire CV run (used if k_folds > 0)')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        # Ensure timestamp is provided for training
        if not args.run_timestamp:
            raise ValueError("--run_timestamp is required when --is_training=1")

        for ii in range(args.itr):
            # setting record of experiments - using specified args and shell timestamp
            setting = '{}_{}_{}_sl{}_dm{}_nh{}_df{}_ts{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.d_model,
                args.n_heads,
                args.d_ff,
                args.run_timestamp) # Use timestamp passed from shell

            # Save the exact setting name to the unique file specified by the shell script
            logs_dir = os.path.dirname(args.setting_file_path)
            os.makedirs(logs_dir, exist_ok=True) # Ensure Logs directory exists
            try:
                with open(args.setting_file_path, 'w') as f:
                    f.write(setting)
                print(f"Saved setting '{setting}' to {args.setting_file_path}")
            except IOError as e:
                print(f"Error: Could not write setting file {args.setting_file_path}: {e}")
                exit(1) # Exit if we can't save the setting

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        # Determine the setting to test by reading the unique setting file
        setting = None
        try:
            with open(args.setting_file_path, 'r') as f:
                setting = f.read().strip()
            if not setting:
                print(f"Error: Setting file {args.setting_file_path} is empty.")
                exit(1)
            print(f"Read setting '{setting}' from {args.setting_file_path}")
        except FileNotFoundError:
            print(f"Error: Setting file not found at {args.setting_file_path}. Cannot determine which checkpoint to test.")
            exit(1)
        except IOError as e:
             print(f"Error reading setting file {args.setting_file_path}: {e}")
             exit(1)

        exp = Exp(args)  # set experiments
        # Decide whether to run test or simulation
        if args.do_simulate:
            print('>>>>>>>simulating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.simulate(setting) # Assuming simulate method loads the model
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
