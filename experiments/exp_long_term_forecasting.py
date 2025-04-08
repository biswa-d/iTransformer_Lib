from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from torch.optim.lr_scheduler import CosineAnnealingLR
# <<< SWA Imports >>>
from torch.optim.swa_utils import AveragedModel, SWALR
# <<< End SWA Imports >>>

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, test_file=None):
        # Override data_path for the 'test' flag if test_file is provided
        if flag == 'test' and test_file:
            self.args.data_path = test_file  # Use the provided test file path

        # <<<--- Add Debug Print Here --->>>
        print(f"[DEBUG] In _get_data (flag='{flag}'), using self.args.data_path: {self.args.data_path}")
        # <<<--------------------------->>>

        # Call the data_provider with updated args
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer.lower() == 'adamw':
            print("Using AdamW optimizer")
            model_optim = optim.AdamW(self.model.parameters(), 
                                   lr=self.args.learning_rate,
                                   weight_decay=self.args.weight_decay) # AdamW handles weight decay correctly
        elif self.args.optimizer.lower() == 'adam':
            print("Using Adam optimizer")
            model_optim = optim.Adam(self.model.parameters(), 
                                   lr=self.args.learning_rate,
                                   weight_decay=self.args.weight_decay) # Standard Adam with L2 regularization
        else:
            # Default to Adam if optimizer arg is missing or unsupported
            if not hasattr(self.args, 'optimizer'):
                 print("Optimizer argument not found, defaulting to Adam.")
            else:
                 print(f"Warning: Unsupported optimizer '{self.args.optimizer}'. Defaulting to Adam.")
            model_optim = optim.Adam(self.model.parameters(), 
                                   lr=self.args.learning_rate,
                                   weight_decay=self.args.weight_decay)
                                   
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros((batch_x.size(0), self.args.pred_len, batch_x.size(2)), device=self.device).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple): outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]

                # --- Select Target for Loss Calculation --- 
                pred = outputs # Shape: [B, pred_len, 1]
                
                # Extract the true target voltage from batch_y
                # batch_y shape is [B, L+pred_len, 4] or similar depending on label_len
                # Target Voltage is the last column
                true = batch_y[:, -self.args.pred_len:, -1:].to(self.device) # Shape: [B, pred_len, 1]
                # --- End Selection --- 
                
                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Determine the base output path for this run based on k_folds and cv_run_dir
        if self.args.k_folds > 0 and self.args.cv_run_dir:
            base_output_dir = self.args.cv_run_dir # Use the specific CV run dir passed from shell
        elif self.args.k_folds > 0:
            base_output_dir = './run_cv/' # Fallback if cv_run_dir not provided
        else:
            base_output_dir = './run_outputs/' # Standard output dir

        output_path = os.path.join(base_output_dir, setting)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Outputs for this run will be saved in: {output_path}")

        # Save the arguments used for this run
        args_path = os.path.join(output_path, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(vars(self.args), f, indent=4)
        print(f"Arguments saved to {args_path}")

        time_now = time.time()

        train_steps = len(train_loader)
        # Pass the correct output path to EarlyStopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=output_path)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # <<< Custom LR Schedule Setup >>>
        initial_lr = 1e-8 # Start from a very small LR for warmup
        base_lr = self.args.learning_rate
        min_lr = self.args.cosine_eta_min # Use this as the minimum target LR
        warmup_epochs = self.args.lr_warmup_epochs
        # Check if custom multi-phase schedule is active
        use_custom_schedule = self.args.main_decay_epochs > 0
        main_decay_epochs = self.args.main_decay_epochs if use_custom_schedule else 0
        exploit_lr = self.args.exploit_lr if self.args.exploit_lr is not None else min_lr
        exploit_cycle_epochs = self.args.exploit_cycle_epochs if use_custom_schedule else 1 # Avoid division by zero if not used
        
        if use_custom_schedule:
            print("--- Using Custom Multi-Phase LR Schedule --- ")
            print(f"  Warmup Epochs: {warmup_epochs} (to {base_lr:.7f})")
            print(f"  Main Decay Epochs: {main_decay_epochs} (Cosine from {base_lr:.7f} to {min_lr:.7f})")
            print(f"  Exploit Start LR: {exploit_lr:.7f}")
            print(f"  Exploit Cycle Epochs: {exploit_cycle_epochs} (Cosine from {exploit_lr:.7f} to {min_lr:.7f})")
        elif warmup_epochs > 0:
             print(f"Using linear LR warmup for {warmup_epochs} epochs, from {initial_lr} to {base_lr}")
        else:
             print(f"Using fixed learning rate: {base_lr}") # Or potentially other default logic if needed
             
        # Set initial optimizer LR if warming up
        if warmup_epochs > 0:
            for param_group in model_optim.param_groups:
                param_group['lr'] = initial_lr
        else: # Set base LR if no warmup
             for param_group in model_optim.param_groups:
                 param_group['lr'] = base_lr
        # <<< End Custom LR Schedule Setup >>>

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # <<<--- Remove Debug Print Here --->>>
                # print(f"Epoch {epoch+1}, Batch {i+1}: Shape of batch_x fed to model: {batch_x.shape}")
                # <<<--------------------------->>>

                # decoder input
                dec_inp = torch.zeros((batch_x.size(0), self.args.pred_len, batch_x.size(2)), device=self.device).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple): outputs = outputs[0]
                        
                        # --- Select Target for Loss Calculation --- 
                        pred = outputs # Shape: [B, pred_len, 1]
                        true = batch_y[:, -self.args.pred_len:, -1:].to(self.device) # Shape: [B, pred_len, 1]
                        # --- End Selection --- 
                        
                        loss = criterion(pred, true)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    
                    # --- Select Target for Loss Calculation --- 
                    pred = outputs # Shape: [B, pred_len, 1]
                    true = batch_y[:, -self.args.pred_len:, -1:].to(self.device) # Shape: [B, pred_len, 1]
                    # --- End Selection --- 

                    loss = criterion(pred, true)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # <<< Implement Custom LR Adjustment >>>
            current_epoch_num = epoch + 1
            new_lr = -1 # Placeholder

            if use_custom_schedule:
                # Phase 1: Warmup
                if warmup_epochs > 0 and current_epoch_num <= warmup_epochs:
                    warmup_factor = current_epoch_num / warmup_epochs
                    new_lr = initial_lr + (base_lr - initial_lr) * warmup_factor
                # Phase 2: Main Decay
                elif current_epoch_num <= warmup_epochs + main_decay_epochs:
                    # Calculate progress within the main decay phase
                    epoch_in_main_decay = current_epoch_num - warmup_epochs
                    # Cosine annealing calculation
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_main_decay / main_decay_epochs))
                    new_lr = min_lr + (base_lr - min_lr) * cosine_decay
                # Phase 3: Exploitation Cycles
                else:
                    # Calculate progress within the current exploitation cycle
                    epoch_in_exploitation_phase = current_epoch_num - warmup_epochs - main_decay_epochs
                    # Use modulo to find position within the cycle (1-based for calculation)
                    epoch_in_current_cycle = (epoch_in_exploitation_phase - 1) % exploit_cycle_epochs + 1
                    # Cosine annealing calculation for the cycle
                    cosine_decay_exploit = 0.5 * (1 + np.cos(np.pi * epoch_in_current_cycle / exploit_cycle_epochs))
                    new_lr = min_lr + (exploit_lr - min_lr) * cosine_decay_exploit
            else: 
                # Fallback to only warmup if custom schedule is not enabled
                 if warmup_epochs > 0 and current_epoch_num <= warmup_epochs:
                     warmup_factor = current_epoch_num / warmup_epochs
                     new_lr = initial_lr + (base_lr - initial_lr) * warmup_factor
                 else:
                     new_lr = base_lr # Maintain base LR if no warmup and no custom schedule
            
            # Set the calculated LR in the optimizer
            if new_lr != -1: # Only set if calculated
                 # print(f"Epoch {current_epoch_num}: Setting LR to {new_lr:.7f}") # Optional debug
                 for param_group in model_optim.param_groups:
                     param_group['lr'] = new_lr
            # <<< End Custom LR Adjustment >>>

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            # --- Early Stopping Logic --- 
            # Decide whether to stop based on pre-SWA model performance
            # Or potentially evaluate swa_model on validation set periodically?
            # Current setup: Stops based on the original model's val loss.
            # SWA model is only finalized at the end.
            early_stopping(vali_loss, self.model) 
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            # --- End Early Stopping --- 

        # --- Final Model Selection and Saving --- 
        # Always load the best single model found by Early Stopping
        print("Loading best model based on validation loss (Early Stopping checkpoint).")
        best_model_path_es = os.path.join(output_path, 'checkpoint.pth') 
        final_model_state_dict = None
        try:
            # Load the state dict from the best checkpoint
            final_model_state_dict = torch.load(best_model_path_es)
            # Load it into the current model instance
            self.model.load_state_dict(final_model_state_dict)
            print(f"Successfully loaded best checkpoint from {best_model_path_es}")
        except Exception as e:
            print(f"Error loading early stopping checkpoint {best_model_path_es}: {e}")
            print("Proceeding with the model state at the end of training loop.")
            # Use the model state as it was at the end of the loop
            final_model_state_dict = self.model.state_dict()

        # --- Save Best Validation Loss (from EarlyStopping) --- 
        best_val_loss = early_stopping.val_loss_min
        val_loss_file_path = os.path.join(output_path, 'best_vali_loss.txt')
        try:
            with open(val_loss_file_path, 'w') as f:
                f.write(f"{best_val_loss:.7f}")
            print(f"Best validation loss during training ({best_val_loss:.7f}) saved to {val_loss_file_path}")
        except Exception as e:
            print(f"Error saving best validation loss: {e}")
        # --- End Save --- 

        # <<< Save the chosen final model (Always the best ES model now) >>>
        final_model_path = os.path.join(output_path, 'final_model.pth') 
        try:
            # Unwrap DataParallel if necessary before saving
            model_to_save_state = self.model # Start with the potentially loaded best model
            if isinstance(model_to_save_state, nn.DataParallel):
                model_to_save_state = model_to_save_state.module # Unwrap DataParallel
                
            # Save the state dict (either loaded best or final loop state)
            torch.save(model_to_save_state.state_dict(), final_model_path)
            print(f"Final model state dict saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving final model state dict: {e}")

        # Ensure self.model has the final state loaded for immediate use
        # (Already done when loading from checkpoint, just ensure consistency)
        if final_model_state_dict:
            try:
                # Reload into self.model just in case it was modified (unlikely here)
                # Need to handle DataParallel wrapping if loading into self.model which might be wrapped
                if isinstance(self.model, nn.DataParallel):
                     self.model.module.load_state_dict(final_model_state_dict)
                else:
                     self.model.load_state_dict(final_model_state_dict)
                print("Ensured self.model holds the final state.")
            except Exception as e:
                 print(f"Error ensuring self.model holds the final state: {e}")
        else:
             print("Warning: Could not ensure self.model holds final state as state_dict was not available.")


        return self.model

    def test(self, setting, test=0):
        # Set the test file dynamically
        test_file = self.args.data_path
        if not test_file:
            raise ValueError("Custom test file is required for testing.")

        print(f"Testing with custom test file: {test_file}")
        test_data, test_loader = self._get_data(flag='test', test_file=test_file)
        
        # Now data_x has 3 features, data_y has 4 (3 features + target V)
        # Comment out references to test_data.data_x and test_data.data_y as they no longer exist
        # print(f"Test data_x shape (Input features): {test_data.data_x.shape}") 
        # print(f"Test data_y shape (Features + Target): {test_data.data_y.shape}") 
        print(f"Test data number of sequences: {len(test_data)}") # Use len() which is now correct
        print(f"Batch size: {self.args.batch_size}")
        print(f"Number of batches: {len(test_loader)}")
        
        # Determine the correct base directory for loading/saving
        # The 'setting' identifies the specific run.
        setting_base_dir = None
        if self.args.cv_run_dir: # If CV base dir is provided, use it
            setting_base_dir = self.args.cv_run_dir
        else: # Otherwise, infer based on setting name (fallback/standard runs)
            is_cv_run_inferred = '_fold' in setting 
            setting_base_dir = './run_cv/' if is_cv_run_inferred else './run_outputs/'
            
        output_path = os.path.join(setting_base_dir, setting)
        # Ensure the directory exists (it should from training, but check)
        os.makedirs(output_path, exist_ok=True)
        print(f"Output files will be saved/loaded relative to: {output_path}")

        if test:
            print('loading model')
            # Construct path using the determined base directory and setting
            # <<< Load the final_model.pth instead of checkpoint.pth >>>
            model_load_path = os.path.join(setting_base_dir, setting, 'final_model.pth') 
            # Force flush the output
            print(f"DEBUG: Attempting to load model from: {model_load_path}", flush=True) 
            # Load model from the constructed path
            if not os.path.exists(model_load_path):
                # Fallback to trying checkpoint.pth if final_model.pth doesn't exist (for older runs)
                print(f"Warning: final_model.pth not found at {model_load_path}. Trying checkpoint.pth...")
                model_load_path = os.path.join(setting_base_dir, setting, 'checkpoint.pth')
                if not os.path.exists(model_load_path):
                    raise FileNotFoundError(f"Neither final_model.pth nor checkpoint.pth found in {os.path.join(setting_base_dir, setting)}. Ensure training completed successfully.")
            
            # <<< Load state dict, handling DataParallel >>>
            loaded_state_dict = torch.load(model_load_path)
            if isinstance(self.model, nn.DataParallel):
                print("Loading state dict into self.model.module (DataParallel detected)")
                self.model.module.load_state_dict(loaded_state_dict)
            else:
                print("Loading state dict directly into self.model")
                self.model.load_state_dict(loaded_state_dict)
            # <<< End loading logic >>>

        # Store predictions (only for Voltage) and true values (only for Voltage)
        voltage_preds = []
        voltage_trues = []
        
        # Get the feature order (now only 3 input features)
        feature_order_input = test_data.feature_cols 
        target_col = self.args.target # Should be 'Voltage'
        print(f"Input feature order: {feature_order_input}")
        print(f"Target column: {target_col}")

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # batch_x shape: [B, L, 3]
                # batch_y shape: [B, 1, 1] (Directly contains target V(t))
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device) # This IS the target V(t)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input (Placeholder, shape depends on model needs for y_mark)
                dec_inp = torch.zeros((batch_x.size(0), self.args.pred_len, batch_x.size(2)), device=self.device).float()

                # Get model outputs
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                         if isinstance(outputs, tuple): outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                
                # Model output shape: [B, pred_len=1, c_out=1]
                pred_v = outputs.detach().cpu() # Shape [B, 1, 1]
                
                # --- Get true voltage Directly from batch_y ---
                # batch_y already contains the target V(t) with shape [B, 1, 1]
                true_v = batch_y.detach().cpu() # Shape [B, 1, 1]
                # --- End Change ---

                voltage_preds.append(pred_v.numpy())
                voltage_trues.append(true_v.numpy())

        voltage_preds = np.concatenate(voltage_preds, axis=0)
        voltage_trues = np.concatenate(voltage_trues, axis=0)
        print('\nFinal shapes after concatenation:')
        print('voltage_preds shape:', voltage_preds.shape)
        print('voltage_trues shape:', voltage_trues.shape)
        
        # Reshape
        voltage_preds = voltage_preds.reshape(-1, voltage_preds.shape[-1])
        voltage_trues = voltage_trues.reshape(-1, voltage_trues.shape[-1])
        print('Final shapes after reshape for metrics:')
        print('voltage_preds shape:', voltage_preds.shape)
        print('voltage_trues shape:', voltage_trues.shape)

        # --- Result saving (all into output_path, which is now correctly determined) --- 

        # Calculate metrics ONLY for Voltage
        mae, mse, rmse, _, _ = metric(voltage_preds, voltage_trues)
        print("--- Metrics Calculation (Voltage Only) ---")
        print(f'Voltage MSE:{mse:.7f}, MAE:{mae:.7f}')

        # Calculate and print the number of learnable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of model parameters: {num_params}")

        # Save metrics to a file within the run's output directory
        metrics_file_path = os.path.join(output_path, 'metrics_summary.txt')
        print(f"--- Saving Metrics to {metrics_file_path} --- ")
        with open(metrics_file_path, 'w') as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"Voltage MSE: {mse:.7f}\n")
            f.write(f"Voltage MAE: {mae:.7f}\n")
            f.write(f"Voltage RMSE: {rmse:.7f}\n")
            f.write(f"Number of Parameters: {num_params}\n")

        # Save predictions and true values as CSV into the run's output directory
        print("--- Saving Results CSV --- ")
        # Use timestamp only for the filename within the setting directory
        timestamp = setting.split('_ts')[-1] if '_ts' in setting else 'test'
        csv_file_path = os.path.join(output_path, f'results_voltage_ts{timestamp}.csv')
        
        results_dict = {
            f'Prediction_{target_col}': voltage_preds.flatten(),
            f'True_{target_col}': voltage_trues.flatten()
        }
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(csv_file_path, index=False)
        print(f'Voltage results saved to: {csv_file_path}')

        return

    def simulate(self, setting):
        print(f"Starting simulation for setting: {setting}")

        # 1. Load Model Checkpoint
        model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")
        print(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode

        # 2. Get Test Data Object and Scaler
        # Use flag='test' to get the dataset object configured for test data
        # We need direct access to its data_x, data_stamp, and scaler
        test_data, _ = self._get_data(flag='test', test_file=self.args.data_path)
        # scaler = test_data.scaler # Removed: No longer using internal scaler
        # Note the warning about the scaler potentially being fit on test data
        print(f"Using test data file: {self.args.data_path}")
        print(f"Test data shape (scaled): {test_data.data_x.shape}")
        print(f"Test data stamps shape: {test_data.data_stamp.shape}")

        # Assume data_x columns: [Current, Temp, SOC, Voltage] -> Indices 0, 1, 2, 3
        current_col_idx = 0
        pred_indices = [1, 2, 3] # Indices for T, S, V in the data_x array

        # 3. Get Initial Seed Sequence & Future/Ground Truth Data
        seq_len = self.args.seq_len
        if len(test_data.data_x) < seq_len:
            raise ValueError("Test data length is less than sequence length.")

        # Initial history window (scaled data and time features)
        current_window_x = torch.from_numpy(test_data.data_x[0:seq_len]).float()
        current_window_mark = torch.from_numpy(test_data.data_stamp[0:seq_len]).float()

        # <<< Move initial history to the correct device >>>
        current_window_x = current_window_x.to(self.device)
        current_window_mark = current_window_mark.to(self.device)

        # Data needed for the loop and final evaluation
        # True future currents (scaled) for input construction
        future_true_current_scaled = test_data.data_x[seq_len:, current_col_idx]
        # True future T, S, V (scaled) for evaluation
        ground_truth_scaled_TSV = test_data.data_x[seq_len:, pred_indices]
        # Time features for the prediction steps
        future_marks = torch.from_numpy(test_data.data_stamp[seq_len:]).float()

        # Simulation horizon
        horizon = len(test_data.data_x) - seq_len
        print(f"Simulation horizon: {horizon} steps")

        # Lists to store unscaled simulation results
        simulated_T_scaled = []
        simulated_S_scaled = []
        simulated_V_scaled = []

        # 4. Autoregressive Simulation Loop
        with torch.no_grad():
            for k in range(horizon):
                # a. Prepare model inputs
                batch_x = current_window_x.unsqueeze(0).to(self.device) # Add batch dim
                batch_x_mark = current_window_mark.unsqueeze(0).to(self.device)

                # Prepare decoder input (assuming pred_len=1 for simulation step)
                # Placeholder for decoder input, shape (1, label_len + pred_len=1, num_features)
                # Need the *next* timestamp for the prediction step's mark
                next_step_mark = future_marks[k].unsqueeze(0) # Shape (1, num_time_features)
                # For iTransformer with label_len=0, pred_len=1
                dec_inp = torch.zeros((1, 1, self.args.enc_in), device=self.device).float() # Placeholder (1, 1, features)
                batch_y_mark = next_step_mark.unsqueeze(1).to(self.device) # Shape (1, 1, time_features)

                # b. Predict the next step (scaled T, S, V and potentially C)
                if self.args.use_amp:
                     with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple): outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]

                # Output shape is (1, 1, enc_in) - predictions for step k+1
                predicted_step_scaled = outputs[:, -1, :] # Take the last (only) prediction step

                # Extract predicted T, S, V (scaled) - indices 1, 2, 3 relative to enc_in
                predicted_TSV_scaled = predicted_step_scaled[:, pred_indices]

                # <<< Clamp scaled predictions to [0, 1] >>>
                predicted_TSV_scaled = torch.clamp(predicted_TSV_scaled, min=0.0, max=1.0)

                # --- Remove Inverse Transform Step ---
                # c. Inverse transform/denormalize predictions
                # predicted_TSV_scaled_np = predicted_TSV_scaled.squeeze(0).cpu().numpy().reshape(1, -1)
                # means_TSV = scaler.mean_[pred_indices]
                # stds_TSV = scaler.scale_[pred_indices]
                # predicted_TSV_unscaled = (predicted_TSV_scaled_np * stds_TSV) + means_TSV

                # d. Store SCALED predictions
                # Convert tensor to numpy for storage
                predicted_TSV_scaled_np = predicted_TSV_scaled.squeeze(0).cpu().numpy()
                simulated_T_scaled.append(predicted_TSV_scaled_np[0]) # Index 0 of TSV is Temp
                simulated_S_scaled.append(predicted_TSV_scaled_np[1]) # Index 1 of TSV is SOC
                simulated_V_scaled.append(predicted_TSV_scaled_np[2]) # Index 2 of TSV is Voltage

                # e. Get true Current for the next step (already scaled)
                true_current_scaled_next = future_true_current_scaled[k] # This is a scalar

                # f. Construct the *next* state vector (scaled) for the history window
                # Combine true scaled Current with predicted scaled T, S, V
                # Ensure true_current_scaled_next is correctly shaped (1,) and is float32
                next_state_scaled = torch.cat([
                    torch.tensor([true_current_scaled_next], device=self.device).float(), # <<< Cast to float32
                    predicted_TSV_scaled.squeeze(0).to(self.device) # Predicted T, S, V
                ], dim=0) # Shape should be (enc_in,)

                # g. Update the history window (append new, remove oldest)
                # Append along the time dimension (dim 0)
                current_window_x = torch.cat([current_window_x[1:], next_state_scaled.unsqueeze(0)], dim=0)
                # <<< Ensure next_step_mark is also on the correct device >>>
                current_window_mark = torch.cat([current_window_mark[1:], next_step_mark.to(self.device)], dim=0)

                # Print progress (optional)
                if (k + 1) % 1000 == 0:
                    print(f"Simulated step {k+1}/{horizon}")

        print("Simulation loop finished.")

        # 5. Save Simulation Results (Scaled)
        sim_results_folder = './results/' + setting + '_simulation/'
        if not os.path.exists(sim_results_folder): os.makedirs(sim_results_folder)

        simulated_T_scaled = np.array(simulated_T_scaled)
        simulated_S_scaled = np.array(simulated_S_scaled)
        simulated_V_scaled = np.array(simulated_V_scaled)

        # Extract scaled ground truth for comparison
        # ground_truth_scaled_TSV = test_data.data_x[seq_len:, pred_indices] # Already extracted above

        np.save(os.path.join(sim_results_folder, 'sim_pred_T_scaled.npy'), simulated_T_scaled)
        np.save(os.path.join(sim_results_folder, 'sim_pred_S_scaled.npy'), simulated_S_scaled)
        np.save(os.path.join(sim_results_folder, 'sim_pred_V_scaled.npy'), simulated_V_scaled)
        np.save(os.path.join(sim_results_folder, 'sim_true_TSV_scaled.npy'), ground_truth_scaled_TSV)

        # Save combined CSV (Scaled)
        sim_df = pd.DataFrame({
            'Simulated_Temp_Scaled': simulated_T_scaled, 'True_Temp_Scaled': ground_truth_scaled_TSV[:, 0],
            'Simulated_SOC_Scaled': simulated_S_scaled, 'True_SOC_Scaled': ground_truth_scaled_TSV[:, 1],
            'Simulated_Voltage_Scaled': simulated_V_scaled, 'True_Voltage_Scaled': ground_truth_scaled_TSV[:, 2]
        })
        sim_csv_path = os.path.join(sim_results_folder, 'simulation_results_scaled.csv')
        sim_df.to_csv(sim_csv_path, index=False)
        print(f"Simulation results saved to: {sim_results_folder}")

        # 6. Evaluate Simulation Metrics (on Scaled Data)
        mae_T, mse_T, rmse_T, _, _ = metric(simulated_T_scaled, ground_truth_scaled_TSV[:, 0])
        mae_S, mse_S, rmse_S, _, _ = metric(simulated_S_scaled, ground_truth_scaled_TSV[:, 1])
        mae_V, mse_V, rmse_V, _, _ = metric(simulated_V_scaled, ground_truth_scaled_TSV[:, 2])

        print("\n--- Simulation Metrics (Scaled) ---")
        print(f"Scaled Temp: MAE={mae_T:.7f}, MSE={mse_T:.7f}, RMSE={rmse_T:.7f}")
        print(f"Scaled SOC:  MAE={mae_S:.7f}, MSE={mse_S:.7f}, RMSE={rmse_S:.7f}")
        print(f"Scaled Volt: MAE={mae_V:.7f}, MSE={mse_V:.7f}, RMSE={rmse_V:.7f}")

        # Save metrics to file
        metrics_summary = {
            'Temp_Scaled': {'MAE': mae_T, 'MSE': mse_T, 'RMSE': rmse_T},
            'SOC_Scaled': {'MAE': mae_S, 'MSE': mse_S, 'RMSE': rmse_S},
            'Voltage_Scaled': {'MAE': mae_V, 'MSE': mse_V, 'RMSE': rmse_V}
        }
        with open(os.path.join(sim_results_folder, 'simulation_metrics_scaled.txt'), 'w') as f:
            f.write(json.dumps(metrics_summary, indent=4))
        print("Simulation metrics saved.")


        return # End of simulate method

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.concatenate(preds, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return