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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

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

        # Call the data_provider with updated args
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

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

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # Set the test file dynamically
        test_file = self.args.data_path
        if not test_file:
            raise ValueError("Custom test file is required for testing.")

        print(f"Testing with custom test file: {test_file}")
        test_data, test_loader = self._get_data(flag='test', test_file=test_file)
        print(f"Test data shape: {test_data.data_x.shape}")
        print(f"Test data length: {len(test_data)}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Number of batches: {len(test_loader)}")
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Get the feature order and target
        feature_order = test_data.feature_cols  # This will be [Current, SOC, Voltage] when Temp is target
        target_col = test_data.target  # This will be 'Temp' or whatever is specified
        
        # Create a mapping of feature names to their indices in the model's output
        # The model always outputs predictions in the order: features first, then target
        feature_to_idx = {feature: idx for idx, feature in enumerate(feature_order)}
        feature_to_idx[target_col] = len(feature_order)  # Target is always last

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                print(f"\nProcessing batch {i+1}/{len(test_loader)}")
                print(f"batch_x shape: {batch_x.shape}")
                print(f"batch_y shape: {batch_y.shape}")
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Get model outputs
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

                print(f"Model outputs shape: {outputs.shape}")

                # Select predictions and true values for each feature
                pred_dict = {}
                true_dict = {}
                
                # Process features in their actual order
                for feature in feature_order + [target_col]:
                    idx = feature_to_idx[feature]
                    pred_dict[feature] = outputs[:, -self.args.pred_len:, idx].detach().cpu().numpy()
                    true_dict[feature] = batch_y[:, -self.args.pred_len:, idx].detach().cpu().numpy()

                # Stack predictions and true values in the correct order
                pred = np.stack([pred_dict[feature] for feature in feature_order + [target_col]], axis=-1)
                true = np.stack([true_dict[feature] for feature in feature_order + [target_col]], axis=-1)

                print(f"Pred shape: {pred.shape}")
                print(f"True shape: {true.shape}")

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('\nFinal shapes after concatenation:')
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('Final shapes after reshape:')
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define a canonical order for saving results consistently
        canonical_save_order = ['Current', 'SOC', 'Temp', 'Voltage']

        # Calculate metrics for each feature using the canonical order for reporting
        metrics = {}
        print("--- Metrics Calculation ---")
        for feature in canonical_save_order:
            if feature in feature_to_idx: # Check if the feature exists in the current run
                idx = feature_to_idx[feature]
                mae, mse, rmse, _, _ = metric(preds[:, :, idx], trues[:, :, idx])
                metrics[feature] = {'mae': mae, 'mse': mse, 'rmse': rmse}
                print(f'{feature} -> Index in preds/trues: {idx}, MSE:{mse:.7f}, MAE:{mae:.7f}')
            else:
                # Use double quotes for the f-string to allow the inner single quote
                print(f"{feature} not found in this run's features/target.")
                # Corrected dictionary assignment without unnecessary backslashes
                metrics[feature] = {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan} # Placeholder

        # Calculate combined metrics (only for features present in the run)
        valid_features = [f for f in canonical_save_order if f in feature_to_idx]
        # Ensure np.mean gets a list of valid numbers (filter NaNs if any were added)
        mae_list = [metrics[f]['mae'] for f in valid_features if not np.isnan(metrics[f]['mae'])]
        mse_list = [metrics[f]['mse'] for f in valid_features if not np.isnan(metrics[f]['mse'])]
        rmse_list = [metrics[f]['rmse'] for f in valid_features if not np.isnan(metrics[f]['rmse'])]
        
        mae_combined = np.mean(mae_list) if mae_list else np.nan
        mse_combined = np.mean(mse_list) if mse_list else np.nan
        rmse_combined = np.mean(rmse_list) if rmse_list else np.nan
        print(f'Avg MSE (over {valid_features}):{mse_combined:.7f}, MAE:{mae_combined:.7f}')

        # Save metrics in canonical order
        print("--- Saving Metrics --- ")
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + " \n")
            f.write(f'mse_avg:{mse_combined:.7f}, mae_avg:{mae_combined:.7f}, rmse_avg:{rmse_combined:.7f}\n')
            for feature in canonical_save_order:
                # Use lowercase for file consistency
                f.write(f'mse_{feature.lower()}:{metrics[feature]["mse"]:.7f}, mae_{feature.lower()}:{metrics[feature]["mae"]:.7f}, rmse_{feature.lower()}:{metrics[feature]["rmse"]:.7f}\n')
            f.write('\n')
        print(f"Metrics saved to result_long_term_forecast.txt")

        # Save predictions and true values as CSV in canonical order
        print("--- Saving Results CSV --- ")
        csv_file_path = os.path.join(folder_path, 'results.csv')
        preds_flat = preds.reshape(-1, preds.shape[-1])
        trues_flat = trues.reshape(-1, trues.shape[-1])

        # Create results dictionary using canonical order for columns
        results_dict = {}
        print(f"Feature to Index Map used: {feature_to_idx}")
        for feature in canonical_save_order:
            if feature in feature_to_idx:
                idx = feature_to_idx[feature]
                results_dict[f'Prediction_{feature}'] = preds_flat[:, idx]
                results_dict[f'True_{feature}'] = trues_flat[:, idx]
                print(f"Saving {feature} (Index {idx}) to CSV columns.")
            else:
                # Handle cases where a feature might not be present (e.g., univariate runs)
                results_dict[f'Prediction_{feature}'] = [np.nan] * len(preds_flat)
                results_dict[f'True_{feature}'] = [np.nan] * len(trues_flat)
                print(f"{feature} not in model output, saving NaNs to CSV columns.")

        results_df = pd.DataFrame(results_dict)
        # Define the exact desired column order for the CSV
        desired_csv_columns = []
        for feature in canonical_save_order:
            desired_csv_columns.append(f'Prediction_{feature}')
            desired_csv_columns.append(f'True_{feature}')
        # Filter to ensure we only try to order columns that exist in the dataframe
        existing_desired_columns = [col for col in desired_csv_columns if col in results_df.columns]
        results_df = results_df[existing_desired_columns] # Reorder the dataframe columns

        results_df.to_csv(csv_file_path, index=False)
        print(f'Results saved to: {csv_file_path} with columns in order: {list(results_df.columns)}')

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
            import json
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