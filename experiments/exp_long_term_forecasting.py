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
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                # encoder - decoder: Use the original batch_x
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
                        # Get the full model output before selecting for evaluation
                        outputs_full = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # <<< ADD DEBUG PRINT >>>
                        #print(f"DEBUG: outputs_full shape after model call: {outputs_full.shape}")
                        # <<< END DEBUG PRINT >>>

                # --- Select targets (Temp=1, SOC=2, Voltage=3) for evaluation during testing ---
                target_indices = [1, 2, 3] # Indices for Temp, SOC, and Voltage
                # Select the relevant columns from the raw model output
                outputs_selected = outputs_full[:, -self.args.pred_len:, target_indices]

                # <<< ADD DEBUG PRINT >>>
                #print(f"DEBUG: batch_y shape on device before selection: {batch_y.shape}")
                #print(f"DEBUG: target_indices: {target_indices}")
                # <<< END DEBUG PRINT >>>

                batch_y_selected = batch_y[:, -self.args.pred_len:, target_indices].to(self.device)

                # Detach outputs and selected batch_y for processing/saving
                outputs_np = outputs_selected.detach().cpu().numpy()
                batch_y_numpy = batch_y.detach().cpu().numpy() # Get full original batch_y as numpy
                batch_y_numpy_selected = batch_y_numpy[:, -self.args.pred_len:, target_indices] # Select targets from numpy version

                # Initialize pred and true with selected model outputs and selected batch_y values
                pred = outputs_np
                true = batch_y_numpy_selected # These now contain Temp, SOC, and Voltage

                # Only rescale if both conditions are met
                if test_data.scale and self.args.inverse:
                    # Fetch mean and std for the output columns
                    # Assuming scaler was fit on [Current, Temp, SOC, Voltage]
                    output_means = test_data.scaler.mean_[target_indices] # Now gets mean for Temp, SOC, Voltage
                    output_stds = test_data.scaler.scale_[target_indices] # Now gets std for Temp, SOC, Voltage

                    # Rescale predictions and true values (update pred and true) - Broadcast across time dimension
                    pred = (outputs_np * output_stds) + output_means
                    true = (batch_y_numpy_selected * output_stds) + output_means

                    # print("Shape of rescaled predictions:", pred.shape)
                    # print("Shape of rescaled true labels:", true.shape)

                # Clamp predictions AFTER potential rescaling (apply column-wise if needed)
                # pred = np.clip(pred, ...)

                # Append to the results (preds/trues now have 3 columns)
                preds.append(pred)
                trues.append(true)

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # --- Adjust Metric Calculation and Saving for 3 Targets (Temp, SOC, Voltage) ---
        # Calculate metrics for each target separately for clarity
        mae_temp, mse_temp, rmse_temp, _, _ = metric(preds[:, :, 0], trues[:, :, 0]) # Metrics for Temp (index 0 in preds/trues)
        mae_soc,  mse_soc,  rmse_soc,  _, _ = metric(preds[:, :, 1], trues[:, :, 1]) # Metrics for SOC (index 1 in preds/trues)
        mae_volt, mse_volt, rmse_volt, _, _ = metric(preds[:, :, 2], trues[:, :, 2]) # Metrics for Voltage (index 2 in preds/trues)

        print(f'Temp MSE:{mse_temp:.7f}, MAE:{mae_temp:.7f}')
        print(f'SOC  MSE:{mse_soc:.7f}, MAE:{mae_soc:.7f}')
        print(f'Volt MSE:{mse_volt:.7f}, MAE:{mae_volt:.7f}') # <-- Uncommented
        # Calculate combined/average metrics if desired (optional) - Now includes Voltage
        mae_combined = np.mean([mae_temp, mae_soc, mae_volt])
        mse_combined = np.mean([mse_temp, mse_soc, mse_volt])
        rmse_combined = np.mean([rmse_temp, rmse_soc, rmse_volt])
        print(f'Avg MSE:{mse_combined:.7f}, MAE:{mae_combined:.7f}')

        # Calculate the number of trainable parameters
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of model parameters: {num_parameters}")

        # Append metrics and model parameters to the results file - Adjust format
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \\n")
            f.write(f'mse_avg:{mse_combined:.7f}, mae_avg:{mae_combined:.7f}, rmse_avg:{rmse_combined:.7f}\\n')
            # f.write(f'mse_curr:{mse_curr:.7f}, mae_curr:{mae_curr:.7f}, rmse_curr:{rmse_curr:.7f}\\n') # Current not evaluated
            f.write(f'mse_temp:{mse_temp:.7f}, mae_temp:{mae_temp:.7f}, rmse_temp:{rmse_temp:.7f}\\n')
            f.write(f'mse_soc:{mse_soc:.7f}, mae_soc:{mae_soc:.7f}, rmse_soc:{rmse_soc:.7f}\\n')
            f.write(f'mse_volt:{mse_volt:.7f}, mae_volt:{mae_volt:.7f}, rmse_volt:{rmse_volt:.7f}\\n') # <-- Uncommented
            f.write(f'parameters:{num_parameters}\\n')
            f.write('\\n')

        np.save(folder_path + 'metrics_avg.npy', np.array([mae_combined, mse_combined, rmse_combined]))
        # np.save(folder_path + 'metrics_curr.npy', np.array([mae_curr, mse_curr, rmse_curr]))
        np.save(folder_path + 'metrics_temp.npy', np.array([mae_temp, mse_temp, rmse_temp]))
        np.save(folder_path + 'metrics_soc.npy', np.array([mae_soc, mse_soc, rmse_soc]))
        np.save(folder_path + 'metrics_volt.npy', np.array([mae_volt, mse_volt, rmse_volt])) # <-- Uncommented

        np.save(folder_path + 'pred.npy', preds) # preds contains Temp, SOC, and Voltage predictions
        np.save(folder_path + 'true.npy', trues) # trues contains Temp, SOC, and Voltage ground truths

        # Save predictions and true values as CSV - Adjust columns
        csv_file_path = os.path.join(folder_path, 'results.csv')
        # Reshape for CSV: each row is one time step
        preds_flat = preds.reshape(-1, preds.shape[-1]) # Shape: (N_samples*pred_len, 3)
        trues_flat = trues.reshape(-1, trues.shape[-1]) # Shape: (N_samples*pred_len, 3)

        results_df = pd.DataFrame({
            # 'Prediction_Current': preds_flat[:, 0], # Index 0 is not Current anymore if Current wasn't in target_indices
            # 'True_Current': trues_flat[:, 0],
            'Prediction_Temp': preds_flat[:, 0],    # Index 0 of selected is Temp
            'True_Temp': trues_flat[:, 0],
            'Prediction_SOC': preds_flat[:, 1],     # Index 1 of selected is SOC
            'True_SOC': trues_flat[:, 1],
            'Prediction_Voltage': preds_flat[:, 2], # Index 2 of selected is Voltage
            'True_Voltage': trues_flat[:, 2]
        })
        results_df.to_csv(csv_file_path, index=False)
        # --- End metric/saving adjustment ---

        print(f'Results saved to: {csv_file_path}')

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
        scaler = test_data.scaler
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
        # True future T, S, V (unscaled) for evaluation
        # Need to get unscaled data - we can inverse transform the relevant part of data_x
        ground_truth_unscaled_full = scaler.inverse_transform(test_data.data_x)
        ground_truth_unscaled_TSV = ground_truth_unscaled_full[seq_len:, pred_indices]
        # Time features for the prediction steps
        future_marks = torch.from_numpy(test_data.data_stamp[seq_len:]).float()

        # Simulation horizon
        horizon = len(test_data.data_x) - seq_len
        print(f"Simulation horizon: {horizon} steps")

        # Lists to store unscaled simulation results
        simulated_T_unscaled = []
        simulated_S_unscaled = []
        simulated_V_unscaled = []

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

                # c. Inverse transform/denormalize predictions
                # Reshape for scaler: (num_samples, num_features) -> need (1, 3)
                predicted_TSV_scaled_np = predicted_TSV_scaled.squeeze(0).cpu().numpy().reshape(1, -1)
                # Use scaler fitted on potentially only test data!
                means_TSV = scaler.mean_[pred_indices]
                stds_TSV = scaler.scale_[pred_indices]
                predicted_TSV_unscaled = (predicted_TSV_scaled_np * stds_TSV) + means_TSV

                # d. Store unscaled predictions
                simulated_T_unscaled.append(predicted_TSV_unscaled[0, 0]) # Temp is index 0 of TSV
                simulated_S_unscaled.append(predicted_TSV_unscaled[0, 1]) # SOC is index 1 of TSV
                simulated_V_unscaled.append(predicted_TSV_unscaled[0, 2]) # Voltage is index 2 of TSV

                # e. Get true Current for the next step (already scaled)
                true_current_scaled_next = future_true_current_scaled[k] # This is a scalar

                # f. Construct the *next* state vector (scaled) for the history window
                # Combine true scaled Current with predicted scaled T, S, V
                # Ensure true_current_scaled_next is correctly shaped (1,)
                next_state_scaled = torch.cat([
                    torch.tensor([true_current_scaled_next], device=self.device), # True C
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

        # 5. Save Simulation Results
        sim_results_folder = './results/' + setting + '_simulation/'
        if not os.path.exists(sim_results_folder): os.makedirs(sim_results_folder)

        simulated_T = np.array(simulated_T_unscaled)
        simulated_S = np.array(simulated_S_unscaled)
        simulated_V = np.array(simulated_V_unscaled)

        np.save(os.path.join(sim_results_folder, 'sim_pred_T.npy'), simulated_T)
        np.save(os.path.join(sim_results_folder, 'sim_pred_S.npy'), simulated_S)
        np.save(os.path.join(sim_results_folder, 'sim_pred_V.npy'), simulated_V)
        np.save(os.path.join(sim_results_folder, 'sim_true_TSV.npy'), ground_truth_unscaled_TSV) # Save ground truth for easy loading

        # Save combined CSV
        sim_df = pd.DataFrame({
            'Simulated_Temp': simulated_T, 'True_Temp': ground_truth_unscaled_TSV[:, 0],
            'Simulated_SOC': simulated_S, 'True_SOC': ground_truth_unscaled_TSV[:, 1],
            'Simulated_Voltage': simulated_V, 'True_Voltage': ground_truth_unscaled_TSV[:, 2]
        })
        sim_csv_path = os.path.join(sim_results_folder, 'simulation_results.csv')
        sim_df.to_csv(sim_csv_path, index=False)
        print(f"Simulation results saved to: {sim_results_folder}")

        # 6. Evaluate Simulation Metrics
        mae_T, mse_T, rmse_T, _, _ = metric(simulated_T, ground_truth_unscaled_TSV[:, 0])
        mae_S, mse_S, rmse_S, _, _ = metric(simulated_S, ground_truth_unscaled_TSV[:, 1])
        mae_V, mse_V, rmse_V, _, _ = metric(simulated_V, ground_truth_unscaled_TSV[:, 2])

        print("\n--- Simulation Metrics ---")
        print(f"Temp: MAE={mae_T:.7f}, MSE={mse_T:.7f}, RMSE={rmse_T:.7f}")
        print(f"SOC:  MAE={mae_S:.7f}, MSE={mse_S:.7f}, RMSE={rmse_S:.7f}")
        print(f"Volt: MAE={mae_V:.7f}, MSE={mse_V:.7f}, RMSE={rmse_V:.7f}")

        # Save metrics to file
        metrics_summary = {
            'Temp': {'MAE': mae_T, 'MSE': mse_T, 'RMSE': rmse_T},
            'SOC': {'MAE': mae_S, 'MSE': mse_S, 'RMSE': rmse_S},
            'Voltage': {'MAE': mae_V, 'MSE': mse_V, 'RMSE': rmse_V}
        }
        with open(os.path.join(sim_results_folder, 'simulation_metrics.txt'), 'w') as f:
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