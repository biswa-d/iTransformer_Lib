import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


# Removed Dataset_ETT_hour and Dataset_ETT_minute
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 noise_voltage=0.005, noise_current=0.003, noise_temp=0.001, noise_soc=0.0002, use_noise=True,
                 k_folds=0, fold=0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = False # Data is pre-scaled
        self.timeenc = timeenc
        self.freq = freq
        
        # Noise parameters
        self.noise_map = { # Map feature names to noise levels
            'Voltage': noise_voltage,
            'Current': noise_current,
            'Temp': noise_temp,
            'SOC': noise_soc
        }
        self.use_noise = use_noise 
        
        self.root_path = root_path
        self.data_path = data_path
        
        # K-Fold params
        self.k_folds = k_folds
        self.fold = fold
        
        # Store full data arrays
        self.data_x_full = None
        self.data_y_full = None
        self.data_stamp_full = None
        self.feature_cols = None
        self.valid_start_indices = None # Indices for sequences
        
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # --- Identify Feature Columns ---
        all_columns = list(df_raw.columns)
        self.feature_cols = [col for col in all_columns if col not in ['date', self.target]]
        cols_for_x = self.feature_cols
        cols_for_y = self.feature_cols + [self.target]

        # --- Get and Store Full Original Data ---
        try:
            self.data_x_full = df_raw[cols_for_x].values
            self.data_y_full = df_raw[cols_for_y].values
        except KeyError as e:
             print(f"Error selecting columns: {e}. Check column names in CSV and target variable.")
             raise

        df_stamp_full = df_raw[['date']]
        df_stamp_full['date'] = pd.to_datetime(df_stamp_full.date)
        if self.timeenc == 0:
            df_stamp_full['month'] = df_stamp_full.date.apply(lambda row: row.month, 1)
            df_stamp_full['day'] = df_stamp_full.date.apply(lambda row: row.day, 1)
            df_stamp_full['weekday'] = df_stamp_full.date.apply(lambda row: row.weekday(), 1)
            df_stamp_full['hour'] = df_stamp_full.date.apply(lambda row: row.hour, 1)
            self.data_stamp_full = df_stamp_full.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp_raw = time_features(pd.to_datetime(df_stamp_full['date'].values), freq=self.freq)
            self.data_stamp_full = data_stamp_raw.transpose(1, 0)
        # --- End Getting Full Data ---

        num_total_samples = len(df_raw)
        num_possible_sequences = num_total_samples - self.seq_len - self.pred_len + 1
        if num_possible_sequences <= 0:
             raise ValueError(f"Dataset length ({num_total_samples}) is too short for seq_len={self.seq_len} and pred_len={self.pred_len}")

        # --- Split Logic (Train/Val Random Split of Sequence Indices, Test Uses Full File) ---
        if self.set_type == 2: # Test flag - Use all possible sequences from the test file
            print(f"Using full pre-scaled test data file. Num possible sequences: {num_possible_sequences}")
            # For test set, valid indices are just 0 to num_possible_sequences-1
            self.valid_start_indices = np.arange(num_possible_sequences)
        else: # Train or Validation flag
            if self.k_folds > 1: # K-Fold Cross-Validation logic
                 if self.fold >= self.k_folds:
                      raise ValueError(f"Current fold ({self.fold}) must be less than k_folds ({self.k_folds})")
                 
                 print(f"Using K-Fold CV: {self.k_folds} folds, current fold {self.fold}")
                 all_start_indices = np.arange(num_possible_sequences)
                 # Ensure consistent shuffle across folds if using fixed seed
                 permuted_start_indices = np.random.permutation(all_start_indices)
                 
                 # Split indices into K folds
                 fold_indices = np.array_split(permuted_start_indices, self.k_folds)
                 
                 if self.set_type == 1: # Validation flag - use the k-th fold
                     self.valid_start_indices = fold_indices[self.fold]
                     print(f"  Assigning fold {self.fold} ({len(self.valid_start_indices)} sequences) for validation.")
                 else: # Training flag - use all other folds
                     train_folds_indices = [fold_indices[i] for i in range(self.k_folds) if i != self.fold]
                     self.valid_start_indices = np.concatenate(train_folds_indices)
                     print(f"  Assigning folds {[i for i in range(self.k_folds) if i != self.fold]} ({len(self.valid_start_indices)} sequences) for training.")
            
            else: # Original 80/20 random split logic
                 num_train_seq = int(num_possible_sequences * 0.8)
                 num_vali_seq = num_possible_sequences - num_train_seq

                 # Generate shuffled sequence start indices ONCE
                 all_start_indices = np.arange(num_possible_sequences)
                 permuted_start_indices = np.random.permutation(all_start_indices)
                 train_seq_indices = permuted_start_indices[:num_train_seq]
                 vali_seq_indices = permuted_start_indices[num_train_seq:]

                 if self.set_type == 0: # Train flag
                     print(f"Using {num_train_seq} randomly selected sequences for training (80%)")
                     self.valid_start_indices = train_seq_indices
                 else: # Validation flag (set_type == 1)
                     print(f"Using {num_vali_seq} randomly selected sequences for validation (20%)")
                     self.valid_start_indices = vali_seq_indices

    def __getitem__(self, index):
        # Use the index to get the actual start position from the shuffled list
        s_begin = self.valid_start_indices[index]
        s_end = s_begin + self.seq_len

        # Slice the sequence from the *original full* data arrays
        seq_x = self.data_x_full[s_begin:s_end]
        seq_x_mark = self.data_stamp_full[s_begin:s_end]

        # Target value V(t) is at the end of the input sequence window
        seq_y = self.data_y_full[s_end - 1:s_end, -1:] # Shape [1, 1]
        
        # Corresponding time mark for the target
        seq_y_mark = self.data_stamp_full[s_end - 1:s_end] # Shape [1, num_time_features]

        # Inject noise (only affects seq_x)
        if self.use_noise:
            noise = np.zeros_like(seq_x) # Noise shape matches seq_x (N features)
            for i, feature_name in enumerate(self.feature_cols):
                if i < noise.shape[1]: # Ensure index is valid for noise array
                    if feature_name in self.noise_map:
                        noise_std = self.noise_map[feature_name]
                        noise[:, i] = np.random.normal(0, noise_std, size=seq_x.shape[0])
                    else:
                        print(f"Warning: Noise level not defined for feature '{feature_name}'")
                else:
                    print(f"Warning: Index {i} for feature '{feature_name}' out of bounds for noise array shape {noise.shape}")
            seq_x = seq_x + noise

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # The length is the number of sequences selected for this set (train/val/test)
        return len(self.valid_start_indices)

    def inverse_transform(self, data):
        print("Warning: inverse_transform called, but data is pre-scaled externally. Returning data as is.")
        return data


# Removed Dataset_PEMS, Dataset_Solar, and Dataset_Pred
