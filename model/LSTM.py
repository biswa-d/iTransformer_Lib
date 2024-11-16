import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.output_size = configs.c_out

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        # Fully connected layer to map hidden states to output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Ignore the x_mark_enc, x_dec, x_mark_dec since LSTM doesn't use them
        lstm_out, _ = self.lstm(x_enc)  # lstm_out shape: [batch_size, seq_len, hidden_size]
        dec_out = self.fc(lstm_out[:, -self.pred_len:, :])  # dec_out shape: [batch_size, pred_len, output_size]
        return dec_out  # [B, L, D]
