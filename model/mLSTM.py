import torch
import torch.nn as nn
from xlstm.blocks.mlstm.cell_m import mLSTMCell, mLSTMCellConfig

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.output_size = configs.c_out
        self.num_heads = configs.num_heads  # Additional for mLSTM

        # Define mLSTM layers
        self.mlstm_cells = nn.ModuleList([
            mLSTMCell(config=mLSTMCellConfig(
                context_length=self.seq_len,
                embedding_dim=self.hidden_size,
                num_heads=self.num_heads
            )) for _ in range(self.num_layers)
        ])

        # Fully connected layer to map hidden states to output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Ignore x_mark_enc, x_dec, x_mark_dec as in the original LSTM implementation
        B, S, _ = x_enc.shape
        q, k, v = x_enc, x_enc, x_enc  # For mLSTMCell

        for cell in self.mlstm_cells:
            q = cell(q=q, k=k, v=v)  # Passing through mLSTM cells

        # Fully connected output
        dec_out = self.fc(q[:, -self.pred_len:, :])  # Shape: [batch_size, pred_len, output_size]
        return dec_out  # [B, L, D]
