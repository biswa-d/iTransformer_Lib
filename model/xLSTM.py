import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Modified version of xLSTM for consistency with transformer-based architectures.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len  # This should be 1, as you want to predict one step
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Optional Embedding Layer - Use if needed for transforming input features
        if configs.d_model != configs.enc_in:
            self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        else:
            self.enc_embedding = None

        # xLSTM Encoder-only architecture (similar to Transformer encoder)
        self.xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=configs.n_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=configs.seq_len,
            num_blocks=configs.e_layers,
            embedding_dim=configs.d_model,
            slstm_at=[1]
        )
        self.encoder = xLSTMBlockStack(self.xlstm_cfg)

        # Projection Layer (to predict the next value)
        self.projector = nn.Linear(configs.d_model, configs.c_out, bias=True)  # Predict single output (e.g., voltage)

    def forecast(self, x_enc, x_mark_enc=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Optional Embedding if needed
        if self.enc_embedding is not None:
            x_enc = self.enc_embedding(x_enc)  # [batch, seq_len, d_model]

        # Pass input through xLSTM encoder block stack
        enc_out = self.encoder(x_enc)  # [batch, seq_len, d_model]

        # Use the last time step's output for prediction
        last_hidden_state = enc_out[:, -1, :]  # [batch, d_model]

        # Project to predict the next value for each sequence
        dec_out = self.projector(last_hidden_state).unsqueeze(1)  # Shape: [batch, 1, c_out]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out  # [batch, 1, c_out]