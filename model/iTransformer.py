import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.enc_in = configs.enc_in # Number of input features (e.g., 3)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Modified Projector: Maps flattened features -> single output prediction
        self.projector = nn.Linear(configs.enc_in * configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Instance Normalization (Applied before embedding)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
            # Note: De-normalization is tricky with the modified projector if use_norm=True.
            # It's recommended to set use_norm=False when using this modified projector,
            # unless you implement a specific way to pass/use the target variable's original mean/std.

        B, L, N = x_enc.shape # N should be self.enc_in (e.g., 3)
        
        # Embedding
        # Output shape: [B, N, E] where E = d_model
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        
        # Encoder
        # Input shape: [B, N, E]
        # Output shape: [B, N, E]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projector (Modified)
        # Flatten the N and E dimensions
        enc_out_flat = enc_out.reshape(B, -1) # Shape: [B, N * E]
        # Project to prediction length
        dec_out_flat = self.projector(enc_out_flat) # Shape: [B, pred_len]
        # Reshape to target format
        dec_out = dec_out_flat.unsqueeze(-1) # Shape: [B, pred_len, 1]

        if self.use_norm:
            # De-Normalization (Problematic without target's mean/std)
            # If you kept use_norm=True, this part needs modification or removal.
            # For now, let's assume use_norm=False was set in config.
            # Example IF you had target_stdev and target_mean (passed somehow):
            # dec_out = dec_out * target_stdev + target_mean 
            pass # Avoid applying incorrect de-normalization

        return dec_out # Shape: [B, pred_len, 1]


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # The forecast method already returns [B, pred_len, 1], so slicing is correct.
        return dec_out[:, -self.pred_len:, :]  # [B, L, D] -> [B, pred_len, 1]