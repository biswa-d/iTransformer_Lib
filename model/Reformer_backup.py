import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # Embedding layer
        self.enc_embedding = DataEmbedding_inverted(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Projection layer to output a single prediction
        self.projection = nn.Linear(configs.d_model, 1, bias=True)  # Output a single prediction value

    def long_forecast(self, x_enc, x_mark_enc):
        # Normalization step (if needed)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, D]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # Encoder outputs a sequence of context-aware representations

        # Extracting the last time step's encoding to make the prediction
        last_hidden_state = enc_out[:, -1, :]  # [B, D]
        dec_out = self.projection(last_hidden_state)  # [B, 1] (Predict one value)

        # Denormalization (optional, depending on your use case)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1))

        return dec_out  # [B, 1] (Single output for each batch)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        # Since we're doing one-step-ahead forecasting, we don't need decoder inputs
        return self.long_forecast(x_enc, x_mark_enc)
