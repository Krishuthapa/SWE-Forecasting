import os
import sys

import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model] — for batch broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# Encoder-Decoder Model
class SWETransformer(nn.Module):
    def __init__(self, model_dim, enc_inp_dim = 20, dec_inp_dim = 1, nhead = 8, n_enc_layers = 3, n_dec_layers = 1, ffn_dim = 2048, output_dim = 1, forecasting_window = 10, batch_first= True):
        super(SWETransformer, self).__init__()

        assert model_dim % nhead == 0, f"Model dim must be divisible by the number of heads. Given d_model : {model_dim} and nhead:{nhead}"

        self.encoder_embedding = nn.Linear(enc_inp_dim, model_dim)
        self.decoder_embedding = nn.Linear(dec_inp_dim, model_dim)

        self.transformer_layer = nn.Transformer(d_model = model_dim, nhead = nhead, num_encoder_layers = n_enc_layers,
                                                num_decoder_layers = n_dec_layers,  dim_feedforward = ffn_dim, batch_first = batch_first)

        self.positional_embedding = PositionalEmbedding(d_model = model_dim)

        self.output_layer = nn.Linear(model_dim, output_dim)

        self.forecasting_window = forecasting_window

    def forward(self, encoder_input_x, decoder_input_x, decoder_attn_mask = None, encoder_attn_mask = None, enc_padding_mask = None, decoder_padding_mask = None, cross_padding_mask = None):

        emb_enc_x = self.encoder_embedding(encoder_input_x)
        pos_emb_enc_x = self.positional_embedding(emb_enc_x)

        emb_dec_x = self.decoder_embedding(decoder_input_x)
        pos_emb_dec_x = self.positional_embedding(emb_dec_x)

        self.encoder_output = self.transformer_layer(pos_emb_enc_x, pos_emb_dec_x,src_mask = encoder_attn_mask,
                                                     tgt_mask = decoder_attn_mask, src_key_padding_mask = enc_padding_mask,
                                                     tgt_key_padding_mask = decoder_padding_mask, memory_key_padding_mask = cross_padding_mask,
                                                     tgt_is_causal = (decoder_attn_mask != None), src_is_causal = (encoder_attn_mask != None))

        output = self.output_layer(self.encoder_output)

        return output[:,-self.forecasting_window:,:]
    