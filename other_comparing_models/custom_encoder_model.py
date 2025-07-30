import torch
import torch.nn as nn

import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.alpha = nn.Parameter(torch.randn(1))  # Initialize randomly
        self.beta = nn.Parameter(torch.randn(1))   # Initialize randomly
        self.gamma = nn.Parameter(torch.randn(1))  # Initialize randomly
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, distances, angularities, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_scores = torch.softmax(attn_scores, dim= -1)
        
        distances = 1 / (1 + distances)
        angularities = 1/ (1 + angularities)

        new_score = self.alpha * attn_scores + self.beta * distances + self.gamma * angularities
        modified_attn_weights = torch.softmax(new_score, dim = -1)

        output = torch.matmul(modified_attn_weights, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, distances, angularities, mask=None):
    
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, distances, angularities, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model = 512,d_ff = 2048):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.relu = nn.ReLU()

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff= 2048, dropout= 0.10):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, distances, angularities, mask= None):
        attn_output = self.self_attn(x, x, x, distances, angularities, mask)

        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        
        x = self.norm2(x + self.dropout(ff_output))

        return x