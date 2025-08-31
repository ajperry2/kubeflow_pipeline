import torch
import torch.nn
from math import sqrt

from ajperry_pipeline.ml.models.blocks.layer_normalization import LayerNormalization


class ResidualConnection(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))


class SelfAttention(torch.nn.Module):
    """
    The Self Attention block of a transformer architecture.

    Attributes:
        w_queries (Parameter): Query weights
        w_keys (Parameter): Key weights
        w_values (Parameter): Value weights
        w_agg (Parameter): Aggregation weights
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = torch.nn.Dropout(dropout)
        assert d_model % h == 0

        self.d_k = d_model // h

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = SelfAttention.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
