import torch
import torch.nn
from math import sqrt, log


class InputEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """Generate Docstring

        Args:
            d_model (int): _description_
            vocab_size (int): _description_
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_layer = torch.nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding_layer(x) * sqrt(self.d_model)


class PositionEncoding(torch.nn.Module):
    def __init__(self, d_model, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-log(10_000.0) / d_model)
        )
        # apply sin to even terms
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
