import torch
import torch.nn
from math import sqrt, log
from einops.layers.torch import Rearrange

from ajperry_pipeline.ml.models.blocks.layer_normalization import LayerNormalization


class ImageEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, image_size: tuple[int, int], patch_size: tuple[int, int]):
        """Generate Docstring

        Args:
            d_model (int): dimension of embeddings in model
            image_size (tuple[int, int]): dimensions of each image
            patch_size (tuple[int, int]): dimensions of each patch
        """
        super().__init__()
        self.d_model = d_model
        channels = 4
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        assert image_height % patch_height == 0
        assert image_width % patch_width == 0
        patch_dim = channels * patch_height * patch_width
        self.embedding_layer = torch.nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            LayerNormalization(),
            torch.nn.Linear(patch_dim, d_model),
            LayerNormalization()
        )

    def forward(self, x):
        return self.embedding_layer(x) * sqrt(self.d_model)


class ImagePositionEncoding(torch.nn.Module):
    def __init__(self, d_model, image_size: tuple[int, int], patch_size: tuple[int, int], dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout)
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        pe = self.posemb_sincos_2d(
                    h = image_height // patch_height,
                    w = image_width // patch_width,
                    dim = d_model,
                ) 
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4)
        omega = 1.0 / (temperature ** omega)
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)
        
    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


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
