import torch

from ajperry_pipeline.ml.models.blocks.self_attention import SelfAttention
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import FeedForwardNeuralNetwork

class Encoder(torch.nn.Module):
    """
    The encoder block of a transformer architecture.

    Attributes:
        self_attention (Tokenizer): Self attention
        feed_forward (FeedForwardNeuralNetwork): Neural Network
    """
    def __init__(self, token_dim: int, embedding_size: int, num_heads: int, num_layers: int, dropout_p: float = 0.0, device: str = "cpu"):
        super().__init__() 
        self.self_attention = SelfAttention(token_dim, embedding_size, num_heads, device=device)
        self.feed_forward = FeedForwardNeuralNetwork(num_layers, token_dim, dropout_p=dropout_p, device=device)
        self.layer_norm_1 = torch.nn.LayerNorm((token_dim,)).to(device)
        self.layer_norm_2 = torch.nn.LayerNorm((token_dim,)).to(device)
        self.device = device
        
    def forward(self, x, attention_mask):
        attentioned = self.self_attention(x, attention_mask)
        attentioned = self.layer_norm_1(attentioned + x)
        return self.layer_norm_2(attentioned + self.feed_forward(attentioned))