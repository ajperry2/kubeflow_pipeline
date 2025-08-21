import torch

from ajperry_pipeline.ml.models.blocks.self_attention import SelfAttention
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import FeedForwardNeuralNetwork

class Encoder(torch.nn.Module):
    def __init__(self, token_dim: int, embedding_size: int, num_heads: int, num_layers: int, dropout_p: float = 0.0):
        super().__init__() 
        self.self_attention = SelfAttention(token_dim, embedding_size, num_heads)
        self.feed_forward = FeedForwardNeuralNetwork(num_layers, token_dim, dropout_p=dropout_p)
        self.layer_norm_1 = torch.nn.LayerNorm((token_dim,))
        self.layer_norm_2 = torch.nn.LayerNorm((token_dim,))
        
    def forward(self, x, attention_mask):
        attentioned = self.self_attention(x, attention_mask)
        attentioned = self.layer_norm_1(attentioned + x)
        return self.layer_norm_2(attentioned + self.feed_forward(attentioned))