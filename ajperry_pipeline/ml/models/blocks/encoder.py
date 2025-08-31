import torch

from ajperry_pipeline.ml.models.blocks.self_attention import (
    SelfAttention,
    ResidualConnection,
)
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import (
    FeedForwardNeuralNetwork,
)
from ajperry_pipeline.ml.models.blocks.layer_normalization import LayerNormalization


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        self_attention: SelfAttention,
        feed_forward: FeedForwardNeuralNetwork,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = torch.nn.ModuleList(
            [ResidualConnection(dropout) for i in range(2)]
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(torch.nn.Module):
    """
    The encoder block of a transformer architecture.

    Attributes:
        self_attention (Tokenizer): Self attention
        feed_forward (FeedForwardNeuralNetwork): Neural Network
    """

    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
