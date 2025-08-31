import torch

from ajperry_pipeline.ml.models.blocks.self_attention import (
    SelfAttention,
    ResidualConnection,
)
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import (
    FeedForwardNeuralNetwork,
)
from ajperry_pipeline.ml.models.blocks.layer_normalization import LayerNormalization


class DecoderBlock(torch.nn.Module):
    """
    The decoder block of a transformer architecture.

    Attributes:
        self_attention (Tokenizer): Self attention
        encoder_decoder_attention (AutoModel): Second input aware self attention
        feed_forward (FeedForwardNeuralNetwork): Neural Network
    """

    def __init__(
        self,
        self_attention: SelfAttention,
        cross_attention: SelfAttention,
        feed_forward: FeedForwardNeuralNetwork,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = torch.nn.ModuleList(
            [ResidualConnection(dropout) for i in range(3)]
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, target_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
