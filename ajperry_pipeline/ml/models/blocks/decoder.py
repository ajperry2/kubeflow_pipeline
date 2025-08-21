import torch

from ajperry_pipeline.ml.models.blocks.self_attention import SelfAttention
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import FeedForwardNeuralNetwork
from ajperry_pipeline.ml.models.blocks.encoder_decoder_attention import EncoderDecoderAttention


class Decoder(torch.nn.Module):
    """
    The decoder block of a transformer architecture.

    Attributes:
        self_attention (Tokenizer): Self attention
        encoder_decoder_attention (AutoModel): Second input aware self attention
        feed_forward (FeedForwardNeuralNetwork): Neural Network
    """
    def __init__(self, token_dim: int, embedding_size: int, num_heads: int, num_layers: int, dropout_p: float = 0.0, device: str = "cpu"):
        super().__init__() 
        self.self_attention = SelfAttention(token_dim, embedding_size, num_heads, device)
        self.encoder_decoder_attention = EncoderDecoderAttention(token_dim, embedding_size, num_heads, device)
        self.feed_forward = FeedForwardNeuralNetwork(num_layers, token_dim, dropout_p=dropout_p, device=device)
        self.layer_norm_1 = torch.nn.LayerNorm((token_dim,)).to(device)
        self.layer_norm_2 = torch.nn.LayerNorm((token_dim,)).to(device)
        self.layer_norm_3 = torch.nn.LayerNorm((token_dim,)).to(device)
        self.device = device

    def forward(self, original_context, x, attention_mask , current_index):
        attentioned = self.self_attention(x, attention_mask)
        attentioned = self.layer_norm_1(attentioned + x)
        attentioned_2 = self.encoder_decoder_attention(attentioned, original_context, attention_mask, current_index)
        attentioned_2 = self.layer_norm_2(attentioned + attentioned_2)
        return self.layer_norm_3(attentioned_2 + self.feed_forward(attentioned_2))