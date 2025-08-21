import torch

from ajperry_pipeline.ml.models.blocks.self_attention import SelfAttention
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import FeedForwardNeuralNetwork
from ajperry_pipeline.ml.models.blocks.encoder_decoder_attention import EncoderDecoderAttention


class Decoder(torch.nn.Module):
    def __init__(self, token_dim: int, embedding_size: int, num_heads: int, num_layers: int, dropout_p: float = 0.0):
        super().__init__() 
        self.self_attention = SelfAttention(token_dim, embedding_size, num_heads)
        self.encoder_decoder_attention = EncoderDecoderAttention(token_dim, embedding_size, num_heads)
        self.feed_forward = FeedForwardNeuralNetwork(num_layers, token_dim, dropout_p=dropout_p)
        self.layer_norm_1 = torch.nn.LayerNorm((token_dim,))
        self.layer_norm_2 = torch.nn.LayerNorm((token_dim,))
        self.layer_norm_3 = torch.nn.LayerNorm((token_dim,))
        
    def forward(self, original_context, x, attention_mask , current_index):
        attentioned = self.self_attention(x, attention_mask)
        attentioned = self.layer_norm_1(attentioned + x)
        attentioned_2 = self.encoder_decoder_attention(attentioned, original_context, attention_mask, current_index)
        attentioned_2 = self.layer_norm_2(attentioned + attentioned_2)
        return self.layer_norm_3(attentioned_2 + self.feed_forward(attentioned_2))