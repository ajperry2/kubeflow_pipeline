import torch

from ajperry_pipeline.ml.models.blocks.embedding import InputEmbedding, PositionEncoding
from ajperry_pipeline.ml.models.blocks.projection import Projection
from ajperry_pipeline.ml.models.blocks.encoder import Encoder, EncoderBlock
from ajperry_pipeline.ml.models.blocks.decoder import Decoder, DecoderBlock
from ajperry_pipeline.ml.models.blocks.self_attention import SelfAttention
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import (
    FeedForwardNeuralNetwork,
)


class Transformer(torch.nn.Module):
    """
    A sequence to sequence model which uses attention for context.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for embedding model.
        embedding_model (AutoModel): The embedding model
        max_length (int): The number of tokens taken as input.
        encoders (list[Encoder]): The models encoders.
        decoders (list[Decoder]): The models decoders.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        target_embed: InputEmbedding,
        src_position: PositionEncoding,
        target_position: PositionEncoding,
        projection: Projection,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        print(src_position)
        self.src_position = src_position
        self.target_position = target_position
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_sequence_length: int,
    target_sequence_length: int,
    d_model: int = 512,
    n: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Embedding Layers
    source_embed: InputEmbedding = InputEmbedding(d_model, source_vocab_size)
    target_embed: InputEmbedding = InputEmbedding(d_model, target_vocab_size)

    source_position: PositionEncoding = PositionEncoding(
        d_model, source_sequence_length, dropout
    )
    target_position: PositionEncoding = PositionEncoding(
        d_model, target_sequence_length, dropout
    )

    # encoder blocks
    encoder_blocks = []
    for _ in range(n):
        self_attention = SelfAttention(d_model, h, dropout)
        feed_forward = FeedForwardNeuralNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks = []
    for _ in range(n):
        self_attention = SelfAttention(d_model, h, dropout)
        cross_attention = SelfAttention(d_model, h, dropout)
        feed_forward = FeedForwardNeuralNetwork(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(
            self_attention, cross_attention, feed_forward, dropout
        )
        decoder_blocks.append(decoder_block)

    # create encoder/decoder
    encoder = Encoder(torch.nn.ModuleList(encoder_blocks))
    decoder = Decoder(torch.nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = Projection(d_model, target_vocab_size)
    transformer = Transformer(
        encoder,
        decoder,
        source_embed,
        target_embed,
        source_position,
        target_position,
        projection_layer,
    )

    # initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return transformer
