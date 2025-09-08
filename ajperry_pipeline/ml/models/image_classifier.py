import torch

from ajperry_pipeline.ml.models.blocks.embedding import ImageEmbedding, ImagePositionEncoding
from ajperry_pipeline.ml.models.blocks.projection import Projection
from ajperry_pipeline.ml.models.blocks.encoder import Encoder, EncoderBlock
from ajperry_pipeline.ml.models.blocks.self_attention import SelfAttention
from ajperry_pipeline.ml.models.blocks.feed_forward_neural_network import (
    FeedForwardNeuralNetwork,
)


class ImageClassifier(torch.nn.Module):
    """
    A module which uses an encoder module to classify an image.

    Attributes:
        encoder (Encoder): The encoder portion of a transformer arch.
        src_embed (ImageEmbedding): The embedding module
        src_position (ImagePositionEncoding): The position encoding module
        projection (Projection): The module which projects to the number of classes
    """

    def __init__(
        self,
        encoder: Encoder,
        src_embed: ImageEmbedding,
        src_position: ImagePositionEncoding,
        projection: Projection,
    ):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_position = src_position
        self.projection = projection

    def encode(self, src):
        src = self.src_embed(src)
        src = self.src_position(src)
        return self.encoder(src,None)

    def project(self, x):
        return self.projection(x)

    def forward(self, x):
        x = self.encode(x)
        x = x.mean(dim = 1)
        return self.project(x)
        
        

def build_image_classifier(
    image_size: tuple[int, int], 
    patch_size: tuple[int, int],
    num_classes: int = 1,
    d_model: int = 512,
    n: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> ImageClassifier:
    # Embedding Layers
    src_embed: ImageEmbedding = ImageEmbedding(d_model, image_size, patch_size)
    src_position: ImagePositionEncoding = ImagePositionEncoding(d_model, image_size, patch_size, .001)

    # encoder blocks
    encoder_blocks = []
    for _ in range(n):
        self_attention = SelfAttention(d_model, h, dropout)
        feed_forward = FeedForwardNeuralNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # create encoder/decoder
    encoder = Encoder(torch.nn.ModuleList(encoder_blocks))
    
    # projection layer
    projection = Projection(d_model, num_classes)
    image_classifier = ImageClassifier(
        encoder=encoder,
        src_embed=src_embed,
        src_position=src_position,
        projection=projection,
    )

    # initialize parameters
    for p in image_classifier.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return image_classifier