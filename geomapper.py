
from hyperbolic import HyperbolicEmbedding
from logexp import ScaledEmbedding
from parabolic import ParabolicEmbedding
from spherical import SphericalEmbedding
from torch import nn

def map_2_geometry(geometry_type, vocab_size, d_model):
    """
    Maps the embedding to a specific geometry type.
    """
    if geometry_type == "hyperbolic":
        return HyperbolicEmbedding(vocab_size, d_model)
    elif geometry_type == "spherical":
        return SphericalEmbedding(vocab_size, d_model)
    elif geometry_type == "parabolic":
        return ParabolicEmbedding(vocab_size, d_model)
    elif geometry_type == "exponential":
        return ScaledEmbedding(vocab_size, d_model, scale_type="exp")
    elif geometry_type == "logarithmic":
        return ScaledEmbedding(vocab_size, d_model, scale_type="log")
    elif geometry_type == "euclidean":
        return nn.Embedding(vocab_size, d_model)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")