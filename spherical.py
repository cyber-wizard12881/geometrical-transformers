from torch import nn
import torch

class SphericalEmbedding(nn.Module):
    """
    Embedding constrained to a sphere (elliptic geometry).
    Analogy: Mapping countries on a globe.
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        e = self.emb(x)
        # Normalize to unit sphere
        return e / torch.norm(e, dim=-1, keepdim=True)
