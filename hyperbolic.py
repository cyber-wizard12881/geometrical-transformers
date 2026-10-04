import torch
import torch.nn as nn

class HyperbolicEmbedding(nn.Module):
    """
    Hyperbolic embedding using Poincaré ball model.
    Analogy: Expanding tree branches — distances grow exponentially.
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        e = self.emb(x)
        # Map to hyperbolic space: norm < 1
        norm = torch.norm(e, dim=-1, keepdim=True)
        return e / (1 + norm)
