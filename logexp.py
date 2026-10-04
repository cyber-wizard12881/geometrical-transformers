import torch
from torch import nn

class ScaledEmbedding(nn.Module):
    """
    Embedding with exponential/logarithmic scaling.
    Analogy: Viral growth or diminishing returns.
    """
    def __init__(self, vocab_size, dim, scale_type="exp"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.scale_type = scale_type

    def forward(self, x):
        e = self.emb(x)
        if self.scale_type == "exp":
            return torch.exp(e)
        elif self.scale_type == "log":
            return torch.log(torch.abs(e) + 1)
        else:
            return e
