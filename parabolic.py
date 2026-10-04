import torch
import torch.nn as nn

class ParabolicEmbedding(nn.Module):
    """
    Embedding layer with parabolic transformation.
    Each embedding vector is squared element-wise, 
    creating a parabolic warp of the input space.
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        e = self.emb(x)
        # Apply parabolic transformation (element-wise square)
        return torch.pow(e, 2)
