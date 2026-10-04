import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from geomapper import map_2_geometry

# -------------------------------
# 1. Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    """
    Adds information about word order to embeddings.
    Analogy: Imagine a playlist of songs. The songs themselves (embeddings)
    tell you what they are, but the track number (positional encoding)
    tells you the order they should be played in.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to embeddings
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# 2. Scaled Dot-Product Attention
# -------------------------------
class ScaledDotProductAttention(nn.Module):
    """
    Core attention mechanism.
    Analogy: Imagine you're at a party trying to listen to multiple conversations.
    You focus more on the voices (words) that are relevant to your topic,
    and tune out the rest. Attention scores decide who you listen to.
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

# -------------------------------
# 3. Multi-Head Attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads look at the same sentence differently.
    Analogy: Like a team of detectives — one looks at motives, another at alibis,
    another at fingerprints. Together they form a complete picture.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        out = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(out)

# -------------------------------
# 4. Position-wise Feed Forward
# -------------------------------
class PositionwiseFeedForward(nn.Module):
    """
    Applies a small neural network to each word vector independently.
    Analogy: Like polishing each gem individually after sorting them —
    every word gets refined before moving on.
    """
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# -------------------------------
# 5. Encoder Layer
# -------------------------------
class EncoderLayer(nn.Module):
    """
    One block of the encoder:
    - Self-attention (understand relationships between words)
    - Feed-forward (refine meaning)
    - Residual + LayerNorm (keep things stable)
    Analogy: Like reading a paragraph carefully, then summarizing it neatly.
    """
    def __init__(self, d_model, num_heads, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

# -------------------------------
# 6. Decoder Layer
# -------------------------------
class DecoderLayer(nn.Module):
    """
    One block of the decoder:
    - Masked self-attention (look only at past words)
    - Cross-attention (listen to encoder’s understanding)
    - Feed-forward
    Analogy: Like writing a story word by word, while constantly checking
    your notes (encoder output) to stay accurate.
    """
    def __init__(self, d_model, num_heads, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)

        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + cross_attn_out)

        ff_out = self.ff(x)
        return self.norm3(x + ff_out)

# -------------------------------
# 7. Full Transformer
# -------------------------------
class Transformer(nn.Module):
    """
    Full Transformer model with encoder + decoder stacks.
    Analogy: Think of it as a translator:
    - Encoder: listens carefully to the source language.
    - Decoder: speaks fluently in the target language.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, geometry_type="euclidean"):
        super().__init__()
        self.src_embedding = map_2_geometry(geometry_type, src_vocab_size, d_model)
        self.tgt_embedding = map_2_geometry(geometry_type, tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed + add positional info
        src = self.pos_encoding(self.src_embedding(src))
        tgt = self.pos_encoding(self.tgt_embedding(tgt))

        # Encoder stack
        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # Decoder stack
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        # Final prediction
        return self.fc_out(dec_out)

# -------------------------------
# 8. Example Usage
# -------------------------------
if __name__ == "__main__":
    src = torch.randint(0, 1000, (2, 5))  # batch=2, src_len=5
    tgt = torch.randint(0, 1000, (2, 5))  # batch=2, tgt_len=5

    model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000, d_model=64, num_heads=4, num_layers=2)
    output = model(src, tgt)

    print("Output shape:", output.shape)
    # Expected: (batch_size, tgt_len, tgt_vocab_size)
