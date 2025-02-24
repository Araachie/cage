import math
from einops import rearrange

import torch
import torch.nn as nn


class TimeDependentMultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            dropout: float):
        super(TimeDependentMultiHeadAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.time_qkv = nn.Linear(dim, 3 * self.head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(dim)

    def forward(
            self,
            x: torch.Tensor,
            time_embeddings: torch.Tensor) -> torch.Tensor:
        """

        :param x: [b n d]
        :param time_embeddings: [b d] time embeddings
        :return: [b n d]
        """

        # Reshape the input
        n = x.size(1)

        # Project to Q, K, V and reshape to heads
        q, k, v = self.qkv(x).chunk(3, -1)  # [b n d]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)  # each is [b h n d]
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)  # each is [b h n d]
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)  # each is [b h n d]

        # Process the time embeddings
        qt, kt, vt = self.time_qkv(time_embeddings).unsqueeze(1).unsqueeze(1).chunk(3, -1)  # each [b 1 1 d]

        # Gather
        q = q + qt  # [b h n d]
        k = k + kt  # [b h n d]
        v = v + vt  # [b h n d]

        # Calculate QK^T
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale  # [b h n n]

        # Apply softmax and dropout
        attn = self.dropout(torch.softmax(attn, dim=-1))  # [b h n n]

        # Calculate AV
        out = torch.matmul(attn, v)  # [b h n d]

        # Reshape output
        out = rearrange(out, "b h n d -> b n (h d)")

        return out


class TimeDependentAttentionLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            dropout: float,
            fc_dim: int):
        super(TimeDependentAttentionLayer, self).__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = TimeDependentMultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, dim)
        )

    def forward(
            self,
            x: torch.Tensor,
            time_embeddings: torch.Tensor) -> torch.Tensor:
        """

        :param x: [b n d]
        :param time_embeddings: [b d]
        :return: [b n d]
        """

        # Apply attention with skip-connection
        x1 = x + self.attn(
            x=self.ln1(x),
            time_embeddings=time_embeddings)

        # Apply feed-forward with skip-connection
        x2 = x1 + self.fc(self.ln2(x1))

        return x2
