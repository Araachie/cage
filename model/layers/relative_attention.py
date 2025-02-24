from typing import Optional
import math
from einops import rearrange

import torch
import torch.nn as nn


class RelativePosition(nn.Module):
    def __init__(self, dim: int, max_relative_position: int):
        super(RelativePosition, self).__init__()

        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, dim), requires_grad=True)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, x: torch.Tensor, pos: torch.LongTensor, chunk_size: int) -> torch.Tensor:
        """
        :param x: [b h N d] where N = l * n
        :param pos: [b l l] relative positions
        :param chunk_size: n
        :return: [b h N N] the product of x and pos embeddings
        """

        # Lookup embeddings in the table
        lookup_indices = torch.clamp(pos, -self.max_relative_position, self.max_relative_position)  # [b l l]
        lookup_indices = lookup_indices + self.max_relative_position  # [b l l]
        embeddings = self.embeddings_table[lookup_indices]  # [b l l d]

        # Reshape x
        assert x.size(2) == pos.size(1) * chunk_size
        x_chunked = rearrange(x, "b h (l n) d -> b h n l d", n=chunk_size)  # [b h n l d]

        # Multiply x by embeddings
        embeddings_t = embeddings.permute(0, 1, 3, 2)  # [b l d l]
        prod = torch.matmul(x_chunked.unsqueeze(4), embeddings_t.unsqueeze(1).unsqueeze(2))  # [b h n l 1 l]

        # Reshape prod
        prod = prod.squeeze(4)  # [b h n l l]
        prod = prod.permute(0, 1, 3, 2, 4)  # [b h l n l]
        prod = prod.unsqueeze(5).expand(-1, -1, -1, -1, -1, chunk_size)  # [b h l n l n]
        prod = rearrange(prod, "b h q n k m -> b h (q n) (k m)")  # [b h N N]
        assert prod.size(2) == prod.size(3)

        return prod


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            dropout: float,
            max_relative_position: int = None):
        super(RelativeMultiHeadAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.max_relative_position = max_relative_position

        self.relative_position = RelativePosition(dim=self.head_dim, max_relative_position=max_relative_position)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.time_qkv = nn.Linear(dim, 3 * self.head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(dim)

    def forward(
            self,
            x: torch.Tensor,
            time_embeddings: torch.Tensor,
            pos: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param x: [b l n d]
        :param time_embeddings: [b l d] time embeddings
        :param pos: [b l l] relative positions
        :param mask: [b l l] eg causal
        :return: [b l n d]
        """

        # Reshape the input
        n = x.size(2)

        # Project to Q, K, V and reshape to heads
        q, k, v = self.qkv(x).chunk(3, -1)  # [b l n d]
        q = rearrange(q, "b l n (h d) -> b h l n d", h=self.num_heads)  # each is [b h l n d]
        k = rearrange(k, "b l n (h d) -> b h l n d", h=self.num_heads)  # each is [b h l n d]
        v = rearrange(v, "b l n (h d) -> b h l n d", h=self.num_heads)  # each is [b h l n d]

        # Process the time embeddings
        assert time_embeddings.size(1) == x.size(1)
        qt, kt, vt = self.time_qkv(time_embeddings).unsqueeze(2).unsqueeze(1).chunk(3, -1)  # each [b 1 l 1 d]

        # Gather
        q = rearrange(q + qt, "b h l n d -> b h (l n) d")  # [b h N d] where N = l * n
        k = rearrange(k + kt, "b h l n d -> b h (l n) d")  # [b h N d]
        v = rearrange(v + vt, "b h l n d -> b h (l n) d")  # [b h N d]

        # Calculate QK^T
        qk_t = torch.matmul(q, k.permute(0, 1, 3, 2))  # [b h N N]

        # Calculate QR
        qr = self.relative_position(q, pos, chunk_size=n)  # [b h N N]

        # Calculate attention
        attn = (qk_t + qr) / self.scale  # [b h N N]

        # Apply mask
        if mask is not None:
            exp_mask = mask.unsqueeze(2).unsqueeze(4).expand(-1, -1, n, -1, n)  # [b l n l n]
            exp_mask = rearrange(exp_mask, "b l n k m -> b (l n) (k m)")  # [b N N]
            exp_mask = exp_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [b h N N]
            attn = attn.masked_fill(exp_mask == 0, torch.finfo(torch.bfloat16).min)

        # Apply softmax and dropout
        attn = self.dropout(torch.softmax(attn, dim=-1))  # [b h N N]

        # Calculate AV
        out = torch.matmul(attn, v)  # [b h N d]

        # Reshape output
        out = rearrange(out, "b h (l n) d -> b l n (h d)", n=n)

        return out


class RelativeAttentionLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            dropout: float,
            fc_dim: int,
            max_relative_position: int,
            causal: bool = False):
        super(RelativeAttentionLayer, self).__init__()

        self.causal = causal

        self.ln1 = nn.LayerNorm(dim)
        self.attn = RelativeMultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            max_relative_position=max_relative_position)

        self.ln2 = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, dim)
        )

    def forward(
            self,
            x: torch.Tensor,
            time_embeddings: torch.Tensor,
            indices: torch.Tensor) -> torch.Tensor:
        """

        :param x: [b l n d]
        :param time_embeddings: [b l d]
        :param indices: [b l] indices
        :return: [b l n d]
        """

        # Calculate pos matrix
        pos = indices[:, None, :] - indices[:, :, None]

        # Apply attention with skip-connection
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)
        x1 = x + self.attn(
            x=self.ln1(x),
            time_embeddings=time_embeddings,
            pos=pos,
            mask=mask)

        # Apply feed-forward with skip-connection
        x2 = x1 + self.fc(self.ln2(x1))

        return x2
