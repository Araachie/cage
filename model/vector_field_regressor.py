import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from lutils.configuration import Configuration
from model.layers.position_encoding import build_position_encoding
from model.layers.relative_attention import RelativeAttentionLayer
from model.layers.time_attention import TimeDependentAttentionLayer


def timestamp_embedding(timesteps, dim, scale=200, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param scale: a premultiplier of timesteps
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: whether to repeat only the values in timesteps along the 2nd dim
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = scale * timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(scale * timesteps, 'b -> b d', d=dim)
    return embedding


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    @staticmethod
    def forward(*x: torch.Tensor) -> torch.Tensor:
        return x[0]


class VectorFieldRegressor(nn.Module):
    def __init__(
            self,
            num_frames: int,
            depth: int,
            mid_depth: int,
            state_size: int,
            state_res: Tuple[int, int],
            feats_dim: int,
            inner_dim: int,
            num_heads: int,
            fc_dim: int,
            dropout: float,
            max_relative_position: int,
            causal: bool,
            masking_ratio: float):
        super(VectorFieldRegressor, self).__init__()

        self.num_frames = num_frames
        self.state_size = state_size
        self.state_height = state_res[0]
        self.state_width = state_res[1]
        self.inner_dim = inner_dim
        self.masking_ratio = masking_ratio

        self.position_encoding = build_position_encoding(self.inner_dim, position_embedding_name="learned")

        self.project_in = nn.Sequential(
            nn.Linear(self.state_size, self.inner_dim)
        )

        self.msk_token = nn.Parameter(torch.randn(feats_dim), requires_grad=True)
        self.feats_project_in = nn.Sequential(
            nn.Linear(feats_dim, self.inner_dim)
        )

        self.time_projection = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.inner_dim)
        )

        def build_spatio_temporal_layer(d_model: int, cross: bool):
            spatial = TimeDependentAttentionLayer(
                dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                fc_dim=fc_dim)
            temporal = RelativeAttentionLayer(
                dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                fc_dim=fc_dim,
                max_relative_position=max_relative_position,
                causal=causal)
            cross = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=fc_dim,
                dropout=dropout,
                norm_first=True,
                batch_first=True) if cross else DummyLayer()
            return nn.ModuleDict({"spatial": spatial, "temporal": temporal, "cross": cross})

        self.main_blocks = nn.ModuleList()
        mid_blocks_ids = [i + (depth - mid_depth) // 2 for i in range(mid_depth)]
        for i in range(depth):
            self.main_blocks.append(build_spatio_temporal_layer(self.inner_dim, cross=i in mid_blocks_ids))

        self.project_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.GELU(),
            nn.LayerNorm(self.inner_dim)
        )
        self.out_conv = nn.Sequential(
            Rearrange("b l (h w) c -> (b l) c h w", h=self.state_height),
            nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1, bias=False),
            Rearrange("(b l) c h w -> b l c h w", l=self.num_frames)
        )

    def forward(
            self,
            input_latents: torch.Tensor,
            indices: torch.Tensor,
            timestamps: torch.Tensor,
            feats: torch.Tensor,
            feats_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param input_latents: [b, l, c, h, w]
        :param indices: [b, l]
        :param timestamps: [b, l]
        :param feats: [b L C h w]
        :param feats_masks: [b L 1 h w]
        :return: [b, c, h, w]
        """

        b = input_latents.size(0)
        l = input_latents.size(1)
        num_spatial_tokens = input_latents.size(3) * input_latents.size(4)

        # Fetch timestamp tokens
        t_batched = timestamp_embedding(rearrange(timestamps, "b l -> (b l)"), dim=self.inner_dim)  # [bl d]
        t = rearrange(t_batched, "(b l) d -> b l d", l=l)  # [b l d]

        # Calculate position embedding
        pos = self.position_encoding(input_latents[:, -1])
        pos = rearrange(pos, "b c h w -> b (h w) c").unsqueeze(1)  # [b 1 n d]

        # Mask input tokens
        flat_input_latents = rearrange(input_latents, "b l c h w -> b l (h w) c")  # [b l n c]
        x_masked, pos_masked, selected_masks, ids_restore = self.random_masking(
            flat_input_latents,
            pos.repeat(1, l, 1, 1))  # [b l m c], [b l m d], [b l n], [b l n]
        num_selected_tokens = x_masked.size(2)
        selected_masks = rearrange(
            selected_masks.unsqueeze(2), "b l c (h w) -> b l c h w", h=self.state_height)  # [b l 1 h w]

        # Pad feats and masks
        feats_masks = torch.cat(
            [torch.zeros(b, l - feats_masks.size(1), 1, self.state_height, self.state_width, device=feats_masks.device),
             feats_masks], dim=1)  # [b l 1 h w]
        feats = torch.cat([
            self.msk_token[None, None, :, None, None].repeat(
                b, l - feats.size(1), 1, self.state_height, self.state_width),
            feats], dim=1)

        # Add msk tokens to feats and project
        feats = feats_masks * feats + (1 - feats_masks) * self.msk_token[None, None, :, None, None]
        cond = pos.detach() + self.feats_project_in(rearrange(feats, "b l c h w -> b l (h w) c"))  # [b l n d]

        # Build input tokens
        x = self.project_in(x_masked)  # [b l m d]
        x = x + pos_masked  # [b l m d]

        # Propagate through the main network
        for block in self.main_blocks:
            x = self.propagate_through_spatio_temporal_block(x, t, indices, cond, block)

        # Project out
        x = self.project_out(x)  # [b l m d]
        out = torch.zeros(
            [b, l, num_spatial_tokens - num_selected_tokens, self.inner_dim]).to(x.device).to(x.dtype)  # [b n-m d]
        out = torch.cat([x, out], dim=2)  # [b l n d]
        out = torch.gather(out, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, self.inner_dim))  # [b l n d]
        out = self.out_conv(out)  # [b l c h w]

        return out, selected_masks

    def random_masking(
            self,
            x: torch.Tensor,
            pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x: [b l n c]
        :param pos: [b l n d]
        """
        B, L, N, C = x.shape  # batch, length_clip, num_tokens, dim
        D = pos.size(3)
        num_selected_tokens = int(N * (1 - self.masking_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).unsqueeze(1).repeat(1, L, 1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :num_selected_tokens]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, C))
        pos_masked = torch.gather(pos, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([B, L, N], device=x.device)
        mask[:, :, :num_selected_tokens] = 1

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return x_masked, pos_masked, mask, ids_restore

    @staticmethod
    def propagate_through_spatio_temporal_block(
            x: torch.Tensor,
            time_embeddings: torch.Tensor,
            indices: torch.Tensor,
            cond: torch.Tensor,
            block: nn.Module) -> torch.Tensor:
        """

        :param x: [b l n d]
        :param time_embeddings: [b l d]
        :param indices: [b l]
        :param cond: [b l m d]
        :param block: module dict with spatial, temporal submodules
        """
        b = x.size(0)
        time_embeddings_batched = rearrange(time_embeddings, "b l d -> (b l) d")  # [bl d]
        x_batched = rearrange(x, "b l n d -> (b l) n d")  # [bl n d]
        x_batched = block["spatial"](x_batched, time_embeddings_batched)
        x_chunked = rearrange(x_batched, "(b l) n d -> b l n d", b=b)  # [b l n d]
        x_chunked = block["temporal"](x_chunked, time_embeddings, indices)
        x_batched = rearrange(x_chunked, "b l n d -> (b l) n d")  # [bl n d]
        cond_batched = rearrange(cond, "b l m d -> (b l) m d")  # [bl m d]
        x_batched = block["cross"](x_batched, cond_batched)  # [bl n d]
        x_chunked = rearrange(x_batched, "(b l) n d -> b l n d", b=b)  # [b l n d]
        return x_chunked


def build_vector_field_regressor(config: Configuration):
    return VectorFieldRegressor(
        num_frames=config["num_frames"],
        state_size=config["state_size"],
        state_res=config["state_res"],
        feats_dim=config["feats_dim"],
        inner_dim=config["inner_dim"],
        depth=config["depth"],
        mid_depth=config["mid_depth"],
        fc_dim=config["fc_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        max_relative_position=config["max_relative_position"],
        masking_ratio=config["masking_ratio"],
        causal=config["causal"]
    )
