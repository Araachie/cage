from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.vector_field_regressor import build_vector_field_regressor
from model.vqgan.taming.autoencoder import vq_f8_ddconfig, vq_f8_small_ddconfig, vq_f16_ddconfig, VQModelInterface
from model.vqgan.vqvae import build_vqvae
from model.utils import spatial_transform
from model.mae import models_mae

import clip


class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config
        self.sigma = config["sigma"]

        if config["autoencoder"]["type"] == "ours":
            self.ae = build_vqvae(
                config=config["autoencoder"],
                convert_to_sequence=True)
            self.ae.backbone.load_from_ckpt(config["autoencoder"]["ckpt_path"])
        else:
            if config["autoencoder"]["config"] == "f8":
                ae_config = vq_f8_ddconfig
            elif config["autoencoder"]["config"] == "f8_small":
                ae_config = vq_f8_small_ddconfig
            else:
                ae_config = vq_f16_ddconfig
            self.ae = VQModelInterface(ae_config, config["autoencoder"]["ckpt_path"])

        if self.config["feature_backbone"] == "dino":
            self.dino = torch.hub.load('facebookresearch/dinov2', config["dino_version"])
            self.feats_dim = self.dino.embed_dim * self.config["num_dino_layers"]
        elif self.config["feature_backbone"] == "mae":
            self.mae = getattr(models_mae, self.config["mae_arch"])()
            mae_checkpoint = torch.load(self.config["mae_ckpt"], map_location='cpu')
            self.mae.load_state_dict(mae_checkpoint['model'], strict=False)
            self.feats_dim = self.mae.embed_dim
        elif self.config["feature_backbone"] == "clip":
            self.clip, self.clip_preprocess = clip.load(self.config["clip_arch"])
            self.feats_dim = self.clip.visual.conv1.weight.shape[0]
        else:
            raise NotImplementedError

        self.config["vector_field_regressor"]["num_frames"] = 1 + \
            self.config["num_target_frames"] + \
            self.config["num_context_frames"]
        self.config["vector_field_regressor"]["feats_dim"] = self.feats_dim
        self.vector_field_regressor = build_vector_field_regressor(
            config=self.config["vector_field_regressor"])

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu")

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            print("is_ddp")
            if not isinstance(self, nn.parallel.DistributedDataParallel):
                print("self is not ddp")
                state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
            else:
                state = loaded_state["model"]
        else:
            if isinstance(self, nn.parallel.DistributedDataParallel):
                state = {f"module.{k}": v for k, v in loaded_state["model"].items()}
            else:
                state = loaded_state["model"]

        # Support old checkpoints
        state = {k.replace("dino_project_in", "feats_project_in"): v for k, v in state.items()}

        dmodel = self.module if isinstance(self, torch.nn.parallel.DistributedDataParallel) else self
        dmodel.load_state_dict(state)

    def forward(
            self,
            observations: torch.Tensor) -> DictWrapper[str, Any]:
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        H, W = observations.shape[-2:]
        num_context_frames = self.config["num_context_frames"]
        num_target_frames = self.config["num_target_frames"]
        num_frames = num_context_frames + num_target_frames + 1
        assert num_observations >= num_frames

        # Sample frames for denoising
        reference_frames_indices = torch.randint(low=num_context_frames, high=num_observations - num_target_frames,
                                                 size=[batch_size])
        context_frames_indices = torch.sort(
            torch.stack(
                [torch.randint(low=0, high=ref_id if ref_id != 0 else 1, size=[num_context_frames])
                for ref_id in reference_frames_indices]),
            dim=-1)[0]
        target_frames_indices = torch.arange(num_target_frames).unsqueeze(0) + reference_frames_indices.unsqueeze(1) + 1
        selected_frames_indices = torch.cat(
            [context_frames_indices, reference_frames_indices.unsqueeze(1), target_frames_indices], dim=1)
        selected_frames_indices = selected_frames_indices.to(observations.device)
        selected_frames = observations[torch.arange(batch_size).unsqueeze(1), selected_frames_indices]

        # Encode observations to latent codes
        target_latents = self.encode(selected_frames)
        h, w = target_latents.shape[-2:]

        # Apply random spatial transform to target frames
        if not self.config.get("no_scale_inv", False):
            z_where = torch.cat([
                torch.rand(batch_size * num_target_frames, 2) * 0.5 + 0.5,
                0.75 * (torch.rand(batch_size * num_target_frames, 2) * 2 - 1)
            ], dim=1).to(selected_frames.device)
        else:
            z_where = torch.cat([
                torch.ones(batch_size * num_target_frames, 2),
                torch.zeros(batch_size * num_target_frames, 2)
            ], dim=1).to(selected_frames.device)
        flat_transformed_target_frames = spatial_transform(
            rearrange(selected_frames[:, -num_target_frames:], "b l c h w -> (b l) c h w"),
            z_where,
            [batch_size * num_target_frames, 3, H, W],
            inverse=False,
            padding_mode="border",
            mode="bilinear")
        flat_allowed_masks = spatial_transform(
            torch.ones(batch_size * num_target_frames, 1, h, w).to(selected_frames.device),
            z_where,
            [batch_size * num_target_frames, 1, h, w],
            inverse=True,
            padding_mode="zeros",
            mode="nearest")
        transformed_target_frames = rearrange(flat_transformed_target_frames, "(b l) c h w -> b l c h w", b=batch_size)
        allowed_masks = rearrange(flat_allowed_masks, "(b l) c h w -> b l c h w", b=batch_size)

        # Calculate features
        feats = self.get_features(transformed_target_frames)
        feats_dim = feats.size(2)

        # Inverse transform features
        flat_feats = spatial_transform(
            rearrange(feats, "b l c h w -> (b l) c h w"),
            z_where,
            [batch_size * num_target_frames, feats_dim, h, w],
            inverse=True,
            padding_mode="zeros",
            mode="nearest")
        feats = rearrange(flat_feats, "(b l) c h w -> b l c h w", b=batch_size)

        # Build random mask for feats
        feats_masks = self.sparsify(feats, p=self.config["feats_select_prob"], allowed=allowed_masks)

        # Sample input latents and calculate target vectors
        noise = torch.randn_like(target_latents).to(target_latents.dtype).to(target_latents.device)
        timestamps = torch.cat([
            torch.ones(batch_size, 2, 1, 1, 1),
            torch.rand(batch_size, num_frames - 2, 1, 1, 1) if not self.config.get("same_t", False)
            else torch.rand(batch_size, 1, 1, 1, 1).repeat(1, num_frames - 2, 1, 1, 1)], dim=1)
        timestamps = timestamps.to(target_latents.dtype).to(target_latents.device)

        # Run in full precision for stability
        with torch.amp.autocast(device_type="cuda", enabled=False):
            input_latents = (1 - (1 - self.sigma) * timestamps.float()) * noise.float() + \
                            timestamps.float() * target_latents.float()
            target_vectors = target_latents.float() - (1 - self.sigma) * noise.float()
        target_vectors = target_vectors
        timestamps = timestamps.squeeze(4).squeeze(3).squeeze(2)

        # Randomized training, like in classifier-free guidance
        if torch.rand(1)[0].item() < self.config["no_context_p"]:
            if torch.rand(1)[0].item() < self.config["no_reference_p"]:
                input_latents[:, :num_context_frames + 1] = torch.randn_like(
                    input_latents[:, :num_context_frames + 1], device=input_latents.device)
            else:
                input_latents[:, :num_context_frames] = torch.randn_like(
                    input_latents[:, :num_context_frames], device=input_latents.device)

        # Predict vectors
        reconstructed_vectors, selected_masks = self.vector_field_regressor(
            input_latents=input_latents,
            indices=selected_frames_indices,
            timestamps=timestamps,
            feats=feats,
            feats_masks=feats_masks)

        target_vectors = target_vectors[:, -num_target_frames:]
        reconstructed_vectors = reconstructed_vectors[:, -num_target_frames:]
        selected_masks = selected_masks[:, -num_target_frames:]

        return DictWrapper(
            # Inputs
            observations=observations,

            # Data for loss calculation
            reconstructed_vectors=reconstructed_vectors,
            target_vectors=target_vectors,
            selected_masks=selected_masks,

            # Log
            num_controls=feats_masks.sum([2, 3, 4]).mean([0, 1]))

    @torch.no_grad()
    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """

        :param observations: [b l C H W]
        :return: [b l c h w]
        """

        self.ae.eval()
        if self.config["autoencoder"]["type"] == "ours":
            latents = self.ae(observations).latents
        else:
            batch_size = observations.size(0)
            flat_input_frames = rearrange(observations, "b l c h w -> (b l) c h w")
            flat_latents = self.ae.encode(flat_input_frames)
            latents = rearrange(flat_latents, "(b l) c h w -> b l c h w", b=batch_size)

        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """

        :param latents: [b l c h w]
        :return: [b l C H W]
        """

        batch_size = latents.size(0)
        flat_latents = rearrange(latents, "b l c h w -> (b l) c h w")
        if self.config["autoencoder"]["type"] == "ours":
            images = self.ae.backbone.decode_from_latents(flat_latents)
        else:
            images = self.ae.decode(flat_latents)
        images = rearrange(images, "(b l) c h w -> b l c h w", b=batch_size)

        return images

    @torch.no_grad()
    def get_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.config["feature_backbone"] == "dino":
            return self.get_dino_features(x, n=self.config["num_dino_layers"])
        elif self.config["feature_backbone"] == "mae":
            return self.get_mae_features(x, **kwargs)
        elif self.config["feature_backbone"] == "clip":
            return self.get_clip_features(x, **kwargs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_dino_features(self, x: torch.Tensor, n: int = 1):
        """

        :param x: [b l c h w]
        :param n: number of layers from the last
        :return: [b l d h w]
        """

        batch_size = x.size(0)

        dino_feats = self.dino.get_intermediate_layers(
            F.interpolate(
                rearrange(x, "b l c h w -> (b l) c h w"),
                size=[224, 224], mode="bilinear", align_corners=False),
            n=n,
            reshape=True)
        dino_feats = torch.cat(dino_feats, dim=1)
        dino_feats = F.interpolate(
            dino_feats,
            size=self.config["vector_field_regressor"]["state_res"], mode="bilinear", align_corners=False)
        dino_feats = rearrange(dino_feats, "(b l) c h w -> b l c h w", b=batch_size)

        return dino_feats

    @torch.no_grad()
    def get_mae_features(self, x: torch.Tensor):
        """

        :param x: [b l c h w]
        :return: [b l d h w]
        """

        batch_size = x.size(0)

        mae_feats = self.mae.forward_encoder(
            F.interpolate(
                rearrange(x, "b l c h w -> (b l) c h w"),
                size=[self.mae.img_size, self.mae.img_size], mode="bilinear", align_corners=False),
            mask_ratio=0.0,
        )[0][:, 1:]
        mae_feats = rearrange(mae_feats, "b (h w) c -> b c h w", h=self.mae.img_size // self.mae.patch_size)
        mae_feats = F.interpolate(
            mae_feats,
            size=self.config["vector_field_regressor"]["state_res"], mode="bilinear", align_corners=False)
        mae_feats = rearrange(mae_feats, "(b l) c h w -> b l c h w", b=batch_size)

        return mae_feats

    @torch.no_grad()
    def get_clip_features(self, x: torch.Tensor):
        """

        :param x: [b l c h w]
        :return: [b l d h w]
        """

        batch_size = x.size(0)
        vision_patch_size = self.clip.visual.conv1.weight.shape[-1]
        grid_size = round((self.clip.visual.positional_embedding.shape[0] - 1) ** 0.5)
        image_res = vision_patch_size * grid_size
        dtype = x.dtype

        x = x.to(torch.float16)
        x = F.interpolate(
                rearrange(x, "b l c h w -> (b l) c h w"),
                size=[image_res, image_res], mode="bicubic", align_corners=False)

        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = x[:, 1:]

        x = rearrange(x, "b (h w) c -> b c h w", h=grid_size)
        x = F.interpolate(
            x,
            size=self.config["vector_field_regressor"]["state_res"], mode="bilinear", align_corners=False)
        x = rearrange(x, "(b l) c h w -> b l c h w", b=batch_size)

        return x.to(dtype)

    @staticmethod
    def sparsify(x: torch.Tensor, p: float, allowed: torch.Tensor = None) -> torch.Tensor:
        """

        :param x: [b l c h w]
        :param p: probability of the token to be selected
        :param allowed: [b l 1 h w] the region from which to select the tokens
        :return: [b l 1 h w] the selected mask
        """

        b, l, _, h, w = x.shape

        prob_map = torch.full([b, l, 1, h, w], p, device=x.device)
        if allowed is not None:
            prob_map = allowed * prob_map

        masks = torch.bernoulli(prob_map)

        return masks

    def predict_vector_field(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            indices: torch.Tensor,
            feats: torch.Tensor,
            feats_masks: torch.Tensor) -> torch.Tensor:
        """

        :param x: [b l c h w]
        :param t: [b l]
        :param indices: [b l]
        :param feats: [b L C h w]
        :param feats_masks: [b L 1 h w]
        :return: [b l c h w]
        """

        mr = self.vector_field_regressor.masking_ratio
        self.vector_field_regressor.masking_ratio = 0.0
        reconstructed_vectors, _ = self.vector_field_regressor(
            input_latents=x,
            indices=indices,
            timestamps=t,
            feats=feats,
            feats_masks=feats_masks)
        self.vector_field_regressor.masking_ratio = mr

        return reconstructed_vectors
