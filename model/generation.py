from torchdiffeq import odeint_adjoint as odeint
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

import torch
import torch.nn as nn


def sample(f, y, t0, t1, model, method="dopri5", step=0.1, jump_t=None):
    t = torch.tensor([t0, t1], device=y.device)
    options = dict(
        step_size=step,
        jump_t=jump_t,
    )
    y = odeint(
        f, y, t, method=method, atol=1e-5, rtol=1e-5, adjoint_params=model.parameters(), options=options)[-1]
    return y


@torch.no_grad()
def generate(
        model: nn.Module,
        num_frames: int,
        device: torch.device,
        initial_frames: torch.Tensor = None,
        feats: torch.Tensor = None,
        feats_masks: torch.Tensor = None,
        batch_size: int = 1,
        method: str = "joint",
        odesolver: str = "euler",
        num_steps: int = 20,
        gamma: float = 0.0,
        past_horizon: int = -1,
        chunk_size: int = 1,
        verbose: bool = False):
    """

    :param model: the model to use for generation
    :param num_frames: number of frames to generate
    :param device: device to use for generation
    :param initial_frames: initial frames to condition generation on, [b l C H W], if None the sequence will be generated from scratch
    :param feats: control for the generated frames, [b k d h w], will be padded with zeros to match [b num_frames d h w]
    :param feats_masks: masks to indicate the selected controls, [b k 1 h w], will be padded with zeros to match [b num_frames 1 h w]
    :param batch_size: a bit redundant, but still needs to be set
    :param method: [joint, rollout], the way of generating, jointly denoising or sliding window with noise schedule, joint works better
    :param odesolver: ["euler", "rk4", "dopri5" ...], first 2 are fixed step solvers and fast, the latter is adaptive, more accurate but super slow
    :param num_steps: number of integration steps, should be set for fixed step solvers
    :param gamma: classifier-free guidance strength
    :param past_horizon: -1 for no horizon, otherwise restricts the past horizon the context frames can come from
    :param chunk_size: for decoding from latents, because the whole video doesn't fit into memory
    :param verbose: if True, a plot with the noise levels will be plotted
    """
    c = model.config["vector_field_regressor"]["state_size"]
    h, w = model.config["vector_field_regressor"]["state_res"]
    model.to(device)
    model_num_target_frames = model.config["num_target_frames"]
    model_num_context_frames = model.config["num_context_frames"]

    # Encode initial frames
    if initial_frames is not None:
        num_initial_frames = initial_frames.size(1)

        initial_frames = initial_frames.to(device)
        initial_y = model.encode(initial_frames)
        if num_initial_frames == 1:
            initial_y = torch.cat([torch.randn([batch_size, 1, c, h, w]).to(device), initial_y], dim=1)
            num_initial_frames = 2
        assert num_initial_frames >= 2
        generate = False
    else:
        num_initial_frames = 2
        initial_y = torch.randn([batch_size, 2, c, h, w]).to(device)
        generate = True

    pad = model_num_target_frames
    y = torch.randn([batch_size, num_frames + pad, c, h, w]).to(device)
    y = torch.cat([initial_y, y], dim=1)

    # Pad feats if necessary
    assert feats.size(1) == feats_masks.size(1)
    if feats.size(1) < num_frames + model_num_target_frames - 1:
        feats = torch.cat([
            feats,
            torch.zeros([
                batch_size,
                num_frames + pad - feats.size(1),
                model.feats_dim,
                h,
                w]).to(device)],
            dim=1)
        feats_masks = torch.cat([
            feats_masks,
            torch.zeros([
                batch_size,
                num_frames + pad - feats_masks.size(1),
                1,
                h,
                w]).to(device)],
            dim=1)

    # Generate frames
    def f(t, x, i, stage="joint", context_from=0):
        b = x.size(0)

        reference_frame_id = torch.tensor([i - 1], dtype=torch.long, device=x.device)
        target_frames_ids = i + \
                            torch.arange(model_num_target_frames, dtype=torch.long, device=x.device)
        if context_from == i - 1:
            context_frames_ids = torch.full(
                size=[model_num_context_frames], fill_value=context_from, dtype=torch.long, device=x.device)
        else:
            context_frames_ids = torch.randint(
                low=context_from, high=i - 1, size=[model_num_context_frames], dtype=torch.long, device=x.device)
        selected_frames_ids = torch.cat([context_frames_ids, reference_frame_id, target_frames_ids])
        selected_frames = x[:, selected_frames_ids]

        timestamps = torch.cat([
            torch.ones(i),
            torch.zeros(num_initial_frames + num_frames + pad - i)
        ]).to(x.device)

        if stage == "joint":
            timestamps[target_frames_ids] = \
                torch.full([model_num_target_frames], t, dtype=x.dtype, device=x.device)
        elif stage == "rolling":
            timestamps[target_frames_ids] = \
                torch.arange(model_num_target_frames, 0, -1, dtype=x.dtype, device=x.device) - 1
            timestamps[target_frames_ids] += \
                torch.full([model_num_target_frames], t, dtype=x.dtype, device=x.device)
            timestamps[target_frames_ids] /= model_num_target_frames
        elif stage == "warmup":
            timestamps[target_frames_ids] = \
                t * torch.arange(model_num_target_frames, 0, -1, dtype=x.dtype, device=x.device)
            timestamps[target_frames_ids] /= model_num_target_frames
        else:
            raise NotImplementedError

        if verbose:
            clear_output(wait=True)
            plt.bar(
                x=np.arange(num_initial_frames + num_frames + pad),
                height=timestamps.cpu().numpy())
            plt.ylim(0, 1)
            plt.show()
        timestamps = timestamps[selected_frames_ids]

        v = model.predict_vector_field(
            x=selected_frames,
            indices=selected_frames_ids.unsqueeze(0).repeat(b, 1),
            t=timestamps.unsqueeze(0).repeat(b, 1),
            feats=feats[:, target_frames_ids - num_initial_frames],
            feats_masks=feats_masks[:, target_frames_ids - num_initial_frames])

        if gamma != 0.0:
            uncond_v = model.predict_vector_field(
                x=selected_frames,
                indices=selected_frames_ids.unsqueeze(0).repeat(b, 1),
                t=timestamps.unsqueeze(0).repeat(b, 1),
                feats=feats[:, target_frames_ids - num_initial_frames],
                feats_masks=torch.zeros_like(feats_masks[:, target_frames_ids - num_initial_frames]).to(x.device))
            v = (1 + gamma) * v - gamma * uncond_v

        out = torch.zeros_like(x).to(x.device).to(x.dtype)
        out[:, target_frames_ids] = v[:, -model_num_target_frames:]

        if stage == "warmup":
            out[:, target_frames_ids] = out[:, target_frames_ids] *\
                torch.arange(
                    model_num_target_frames,
                    0,
                    -1,
                    dtype=x.dtype,
                    device=x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            out[:, target_frames_ids] = out[:, target_frames_ids] / model_num_target_frames

        return out

    # Sample first
    h = 1.0 / num_steps
    y = sample(partial(f, i=num_initial_frames, stage="joint"), y, 0.0, 1.0, model, method=odesolver, step=h)
    if num_frames > model_num_target_frames:
        initial_frame_id = 0 if not generate else 2
        if method == "rollout":
            # Warmup
            if past_horizon != -1:
                context_from = max(initial_frame_id, num_initial_frames + model_num_target_frames - past_horizon)
            else:
                context_from = initial_frame_id
            y = sample(
                partial(
                    f, i=num_initial_frames + model_num_target_frames, stage="warmup", context_from=context_from),
                y, 0.0, 1.0, model, method=odesolver, step=h)
            # Rolling
            for i in range(1, num_frames - model_num_target_frames):
                if past_horizon != -1:
                    context_from = max(
                        initial_frame_id, num_initial_frames + model_num_target_frames + i - past_horizon)
                else:
                    context_from = initial_frame_id
                y = sample(
                    partial(f, i=num_initial_frames + model_num_target_frames + i,
                            stage="rolling", context_from=context_from),
                    y, 0.0, 1.0, model, method=odesolver, step=h)
        elif method == "joint":
            for i in range(0, num_frames - model_num_target_frames, model_num_target_frames):
                if past_horizon != -1:
                    context_from = max(
                        initial_frame_id, num_initial_frames + model_num_target_frames + i - past_horizon)
                else:
                    context_from = initial_frame_id
                y = sample(
                    partial(f, i=num_initial_frames + model_num_target_frames + i,
                            stage="joint", context_from=context_from),
                    y, 0.0, 1.0, model, method=odesolver, step=h)
        else:
            raise NotImplementedError
    y = y[:, :num_initial_frames + num_frames]

    generated_frames = []
    if initial_frames is not None:
        generated_frames.append(initial_frames)
    for i in range(0, num_frames, chunk_size):
        generated_chunk = model.decode(y[:, i+num_initial_frames:i+num_initial_frames+chunk_size])
        generated_frames.append(generated_chunk)
    generated_frames = torch.cat(generated_frames, dim=1)

    return generated_frames.cpu()
