from dataset import VideoDataset
from model import Model
from lutils.configuration import Configuration

from PIL import Image
import torch

import matplotlib.pyplot as plt

from IPython.display import clear_output
import time


def to_image(x):
    return (127.5 * (torch.clamp(x, -1, 1) + 1)).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def show_video(video):
    out = []
    for i in range(video.shape[1]):
        clear_output(wait=True)
        plt.figure(figsize=(5 * video.shape[0], 5))
        for j, f in enumerate(video[:, i]):
            im = Image.fromarray(to_image(f))
            out.append(im)
            plt.subplot(1, video.shape[0], j + 1)
            plt.imshow(im)
            plt.annotate(f"{i}", (im.size[0] - im.size[0] / 10, 20), fontsize=16, c="r")

        plt.show()
        time.sleep(0.001)

    return out


def load_model_and_data(config_name: str, ckpt_path: str, random_time: bool = False):
    config_path = f"configs/{config_name}.yaml"

    config = Configuration(config_path)

    if config_name.startswith("bair"):
        data_path = "./data/BAIR_h5/test"
        im_res = 256
    elif config_name.startswith("clevrer"):
        data_path = "./data/CLEVRER_h5/test"
        im_res = 128
    elif config_name.startswith("epic"):
        data_path = "./data/EPIC_h5/test"
        im_res = 256
    else:
        raise NotImplementedError

    dataset = VideoDataset(
        data_path=data_path,
        input_size=config["data"]["input_size"],
        crop_size=config["data"]["crop_size"],
        frames_per_sample=config["data"]["frames_per_sample"],
        skip_frames=config["data"]["skip_frames"],
        random_horizontal_flip=config["data"]["random_horizontal_flip"],
        aug=False,
        random_time=random_time,
        albumentations=config["data"]["albumentations"],
        frames_key=config["data"].get("frames_key", None))

    model = Model(config["model"])
    model.load_from_ckpt(ckpt_path)
    model.eval()

    print("Loaded the model from", ckpt_path)

    return model, dataset, im_res


CHECKPOINTS = {
    "bair": "./model_weights/bair/model.pth",
    "clevrer": "./model_weights/clevrer/model.pth",
    "epic": "./model_weights/epic/model.pth",
}
