from model.generation import generate

import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from .common import load_model_and_data, CHECKPOINTS


class Demo:
    STILL_MODE = "s"
    VIS_MODE = "c"
    MOVE_MODE = "m"
    GENERATE_SIGNAL = "enter"
    SAVE_SIGNAL = "x"

    def __init__(
            self,
            dataset_name: str,
            random_time: bool = False,
            device: str = "cpu",
            steps: int = 20,
            num_frames: int = 3,
            past_horizon: int = -1):
        self.dataset_name = dataset_name
        self.model, self.dataset, self.im_res = load_model_and_data(
            dataset_name,
            CHECKPOINTS[dataset_name],
            random_time=random_time)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.steps = steps
        self.grid_size = self.model.config["vector_field_regressor"]["state_res"][0]
        self.ts = int(self.im_res / self.grid_size)
        self.num_frames = num_frames
        self.past_horizon = past_horizon

        self.reinit()

    def run(self, index: int = None):
        if index is None:
            index = np.random.randint(len(self.dataset))
        dimage = self.dataset[index][0]
        self.images.append(dimage)

        image = (np.clip(dimage.permute(1, 2, 0), -1, 1) + 1) / 2

        self.fig = plt.figure(figsize=(9.5, 4))
        self.fig.suptitle(f'Interactive video generation on {self.dataset_name}, idx: {index}', fontsize=16)
        self.ax = plt.subplot(1, 1, 1)

        self.ax.imshow(image)

        feats = self.model.get_features(
            dimage.unsqueeze(0).unsqueeze(1).to(self.device))
        self.features.append(feats.squeeze(1).squeeze(0))

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def reinit(self):
        self.images = []
        self.ax = None
        self.fig = None
        self.mode = None
        self.mode_txt = None
        self.press = False
        self.cur_rtg_group = []
        self.masks = [torch.zeros(
            size=[self.grid_size, self.grid_size], device=self.device)]
        self.features = []
        self.frame_id = 1
        self.out_images = []

    def on_press(self, event):
        if self.ax is None or event.inaxes != self.ax:
            return

        if self.mode == Demo.VIS_MODE:
            self.press = True
            self.frame_id = 1
            rtg, to_be_placed = self.get_rectangle(event.xdata, event.ydata)
            if to_be_placed:
                self.cur_rtg_group.append(rtg)
                self.ax.add_patch(rtg)
        elif self.mode == Demo.MOVE_MODE:
            self.press = True
            self.moving_rtg_group = self.draw_rtg_group(self.cur_rtg_group, pos=(event.xdata, event.ydata), color="b")
        else:
            return

    def on_motion(self, event):
        if self.ax is None or event.inaxes != self.ax or not self.press:
            return

        if self.mode == Demo.VIS_MODE:
            rtg, to_be_placed = self.get_rectangle(event.xdata, event.ydata)
            if to_be_placed:
                self.cur_rtg_group.append(rtg)
                self.ax.add_patch(rtg)
        elif self.mode == Demo.MOVE_MODE:
            if self.moving_rtg_group is not None:
                self.clean_rtg_group(self.moving_rtg_group)
            self.moving_rtg_group = self.draw_rtg_group(
                self.cur_rtg_group, pos=(event.xdata, event.ydata), color="b")
        else:
            return

    def on_release(self, event):
        self.press = False

        if self.mode == Demo.VIS_MODE:
            for rtg in self.cur_rtg_group:
                self.masks[0][rtg._grid_i, rtg._grid_j] = 1.0
        elif self.mode == Demo.MOVE_MODE:
            if self.frame_id >= len(self.masks):
                self.masks.append(torch.zeros(
                    size=[self.grid_size, self.grid_size], device=self.device))
                self.features.append(torch.zeros_like(self.features[0], device=self.device))
            for rtg, mrtg in zip(self.cur_rtg_group, self.moving_rtg_group):
                self.masks[self.frame_id][mrtg._grid_i, mrtg._grid_j] = 1.0
                self.features[self.frame_id][:, mrtg._grid_i, mrtg._grid_j] = \
                    self.features[0][:, rtg._grid_i, rtg._grid_j]
            self.frame_id += 1

    def on_key(self, event):
        if event.key in [Demo.VIS_MODE, Demo.MOVE_MODE]:
            self.mode = event.key
            if self.mode_txt is not None:
                self.mode_txt.remove()
            self.mode_txt = self.ax.annotate(event.key, (self.im_res - 20, 20), fontsize=16)

            if self.mode == Demo.VIS_MODE:
                self.cur_rtg_group = []
        elif event.key == Demo.SAVE_SIGNAL:
            self.fig.tight_layout()
            im = Image.fromarray(np.array(self.fig.canvas.buffer_rgba())).convert("RGB")
            self.out_images.append(im)
        elif event.key == Demo.GENERATE_SIGNAL:
            if self.mode_txt is not None:
                self.mode_txt.remove()
            self.mode_txt = self.ax.annotate("G", (self.im_res - 20, 20), fontsize=16)
            self.generated_frames = generate(
                model=self.model,
                num_frames=self.num_frames,
                device=self.device,
                initial_frames=torch.stack(self.images, dim=0).unsqueeze(0),
                feats=torch.stack(self.features[1:], dim=0).unsqueeze(0),
                feats_masks=torch.stack(self.masks[1:], dim=0).unsqueeze(0).unsqueeze(2),
                batch_size=1,
                num_steps=self.steps,
                method="joint",
                odesolver="euler",
                past_horizon=self.past_horizon,
                chunk_size=4,
                verbose=False,
            )
            self.mode_txt.remove()
            self.mode_txt = self.ax.annotate("F", (self.im_res - 20, 20), fontsize=16)

    def draw_rtg_group(self, rtg_group, pos, color="r"):
        if len(rtg_group) == 0:
            return

        mx, my = 0, 0
        for rtg in rtg_group:
            mx += rtg.xy[0]
            my += rtg.xy[1]
        mx /= len(rtg_group)
        my /= len(rtg_group)

        new_rtg_group = []

        for rtg in rtg_group:
            new_rtg, _ = self.get_rectangle(
                rtg.xy[0] + pos[0] - mx,
                rtg.xy[1] + pos[1] - my,
                color=color)
            new_rtg_group.append(new_rtg)
            self.ax.add_patch(new_rtg)

        return new_rtg_group

    def get_rectangle(self, xdata, ydata, color="r"):
        assert self.ax is not None

        x = int(xdata) // self.ts * self.ts
        y = int(ydata) // self.ts * self.ts

        i, j = int(ydata) // self.ts, int(xdata) // self.ts

        rtg = Rectangle(
            xy=(x, y),
            width=self.ts,
            height=self.ts,
            fill=True,
            alpha=0.2,
            facecolor=color)
        rtg._grid_i = i
        rtg._grid_j = j

        # check if is there
        to_be_placed = True
        for ref in self.cur_rtg_group:
            if ref._grid_i == i and ref._grid_j == j:
                to_be_placed = False

        return rtg, to_be_placed

    def clean_rtg_group(self, rtg_group):
        for rtg in rtg_group:
            rtg.remove()
