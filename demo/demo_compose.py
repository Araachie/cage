from typing import List

from model.generation import generate

import torch
from torchvision import transforms as T

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from .common import load_model_and_data, CHECKPOINTS


class Demo:
    STILL_MODE = "s"
    VIS_MODE = "c"
    PASTE_MODE = "v"
    MOVE_MODE = "m"
    GENERATE_SIGNAL = "enter"
    NEXT_SIGNAL = "n"
    SAVE_SIGNAL = "x"

    def __init__(
            self,
            config: str,
            source_image: str = None,
            random_time: bool = False,
            device: str = "cpu",
            steps: int = 20,
            num_frames: int = 3,
            past_horizon: int = -1):
        self.config = config
        self.model, self.dataset, self.im_res = load_model_and_data(
            config,
            CHECKPOINTS[config],
            random_time=random_time)
        if source_image is not None:
            tr = T.Compose([
                T.ToTensor(),
                T.Resize(size=self.im_res, antialias=True),
                T.CenterCrop(size=self.im_res),
            ])
            self.dataset = [[tr(Image.open(source_image).convert("RGB"))]]
        self.device = torch.device(device)
        self.model.to(self.device)
        self.steps = steps
        self.grid_size = self.model.config["vector_field_regressor"]["state_res"][0]
        self.ts = int(self.im_res / self.grid_size)
        self.num_frames = num_frames
        self.past_horizon = past_horizon

        self.reinit()

    def run(self, indices: List[int] = None):
        self.indices = indices

        self.fig = plt.figure(figsize=(9.5, 4))
        self.ax = plt.subplot(1, 1, 1)
        self.set_mode(Demo.STILL_MODE)

        self.show_image()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def reinit(self):
        self.ax = None
        self.fig = None
        self.mode = None
        self.mode_txt = None
        self.indices = None
        self.im_id = 0
        self.features = []
        self.rtg_groups = []
        self.press = False
        self.cur_object = 0
        self.masks = []
        self.controls = []
        self.ctrl_id = 0
        self.moving_rtg_group = None
        self.out_images = []

    def on_key(self, event):
        if event.key == Demo.NEXT_SIGNAL:
            if self.mode == Demo.MOVE_MODE:
                self.set_mode(Demo.PASTE_MODE)
            if self.mode == Demo.PASTE_MODE:
                if self.cur_object < len(self.rtg_groups) - 1:
                    self.cur_object += 1
                    self.fig.suptitle(f'Paste and move object {self.cur_object + 1}', fontsize=16)
                    self.ctrl_id = 0
                else:
                    self.fig.suptitle(f'All objects placed', fontsize=16)
                    self.set_mode(Demo.STILL_MODE)
            else:
                self.clean_rtg_group(self.rtg_groups[-1])
                self.show_image()
                self.set_mode(Demo.STILL_MODE)
        elif event.key in [Demo.VIS_MODE, Demo.MOVE_MODE]:
            self.set_mode(event.key)
        elif event.key == Demo.PASTE_MODE:
            self.clean_rtg_group(self.rtg_groups[-1])
            c = np.ones([self.im_res, self.im_res, 3])
            self.ax.imshow(c)

            self.fig.suptitle(f'Paste and move object {self.cur_object + 1}', fontsize=16)

            self.masks.append(torch.zeros([self.grid_size, self.grid_size], device=self.device))
            self.controls.append(torch.zeros(
                [self.model.feats_dim, self.grid_size, self.grid_size],
                device=self.device))

            self.set_mode(Demo.PASTE_MODE)
        elif event.key == Demo.MOVE_MODE:
            if self.mode == Demo.PASTE_MODE:
                self.set_mode(Demo.MOVE_MODE)
            elif self.mode == Demo.MOVE_MODE:
                self.set_mode(Demo.PASTE_MODE)
        elif event.key == Demo.SAVE_SIGNAL:
            self.fig.tight_layout()
            im = Image.fromarray(np.array(self.fig.canvas.buffer_rgba())).convert("RGB")
            self.out_images.append(im)
        elif event.key == Demo.GENERATE_SIGNAL:
            if self.mode_txt is not None:
                self.mode_txt.remove()
            self.mode_txt = self.ax.annotate("G", (self.im_res - 20, 20), fontsize=16)

            controls = torch.stack(self.controls, dim=0)
            masks = torch.stack(self.masks, dim=0).unsqueeze(1)
            self.generated_frames = generate(
                model=self.model,
                num_frames=self.num_frames,
                device=self.device,
                initial_frames=None,
                feats=controls.unsqueeze(0),
                feats_masks=masks.unsqueeze(0),
                batch_size=1,
                method="joint",
                odesolver="euler",
                num_steps=self.steps,
                past_horizon=self.past_horizon,
                chunk_size=4,
                verbose=False,
            )
            self.mode_txt.remove()
            self.mode_txt = self.ax.annotate("F", (self.im_res - 20, 20), fontsize=16)

    def on_press(self, event):
        if self.ax is None or event.inaxes != self.ax:
            return

        if self.mode == Demo.VIS_MODE:
            self.press = True
            self.rtg_groups.append([])
            rtg, to_be_placed = self.get_rectangle(event.xdata, event.ydata)
            if to_be_placed:
                self.rtg_groups[-1].append(rtg)
                self.ax.add_patch(rtg)
        elif self.mode == Demo.PASTE_MODE:
            if self.cur_object < len(self.rtg_groups):
                rtg_group = self.draw_rtg_group(self.rtg_groups[self.cur_object], pos=(event.xdata, event.ydata))
                self.ctrl_id = 0
                self.add_controls(
                    rtg_group,
                    self.rtg_groups[self.cur_object],
                    self.features[self.cur_object],
                    self.ctrl_id)
        elif self.mode == Demo.MOVE_MODE:
            self.press = True
            self.ctrl_id += 1
            if self.moving_rtg_group is not None:
                self.clean_rtg_group(self.moving_rtg_group)
            self.moving_rtg_group = self.draw_rtg_group(
                self.rtg_groups[self.cur_object], pos=(event.xdata, event.ydata), color="b")

    def on_motion(self, event):
        if self.ax is None or event.inaxes != self.ax or not self.press:
            return

        if self.mode == Demo.VIS_MODE:
            rtg, to_be_placed = self.get_rectangle(event.xdata, event.ydata)
            if to_be_placed:
                self.rtg_groups[-1].append(rtg)
                self.ax.add_patch(rtg)
        elif self.mode == Demo.MOVE_MODE:
            if self.moving_rtg_group is not None:
                self.clean_rtg_group(self.moving_rtg_group)
            self.moving_rtg_group = self.draw_rtg_group(
                self.rtg_groups[self.cur_object], pos=(event.xdata, event.ydata), color="b")

    def on_release(self, event):
        self.press = False

        if self.mode == Demo.MOVE_MODE:
            if len(self.controls) == self.ctrl_id:
                self.controls.append(torch.zeros(
                    [self.model.feats_dim, self.grid_size, self.grid_size],
                    device=self.device))
                self.masks.append(torch.zeros([self.grid_size, self.grid_size], device=self.device))
            self.add_controls(
                self.moving_rtg_group,
                self.rtg_groups[self.cur_object],
                self.features[self.cur_object],
                self.ctrl_id)
            self.moving_rtg_group = None

    def set_mode(self, mode):
        if self.mode_txt is not None:
            self.mode_txt.remove()
        self.mode = mode
        self.mode_txt = self.ax.annotate(self.mode, (self.im_res - 20, 20), fontsize=16)

    def show_image(self):
        if self.indices is None:
            index = np.random.randint(len(self.dataset))
        else:
            index = self.indices[self.im_id]
            self.im_id += 1
            if self.im_id >= len(self.indices):
                self.indices = None

        dimage = self.dataset[index][0]
        image = (np.clip(dimage.permute(1, 2, 0), -1, 1) + 1) / 2

        self.fig.suptitle(f'Interactive demo on {self.config}, idx={index}', fontsize=16)
        self.ax.imshow(image)

        feats = self.model.get_features(
            dimage.unsqueeze(0).unsqueeze(1).to(self.device))
        self.features.append(feats.squeeze(1).squeeze(0))

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
        for ref in self.rtg_groups[-1]:
            if ref._grid_i == i and ref._grid_j == j:
                to_be_placed = False

        return rtg, to_be_placed

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

    def clean_rtg_group(self, rtg_group):
        for rtg in rtg_group:
            rtg.remove()

    def add_controls(self, rtg_group, ref_rtg_group, ref_features, index):
        assert len(rtg_group) == len(ref_rtg_group)
        for rtg, ref in zip(rtg_group, ref_rtg_group):
            self.masks[index][rtg._grid_i, rtg._grid_j] = 1.0
            self.controls[index][:, rtg._grid_i, rtg._grid_j] = ref_features[:, ref._grid_i, ref._grid_j]
