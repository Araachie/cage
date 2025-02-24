<h1 align="center">
  <br>
	CAGE: Unsupervised Visual Composition and Animation for Controllable Video Generation
  <br>
</h1>
  <p align="center">
    <a href="https://araachie.github.io">Aram Davtyan</a> •
    <a href="https://github.com/separius">Sepehr Sameni</a> •
    <a href="https://ommer-lab.com/people/ommer/">Björn Ommer</a> •
    <a href="https://www.cvg.unibe.ch/people/favaro">Paolo Favaro</a>
  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">at AAAI 2025</h4>

<h4 align="center"><a href="https://araachie.github.io/cage/">Website</a> • <a href="https://arxiv.org/abs/2403.14368">Arxiv</a>

#
> **Abstract:** *The field of video generation has expanded significantly in recent years,
> with controllable and compositional video generation garnering considerable interest.
> Traditionally, achieving this has relied on leveraging annotations such as text,
> objects' bounding boxes, and motion cues, which require substantial human effort and
> thus limit its scalability. Thus, we address the challenge of controllable and compositional
> video generation without any annotations by introducing a novel unsupervised approach.
> Once trained from scratch on a dataset of unannotated videos, our model can effectively compose
> scenes by assembling predefined object parts and animating them in a plausible and controlled manner.
> The core innovation of our method lies in its training process, where video generation is conditioned
> on a randomly selected subset of pre-trained self-supervised local features. This conditioning
> compels the model to learn how to inpaint the missing information in the video both spatially and
> temporally, thereby resulting in understanding the inherent compositionality and the dynamics of
> the scene. The abstraction level and the imposed invariance of the conditioning to minor visual
> perturbations enable control over object motion by simply moving the features to the desired future
> locations. We call our model CAGE, which stands for visual Composition and Animation for video
> GEneration. We conduct extensive experiments to validate the effectiveness of CAGE across various
> scenarios, demonstrating its capability to accurately follow the control and to generate high-quality
> videos that exhibit coherent scene composition and realistic animation.*

## Citation

```
@inproceedings{davtyan2025cage,
	title={{CAGE}: Unsupervised Visual Composition and Animation for Controllable Video Generation},
	author={Davtyan, Aram and Sameni, Sepehr and Ommer, Björn and Favaro, Paolo},
	booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
	year={2025},
}
```

## Prerequisites

For convenience, we provide an `environment.yml` file that can be used to install the required packages 
to a `conda` environment with the following command 

```conda env create -f environment.yml```

The code was tested with cuda=12.1 and python=3.9.

## Pretrained models

We share the weights of the models pretrained on the datasets considered in the paper.

<table style="margin:auto">
    <thead>
        <tr>
          <th>Dataset</th>
          <th>Resolution</th>
          <th>Training iterations</th>
          <th>Autoencoder</th>
          <th>Main model</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>BAIR</td>
            <td>256 x 256</td>
            <td>100k</td>
            <td><a href="https://huggingface.co/cvg-unibe/cage_bair_256/blob/main/vqvae.ckpt">download</a></td>
            <td><a href="https://huggingface.co/cvg-unibe/cage_bair_256/blob/main/model.pth">download</a></td>
        </tr>
        <tr>
            <td>CLEVRER</td>
            <td>128 x 128</td>
            <td>100k</td>
            <td><a href="https://huggingface.co/cvg-unibe/cage_clevrer_128/blob/main/vqvae.pth">download</a></td>
            <td><a href="https://huggingface.co/cvg-unibe/cage_clevrer_128/blob/main/model.pth">download</a></td>
        </tr>
        <tr>
            <td>EPIC-KITCHENS P03</td>
            <td>256 x 256</td>
            <td>100k</td>
            <td><a href="https://huggingface.co/cvg-unibe/cage_epic_256/blob/main/vqvae.ckpt">download</a></td>
            <td><a href="https://huggingface.co/cvg-unibe/cage_epic_256/blob/main/model.pth">download</a></td>
        </tr>
    </tbody>
</table>

## Running pretrained models

To use a model that was trained with the code in this repository, 
you may utilize the `generate_frames` method of the model class.
Usage example:

```angular2html
from lutils.configuration import Configuration
from lutils.logging import to_video
from model import Model
from model.geenration import generate
from demo.common import show_video

config = Configuration(<path_to_config_file>)
model = Model(config["model"])
model.load_from_ckpt(<path_to_checkpoint_file>)
model.cuda()
model.eval()

generated_frames = generate(
    model=model,
    num_frames=num_frames,
    device="cuda:0",
    initial_frames=initial_frames,  # [b l C H W]
    feats=control_features,  # [b s d h w]
    feats_masks=control_features_masks,  # [b s 1 h w]
    verbose=True)

--------------
generated_frames = to_video(generated_frames)  # to save the generated video
------OR------
show_video(generated_frames)  # to display the generated video with matplotlib (e.g. in Jupyter notebook)
--------------
```

## Demo notebooks

For the best experience with our pretrained models, check the `demo.ipynb` and `demo_compose.ipynb` files.

Before running the demos, download the pretrained models and store them at the following paths:

```
CHECKPOINTS = {
    "bair": "./model_weights/bair/model.pth",
    "clevrer": "./model_weights/clevrer/model.pth",
    "epic": "./model_weights/epic/model.pth",
}
```

Also, make sure the test h5 files with videos (check the datasets section below) are located at:

```
BAIR: ./data/BAIR_h5/test
CLEVRER: ./data/CLEVRER_h5/test
EPIC-KITCHENS: ./data/EPIC_h5/test
```

Alternatively, you may also change these paths in `demo/common.py`.


## Training your own models

To train your own video prediction models you need to start by preparing the data. 

### Datasets

The training code expects the dataset to be packed into .hdf5 files in a custom manner. 
To create such files, use the provided `dataset/convert_to_h5.py` script. 
Usage example:

```angular2html
python dataset/convert_to_h5.py --out_dir <directory_to_store_the_dataset> --data_dir <path_to_video_frames> --image_size 128 --extension png
```

The output of `python dataset/convert_to_h5.py --help` is as follows:

```angular2html
usage: convert_to_h5.py [-h] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--image_size IMAGE_SIZE] [--extension EXTENSION] [--with_flows]

optional arguments:
  -h, --help            show this help message and exit
  --out_dir OUT_DIR     Directory to save .hdf5 files
  --data_dir DATA_DIR   Directory with videos
  --image_size IMAGE_SIZE
                        Resolution to resize the images to
  --extension EXTENSION
                        Video frames extension
```

The video frames at `--data_dir` should be organized in the following way:

```angular2html
data_dir/
|---train/
|   |---00000/
|   |   |---00000.png
|   |   |---00001.png
|   |   |---00002.png
|   |   |---...
|   |---00001/
|   |   |---00000.png
|   |   |---00001.png
|   |   |---00002.png
|   |   |---...
|   |---...
|---val/
|   |---...
|---test/
|   |---...
```

To extract individual frames from a set of video files, we recommend using the `convert_video_directory.py` script from the [official PVG repository](https://github.com/willi-menapace/PlayableVideoGeneration#custom-datasets).

**BAIR:** Collect the dataset following instruction from the [official PVG repository](https://github.com/willi-menapace/PlayableVideoGeneration#preparing-datasets).

**CLEVRER:** Download the videos from the [official dataset's website](http://clevrer.csail.mit.edu/).

**EPIC-KITCHENS:** Download the videos from the [official dataset's website](https://epic-kitchens.github.io/2025).

### Training autoencoder

We recommend to use the official [taming transformers repository](https://github.com/CompVis/taming-transformers) for 
training VQGAN. To use the trained VQGAN at the second stage, update the `model->autoencoder` field in the config accordingly. 
To do this, set `type` to `ldm-vq`, `config` to `f8_small`, `f8` or `f16` depending on the VQGAN config that was used at training.
We recommend using low-dimensional latents, e.g. from 4 to 8, and down-sampling images at least to 16 x 16 resolution. 

Besides, we also provide our own autoencoder architecture at `model/vqgan/vqvae.py` that one may use to train simpler VQVAEs.
For instance, our pretrained model on the CLEVRER dataset uses this custom implementation.

### Training main model

To launch the training of the main model, use the `train.py` script from this repository.
Usage example:

```angular2html
python train.py --config <path_to_config> --run-name <run_name> --wandb
```

The output of `python train.py --help` is as follows:

```angular2html
usage: train.py [-h] --run-name RUN_NAME --config CONFIG [--num-gpus NUM_GPUS] [--resume-step RESUME_STEP] [--random-seed RANDOM_SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME   Name of the current run.
  --config CONFIG       Path to the config file.
  --num-gpus NUM_GPUS   Number of gpus to use for training. By default uses all available gpus.
  --resume-step RESUME_STEP
                        Step to resume the training from.
  --random-seed RANDOM_SEED
                        Random seed.
  --wandb               If defined, use wandb for logging.
```

Use the configs provided in this repository as examples. 