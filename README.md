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

<p float="left">
  <img src="media/robot_n5.gif" width="30%" />
  <img src="media/zebra_n5.gif" width="30%" />
  <img src="media/zebra_back_n5.gif" width="30%" />
</p>

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
