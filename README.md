# 6.S980 class project for multi view object pose refinement

This project aims to refine 3D object poses by performing gradient descent to minimize the loss between the actual image and the rendered images. We also performed bundle adjustment in the pose refinement by taking  the images rendered from multiple cameras into consideration.

The figure_\*.py files and src/\* were created by us to compare the performence under different combinatons of loss functions and gradient descent hyperparameters.


### Installation
<!-- - [theseus](https://github.com/facebookresearch/theseus#getting-started) -->
- [pypose](https://github.com/pypose/pypose)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- `pip install tqdm imageio imageio-ffmpeg lpips`

