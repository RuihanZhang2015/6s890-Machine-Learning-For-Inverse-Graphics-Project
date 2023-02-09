# multi_view_pose_refine
6.S980 class project for multi view object pose refinement
## Install

### Prerequisites
<!-- - [theseus](https://github.com/facebookresearch/theseus#getting-started) -->
- [pypose](https://github.com/pypose/pypose)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- `pip install tqdm imageio imageio-ffmpeg lpips`

## Develop

We're using [pre-commit](https://pre-commit.com/) for automatic linting. To install `pre-commit` run:
```
pip install pre-commit
```
You can verify your installation went through by running `pre-commit --version` and you should see something like `pre-commit 2.14.1`.

To get started using `pre-commit` with this codebase, run:
```
cd ${project_root}
pre-commit install
```
Now, each time you `git add` new files and try to `git commit` your code will automatically be run through a variety of linters. You won't be able to commit anything until the linters are happy with your code.
