# DocProj

### [Project page](https://xiaoyu258.github.io/projects/docproj) | [Paper](https://arxiv.org/abs/1909.09470)

The source code of Document Rectification and Illumination Correction using a Patch-based CNN by Li et al, to appear at SIGGRAPH Asia 2019. 

<img src='imgs/teaser.jpg' align="center" width=850> 

## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Dataset Generation
We use [Blender](https://www.blender.org/) to automatically generate synthetic distorted document image and the corresponding flow.

<img src='imgs/syn_dataset.jpg' align="center" width=850> 

You can download a sample dataset (438MB) from [here](https://drive.google.com/open?id=1b3kBs46ZSq5cWjvGdMjWNY854YaJ6dZ1) and we are still working on how to find a good way to upload the full dataset since it is too large (65GB).

The dataset includes three folders: img (the distorted images), img_mask (the mask of background), flow (the forward flow of the distorted images).

The first thing you need to do is to crop the dataset to patches for training. Change the path to your own directory and run the following commands. For help message about optional arguments, run `python xxx.py --h`
```bash
python local_patch.py   # crop images and flows to local patches and local patch flows
python global_patch.py  # crop images to global patches
```
