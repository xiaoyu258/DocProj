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

You can download a small dataset with 20 samples (438MB) from [here](https://drive.google.com/open?id=1b3kBs46ZSq5cWjvGdMjWNY854YaJ6dZ1) for fun and the full dataset with 2450 samples (65GB) from [here](https://drive.google.com/open?id=1WkzMukIHS_smGPyjcyj7LIiWUk0RJriN).

The dataset includes three folders: 
- img (the distorted images, with the shape of [2400, 1800, 3])
- img_mask (the mask of background, with the shape of [2400, 1800])
- flow (the forward flow of the distorted images, with the shape of [2, 2400, 1800])

The first thing you need to do is to crop the dataset to patches for training. Change arguments to your own and run the following commands. For help message about optional arguments, run `python xxx.py --h`
```bash
python local_patch.py   # crop images and flows to local patches and local patch flows
python global_patch.py  # crop images to global patches
```

### Training
Run the following command for training and change the optional arguments like dataset directory, etc.
```bash
python train.py
```

### Use a Pre-trained Model
You can download the pre-trained model [here](https://drive.google.com/open?id=1EPmFYd7OwfUZBLkJQ9sO8G1r5tLniKDh).

Run the following command for resizing and cropping the test document image to local and global patches and estimate the patch flows:
```bash
python eval.py [--imgPath [PATH]] [--modelPath [PATH]]
               [--saveImgPath [PATH]] [--saveFlowPath [PATH]]
               
--imgPath             Path to input image
--modelPath           Path to pre-trained model
--saveImgPath         Path to saved cropped image
--saveFlowPath        Path to saved estimated flows
```
