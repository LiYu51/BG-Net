
# BG-Net
this is a demo
# Boundary-aware Gradient Operator Network for Medical Image Segmentation
## News
2023.5.28: The BG-Net model has been optimised. The paper will be updated later.

## Requirements
- PyTorch 1.x or 0.41

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training on [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) dataset
1. Download dataset from [here]( https://challenge.isic-archive.com/data/#2018) to inputs/ and unzip. The file structure is the following:
```
inputs
└── ISIC2018
    ├── images
                ├──ISIC_00000001.png
                ├──ISIC_00000002.png
                ├──ISIC_0000000n.png
          ├── masks
                ├──ISIC_00000001.png
                ├──ISIC_00000002.png
                ├──ISIC_0000000n.png
    ...
```
```
2. Train the model.
```sh
python train.py --dataset ISIC2018 --arch BG-Net
```
3. Evaluate.
```sh
python text.py --name ISIC2018-model
```

