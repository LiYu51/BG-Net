import torch
from torch import nn

__all__ = ['UNet', 'NestedUNet']
from torchvision.transforms import Resize

from gradconv import gradconvnet
from BiFusion import BiFusion_block

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        # self.relu = FReLU(middle_channels)
        # 卷积添加了 dilation=(2,)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


