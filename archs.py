import torch
from torch import nn

__all__ = ['UNet', 'NestedUNet']

from torchvision.transforms import Resize


from gradconv import gradconvnet


import torch.nn.functional as F

from Fusion import Fusion_block


class FReLU(nn.Module):
    """
    FReLU https://arxiv.org/abs/2007.11824
    """

    def __init__(self, c1, k=5):  # ch_in, kernel
        super().__init__()

        self.conv = nn.Conv2d(c1, c1, k, 1, 2, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):

        return torch.max(x, self.bn(self.conv(x)))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.relu = FReLU(middle_channels)

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


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = x0_4
        return output


def crop(inputs):
    img_1 = []
    img_2 = []
    img_3 = []
    img_4 = []

    for input in inputs:
        img_tensor_up, img_tensor_down = input.split(128, dim=1)
        img_tensor_1, img_tensor_2 = img_tensor_up.split(192, dim=2)
        img_tensor_3, img_tensor_4 = img_tensor_down.split(192, dim=2)
        img_1.append(img_tensor_1)
        img_2.append(img_tensor_2)
        img_3.append(img_tensor_3)
        img_4.append(img_tensor_4)

    img1 = torch.stack(img_1, 0)
    img2 = torch.stack(img_2, 0)
    img3 = torch.stack(img_3, 0)
    img4 = torch.stack(img_4, 0)

    img = torch.stack((img1, img2, img3, img4), 0)
    return img


def concat(input):

    img_1 = torch.cat((input[0], input[1]), 3)

    img_2 = torch.cat((input[2], input[3]), 3)

    img_res = torch.cat((img_1, img_2), 2)

    return img_res


class NestedUNet(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.edge_net = gradconvnet('gradconv')

        self.resize = Resize((128, 192), 2)
        self.deep_supervision = deep_supervision
        self.grad_conv = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.fusion1 = Fusion_block(32, 32, 2, 32, 32, 0.1)
        self.fusion2 = Fusion_block(32, 32, 2, 32, 32, 0.1)
        self.fusion3 = Fusion_block(32, 32, 2, 32, 32, 0.1)
        self.fusion4 = Fusion_block(32, 32, 2, 32, 32, 0.1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final_unet = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final_fusion = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        imgs = crop(input)

        img1 = self.edge_net(imgs[0])
        img2 = self.edge_net(imgs[1])
        img3 = self.edge_net(imgs[2])
        img4 = self.edge_net(imgs[3])
        img_tensor = torch.stack((img1, img2, img3, img4), 0)
        img_feature = concat(img_tensor)

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x_feture1 = self.fusion1(img_feature, x0_1)
        x_feture2 = self.fusion2(img_feature, x0_2)
        x_feture3 = self.fusion3(img_feature, x0_3)
        x_feture4 = self.fusion4(img_feature, x0_4)


        if self.deep_supervision:
            output1 = self.final1(x_feture1)

            output2 = self.final2(x_feture2)

            output3 = self.final3(x_feture3)

            output4 = self.final4(x_feture4)

            return [output1, output2, output3, output4]

        else:
            output = self.final_fusion(x_feture4)
            unetput = self.final_unet(x0_4)

            gradconv_out = self.grad_conv(img_feature)
            return output, unetput, gradconv_out
