import torch
import torch.nn as nn
import torch.nn.functional as F
import ssim
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

ssim_loss = ssim.SSIM(window_size=11, size_average=True)


def binary_cross_entropy_loss(pred, target):
    return F.binary_cross_entropy(pred, target)


def iou_loss1(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter + 1e-6)#+1
    return wiou.mean()


def iou_loss(pred, mask):

    smooth = 1e-6
    avg_mask = F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)
    weit = 1 + 5 * torch.abs(avg_mask - mask) + smooth

    pred = torch.sigmoid(pred)

    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask - pred * mask) * weit).sum(dim=(2, 3))

    union = torch.clamp(union, min=inter + smooth)

    wiou = 1 - (inter + smooth) / union

    return wiou.mean()



def mixed_loss(gc_branch_output, unet_output, f_output, gt):
    bce_loss_gc = binary_cross_entropy_loss(gc_branch_output, gt)

    ssim_loss_unet = ssim_loss(unet_output, gt)
    iou_loss_f =iou_loss(f_output, gt)

    total_loss = bce_loss_gc + ssim_loss_unet + iou_loss_f

    return total_loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        return bce


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
