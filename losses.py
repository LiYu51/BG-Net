import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
        
def binary_cross_entropy_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=target.max() - target.min())

def iou_loss(pred, target):
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    iou = intersection / union
    return 1 - iou

def mixed_loss(gc_branch_output, unet_output,f_output,gt):
    bce_loss_gc = binary_cross_entropy_loss(gc_branch_output, gt)
    ssim_loss_unet = ssim_loss(unet_output, gt)
    iou_loss_f = iou_loss(f_output, gt)
    
    total_loss = bce_loss_gc + ssim_loss_unet + iou_loss_f
    
    return total_loss

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
