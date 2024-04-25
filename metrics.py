import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage






def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu()
    if torch.is_tensor(target):
        target = target.data.cpu()
    output = output > 0.5

    inter = (output * target).sum(dim=(2, 3))
    union = (output + target).sum(dim=(2, 3))

    wiou = 1 - (inter + 1) / (union - inter + 1)

    # return round(1 - float((intersection + smooth) / (union + smooth)), 4)
    return wiou.mean()


def iou_score2(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu()
    if torch.is_tensor(target):
        target = target.data.cpu()

    output = output > 0.5
    target = target > 0.5

    inter = (output & target).sum(dim=(1, 2, 3)).float()
    union = (output | target).sum(dim=(1, 2, 3)).float()

    iou = (inter + smooth) / (union - inter + smooth)

    return iou.mean()


def dice_coef(output, target):
    smooth = 1e-5

    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = np.sum(output * target)

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def dice_coef2(output, target):
    smooth = 1e-5

    output_flat = output.view(-1)
    target_flat = target.view(-1)

    intersection = torch.sum(output_flat * target_flat)
    union = torch.sum(output_flat) + torch.sum(target_flat)

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


def binary_pa(output, target):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    #
    output = output.view(-1).data.cpu().numpy()
    # print(output)
    target = target.view(-1).data.cpu().numpy()
    # output = output.data.cpu().numpy()
    # target = target.data.cpu().numpy()

    intersection = np.float(np.sum(output * target))
    # print("1",intersection)
    intersection0 = np.float(np.sum((1 - output) * (1 - target)))

    pa = (intersection0 + intersection) / target.shape[0]


    return pa

def binary_pa1(output, target):
    """
    Calculate the pixel accuracy of two N-d volumes.
    output: the segmentation volume as a numpy array
    target: the ground truth volume as a numpy array
    """

    output_flat = output.flatten()
    target_flat = target.flatten()


    intersection = np.sum((output_flat == target_flat).astype(int))


    total_pixels = len(target_flat)


    pa = intersection / total_pixels

    return pa

def binary_pa2(output, target):
    """
    Calculate the pixel accuracy of two N-d volumes.
    output: the segmentation volume as a PyTorch tensor
    target: the ground truth volume as a PyTorch tensor
    """

    output_flat = output.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    print(output_flat)
    print(target_flat)

    intersection = np.sum((output_flat == target_flat).astype(int))

    total_pixels = len(target_flat)


    pa = intersection / total_pixels

    return pa

def compute_class_sens_spec11(outputs, targets, batch_size, threshold=0.5):

    outputs = torch.where(outputs >= threshold, torch.ones_like(outputs), torch.zeros_like(outputs))

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(0, outputs.size()[0], batch_size):
        batch_outputs = outputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]

        TP += ((batch_outputs == 1) & (batch_targets == 1)).sum().item()
        TN += ((batch_outputs == 0) & (batch_targets == 0)).sum().item()
        FP += ((batch_outputs == 1) & (batch_targets == 0)).sum().item()
        FN += ((batch_outputs == 0) & (batch_targets == 1)).sum().item()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity



def compute_class_sens_spec(outputs, targets, batch_size, threshold=0.5):
    """
    Compute sensitivity and specificity for a particular example
    for a given class for binary.
    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (height, width, depth).
        label (np.array): binary array of labels, shape is
                          (height, width, depth).
    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """



    sensitivity = 0
    specificity = 0

    for i in range(len(outputs)):


        output = outputs[i]
        target = targets[i]
        output = output > threshold
        target = target == torch.max(target)

        TP = ((output == True) & (target == True))
        FN = ((output == False) & (target == True))
        TN = ((output == False) & (target == False))
        FP = ((output == True) & (target == False))

        sensitivity += float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
        specificity += float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)


    sensitivity = sensitivity / batch_size
    specificity = specificity / batch_size
    return sensitivity, specificity

def compute_class_sens_spec2(outputs, targets, threshold=0.5):
    smooth = 1e-6
    batch_size = len(outputs)

    sensitivity = 0
    specificity = 0

    for i in range(batch_size):

        output = outputs[i] > threshold
        target = targets[i].bool()

        TP = torch.sum(output & target)
        FN = torch.sum(~output & target)
        TN = torch.sum(~output & ~target)
        FP = torch.sum(output & ~target)

        sensitivity += float(TP) / (float(TP + FN) + smooth)
        specificity += float(TN) / (float(TN + FP) + smooth)

    sensitivity /= batch_size
    specificity /= batch_size

    return sensitivity, specificity

