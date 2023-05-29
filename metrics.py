import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


# import GeodisTK


# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#
#     return (intersection + smooth) / (union + smooth)


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
    # target_ = target > 0.5
    # intersection = np.sum(output * target)

    # intersection = (output & target).sum()
    # print(intersection)
    # union = (output | target).sum()
    # print(union)
    # print
    # print(output)
    # print(target)
    # print(output * target)
    # print(output + target)
    inter = (output * target).sum(dim=(2, 3))
    union = (output + target).sum(dim=(2, 3))
    # print(inter)
    # print(union)
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # return round(1 - float((intersection + smooth) / (union + smooth)), 4)
    return wiou.mean()


def dice_coef(output, target):
    smooth = 1e-5

    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = np.sum(output * target)

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


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
    # pa = (((output_1 & target_1)|(output_2 & target_2)).sum()) / target.size

    # tp = torch.sum((output != 0) * (target != 0))
    # fp = torch.sum((output != 0) * (target == 0))
    # tn = torch.sum((output == 0) * (target == 0))
    # fn = torch.sum((output == 0) * (target != 0))
    #
    # score = (tp + tn).float() / (tp + fp + tn + fn).float()
    #
    # return score.sum() / target.shape[0]

    return pa


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

    # outputs = outputs.data.cpu().numpy()

    sensitivity = 0
    specificity = 0
    # if torch.is_tensor(outputs):
    #     print("**************")
    #     outputs = outputs.data.cpu().numpy()
    for i in range(len(outputs)):
        # print(outputs[i].shape)
        # tp = np.sum((outputs[i] >= 0.5) & (targets[i] >= 0.5))
        # tn = np.sum((outputs[i] < 0.5) & (targets[i] < 0.5))
        # fp = np.sum((outputs[i] >= 0.5) & (targets[i] < 0.5))
        # fn = np.sum((outputs[i] < 0.5) & (targets[i] >= 0.5))

        output = outputs[i]
        target = targets[i]
        output = output > threshold
        target = target == torch.max(target)
        # target = target.data.cpu().numpy()

        # print(output)
        # print(target)
        TP = ((output == True) & (target == True))
        FN = ((output == False) & (target == True))
        TN = ((output == False) & (target == False))
        FP = ((output == True) & (target == False))

        sensitivity += float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
        specificity += float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

        # intersection = np.float(np.sum(output * target))
        # sensitivity = sensitivity + (intersection / np.sum(target))
        #
        # intersection0 = np.float(np.sum((1 - output) * (1 - target)))
        # specificity = specificity + (intersection0 / np.sum(1 - target))
    sensitivity = sensitivity / batch_size
    specificity = specificity / batch_size
    return sensitivity, specificity

# # Hausdorff and ASSD evaluation
# def get_edge_points(img):
#     """
#     get edge points of a binary segmentation result
#     """
#     dim = len(img.shape)
#     if (dim == 2):
#         strt = ndimage.generate_binary_structure(2, 1)
#     else:
#         strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
#     ero = ndimage.morphology.binary_erosion(img, strt)
#     edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
#     return edge
#
# def binary_hausdorff95(s, g, spacing=None):
#     """
#     get the hausdorff distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert (image_dim == len(g.shape))
#     if spacing is None:
#         spacing = [1.0] * image_dim
#     else:
#         assert (image_dim == len(spacing))
#     img = np.zeros_like(s)
#     if (image_dim == 2):
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif (image_dim == 3):
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)
#
#     dist_list1 = s_dis[g_edge > 0]
#     dist_list1 = sorted(dist_list1)
#     dist1 = dist_list1[int(len(dist_list1) * 0.95)]
#     dist_list2 = g_dis[s_edge > 0]
#     dist_list2 = sorted(dist_list2)
#     dist2 = dist_list2[int(len(dist_list2) * 0.95)]
#     return max(dist1, dist2)
