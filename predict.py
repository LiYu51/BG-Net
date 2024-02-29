import argparse
import math
import os
from glob import glob
import cv2

import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import albumentations as albu
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

import torch
from albumentations.augmentations import transforms

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    with open(
            ''E:/project/BG-Net/models/config.yml',
            'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # print(img_ids)

    _, val_img_ids = train_test_split(img_ids, train_size=0.1, random_state=1)

    model.load_state_dict(torch.load(
        'E:/project/BG-Net/models/model.pth'))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    print(config['name'])
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        n = 0
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output, output_list = model(input)

            print(output_list[0].shape)
            print(output_list[1].shape)
            print(output_list[2].shape)
            print(output_list[3].shape)

            for j in range(len(output_list)):
                output_list[j] = output_list[j].cpu().numpy()
                output_list[j] = (output_list[j] - np.min(output_list[j])) / (np.max(output_list[j]) - np.min(output_list[j]))

            iou = iou_score(output.int(), target.int())
            avg_meter.update(1 - iou, input.size(0))

            # for j in output_list:
            #     output_list[j] = output_list[j].
           
            backup = output.cpu().numpy()
            backup = (backup - np.min(backup)) / (np.max(backup) - np.min(backup))

            output = torch.sigmoid(output).cpu().numpy()
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    # print(output[i, c].shape)
                    for l in range(output[i, c].shape[0]):
                        for m in range(output[i, c].shape[1]):
                            output[i, c][l][m] = 1 if output[i, c][l][m] >= 0.5 else 0
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

                   
                    heatmap = np.uint8(255 * backup[i, c])
                    
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    # img = cv2.resize(org_img[i, c], (384, 256))
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '_backup.jpg'),
                                heatmap)

                    for m in range(32):
                        heatmap = np.uint8(255 * output_list[3][i, m])
                       
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        # img = cv2.resize(org_img[i, c], (384, 256))
                        cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '_backup_' + str(m) + '.jpg'),
                                    heatmap)
                    n = n + 1
            #     if n == 4:
            #         break
            # if n == 4:
            #     break

    print('IoU: %.4f' % avg_meter.avg)
    torch.cuda.empty_cache()


import cv2


def add():
   
    heat_img = cv2.imread("./outputs/2022_12211-SZ-CXR1340/0/253_2_backup_22.jpg", 1)
 
    org_img = cv2.imread("./inputs/dsb2018_96/images/253_2.png", 1)
    imagesize = heat_img.shape

    org_img = cv2.resize(org_img, (imagesize[1], imagesize[0]))
    # heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

    cv2.imwrite("./result/heat_u-net_1.jpg", heat_img)
   
    add_img = cv2.addWeighted(org_img, 0.9, heat_img, 0.5, 0)
    cv2.imwrite("./result/add_u-net_1.jpg", add_img)


if __name__ == '__main__':
    main()
    # add()
