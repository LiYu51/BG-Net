import argparse
import math
import os
from glob import glob

import cv2
import torch.backends.cudnn as cudnn
import yaml
import albumentations as albu
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score, dice_coef, compute_class_sens_spec
from utils import AverageMeter

import torch
from albumentations.augmentations import transforms

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='dsb2018_96_NestedUNet_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    with open(
            'E:/project/project/pytorch_nested_unet_master/pytorch_nested_unet_master/models/dsb2018_96_NestedUNet_woDS/config.yml',
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
        'E:/project/project/pytorch_nested_unet_master/pytorch_nested_unet_master/models/dsb2018_96_NestedUNet_woDS/model.pth'))
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

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                # output = model(input)
                output,unetoutput,gradconv_out= model(input)

            output_sig = torch.sigmoid(output)

            output_sig[output_sig > 0.5] = 1
            output_sig[output_sig < 0.5] = 0

            iou = iou_score(output.int(), target.int())
            avg_meter.update(1 - iou, input.size(0))

            dice = dice_coef(output_sig.int(), target.int())
            se1, sp1 = compute_class_sens_spec(torch.reshape(output_sig, ((config['batch_size'], 256, 384, 1))),torch.reshape(target.int(), ((config['batch_size'], 256, 384, 1))),config['batch_size'])

            backup = output.cpu().numpy()
            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    # print(output[i, c].shape)
                    for l in range(output[i, c].shape[0]):
                        for m in range(output[i, c].shape[1]):
                            output[i, c][l][m] = 1 if output[i, c][l][m] >= 0.5 else 0
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))
                    # cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '_backup.jpg'),
                    #             (backup[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    print('dice: %.4f' % dice)
    print('se: %.4f' % se1)
    print('sp: %.4f' % sp1)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
