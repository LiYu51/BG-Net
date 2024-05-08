
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
import ssim
from dataset import Dataset
from metrics import iou_score, dice_coef, binary_pa, compute_class_sens_spec
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
ssim_loss = ssim.SSIM(window_size=11, size_average=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=384, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--bce', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='bce: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--mixloss', default='mixed_loss',
                        choices=LOSS_NAMES,
                        help='mixloss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: mixed_loss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                         help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer

    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer, last_loss):
    avg_meters = {'loss': AverageMeter(),
                  'gc_loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'ssim': AverageMeter(),
                  }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()

        target = target.cuda()

        if config['deep_supervision']:
            outputs = model(input)
            bce = 0
            iou = 0
            ssim = 0
            dice = 0
            pa = 0
            se = 0
            sp = 0
            for output in outputs:

                bce = bce + criterion(output, target)
                output_sig = torch.sigmoid(output)

                output_sig[output_sig > 0.5] = 1
                output_sig[output_sig < 0.5] = 0

                iou = iou + iou_score(output_sig.int(), target.int())
                ssim = ssim + (1 - ssim_loss(output_sig, target))
                dice = dice + dice_coef(output_sig.int(), target.int())
                pa = pa + binary_pa(output_sig.int(), target.int())

                se1, sp1 = compute_class_sens_spec(torch.reshape(output_sig, ((config['batch_size'], 256, 384, 1))),
                                                   torch.reshape(target.int(), ((config['batch_size'], 256, 384, 1))), config['batch_size'])
                se = se + se1
                sp = sp + sp1

            bce = bce / len(outputs)
            iou = iou / len(outputs)
            ssim = ssim / len(outputs)
            dice = dice / len(outputs)
            pa = pa / len(outputs)
            se = se / len(outputs)
            sp = sp / len(outputs)

        else:


            output, unet_output, gc_branch_output = model(input)

        # loss = bce + iou + ssim.item()

        gc_loss = criterion(gc_branch_output, target)

        unet_sig = torch.sigmoid(unet_output)

        n_ssim = 1 - ssim_loss(unet_sig, target)

        f_iou = losses.iou_loss(output, target)
        
        loss = gc_loss + n_ssim + f_iou
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['gc_loss'].update(gc_loss.item(), input.size(0))
        avg_meters['iou'].update(1 - f_iou.item(), input.size(0))
        avg_meters['ssim'].update(n_ssim.item(), input.size(0))


        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('gc_loss', avg_meters['gc_loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('ssim', avg_meters['ssim'].avg),

        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('gc_loss', avg_meters['gc_loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('ssim', avg_meters['ssim'].avg),

                        ])


def validate(config, val_loader, model, criterion, last_loss):
    avg_meters = {'loss': AverageMeter(),
                  'gc_loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'ssim': AverageMeter(),

                  }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)

                bce = 0
                ssim = 0
                iou = 0
                dice = 0
                pa = 0
                se = 0
                sp = 0
                for output in outputs:

                    bce = bce + criterion(output, target)
                    output_sig = torch.sigmoid(output)

                    output_sig[output_sig > 0.5] = 1
                    output_sig[output_sig < 0.5] = 0

                    iou = iou + iou_score(output_sig.int(), target.int())
                    ssim = ssim + (1 - ssim_loss(output_sig, target))
                    dice = dice + dice_coef(output_sig.int(), target.int())
                    pa = pa + binary_pa(output_sig.int(), target.int())

                    se1, sp1 = compute_class_sens_spec(torch.reshape(output_sig, ((config['batch_size'], 256, 384, 1))),
                                                       torch.reshape(target.int(), ((config['batch_size'], 256, 384, 1))), config['batch_size'])
                    se = se + se1
                    sp = sp + sp1

                bce = bce / len(outputs)
                iou = iou / len(outputs)
                ssim = ssim / len(outputs)
                dice = dice / len(outputs)
                pa = pa / len(outputs)
                se = se / len(outputs)
                sp = sp / len(outputs)

            else:

                output, unet_output, gc_branch_output = model(input)

            gc_loss = criterion(gc_branch_output, target)

            unet_sig = torch.sigmoid(unet_output)

            n_ssim = 1 - ssim_loss(unet_sig, target)

            f_iou = losses.iou_loss(output, target)
            

            loss = gc_loss + n_ssim + f_iou
           

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['gc_loss'].update(gc_loss.item(), input.size(0))
            avg_meters['iou'].update(1 - f_iou.item(), input.size(0))
            avg_meters['ssim'].update(n_ssim.item(), input.size(0))


            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('gc_loss', avg_meters['gc_loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('ssim', avg_meters['ssim'].avg),

            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('gc_loss', avg_meters['gc_loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('ssim', avg_meters['ssim'].avg),
                        ])


def main():
    config = vars(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['bce'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['bce']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]


    train_img_ids, val_img_ids = train_test_split(img_ids, train_size=0.8, test_size=0.2, random_state=41)

    train_transform = Compose([
        albu.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('train_loss', []),
        ('train_gc_loss', []),
        ('train_iou', []),
        ('train_ssim', []),

        ('val_loss', []),
        ('val_gc_loss', []),
        ('val_iou', []),
        ('val_ssim', []),

    ])

    best_loss = 100
    trigger = 0
    last_loss = torch.tensor(0, device=device)
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, last_loss)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, last_loss)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])



        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['train_loss'].append(train_log['loss'])
        log['train_gc_loss'].append(train_log['gc_loss'])
        log['train_iou'].append(train_log['iou'])
        log['train_ssim'].append(train_log['ssim'])
        log['val_loss'].append(val_log['loss'])
        log['val_gc_loss'].append(val_log['gc_loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_ssim'].append(val_log['ssim'])

        log_cpu = {}
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log_cpu[key] = value.cpu().detach().numpy()  
            else:
                log_cpu[key] = value


        df_data = {key: value for key, value in log_cpu.items()}
        df = pd.DataFrame.from_dict(df_data, orient='index').T


        df.to_csv('models/%s/log.csv' % config['name'], index=False)


        trigger += 1

        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_loss = val_log['loss']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
