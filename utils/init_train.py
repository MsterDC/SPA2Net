import sys
sys.path.append('../')
import argparse
import os
import shutil
import numpy as np
import warnings
import random
import torch
from torch import optim
import torch.nn.functional as F
import torch.backends.cuda as cudnn
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD

from utils.scheduler import GradualWarmupScheduler
from utils.restore import restore
from utils import AverageMeter, MoveAverageMeter
from models import *


# default settings
ROOT_DIR = os.getcwd()
LR = 0.001
EPOCH = 21
DISP_INTERVAL = 20


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='TPAMI2022-SST')
        self.parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help='Root dir for the project')
        self.parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
        self.parser.add_argument("--vis_name", type=str, default='')
        self.parser.add_argument("--train_list", type=str, default='')
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='cub')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_DA')
        self.parser.add_argument("--in_norm", type=str, default='True', help='normalize input or not')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
        self.parser.add_argument("--resume", type=str, default='False')
        self.parser.add_argument("--tencrop", type=str, default='False')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--mixp", action='store_true', help='turn on amp training.')
        self.parser.add_argument("--pretrained_model_dir", type=str, default='../pretrained_models')
        self.parser.add_argument("--pretrained_model", type=str, default='vgg16.pth')
        self.parser.add_argument("--seed", default=None, type=int, help='seed for initializing training. ')
        self.parser.add_argument("--scg_blocks", type=str, default='2,3,4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.2)
        self.parser.add_argument("--scg_sosc_th", type=float, default=1)
        self.parser.add_argument("--scg_order", type=int, default=2, help='the order of similarity of HSC.')
        self.parser.add_argument("--scg_so_weight", type=float, default=1)
        self.parser.add_argument("--scg_com", action='store_true', help='switch on second order supervised.')
        self.parser.add_argument("--ram", action='store_true', help='switch on restricted activation module.')
        self.parser.add_argument("--ra_loss_weight", type=float, default=0.1, help='loss weight for the ra loss.')
        self.parser.add_argument("--ram_start", type=float, default=10, help='the start epoch to introduce ra loss.')
        self.parser.add_argument("--ram_th_bg", type=float, default=0.2, help='the variance threshold for back ground.')
        self.parser.add_argument("--ram_bg_fg_gap", type=float, default=0.5, help='gap between fg & bg in ram.')
        self.parser.add_argument("--use_tap", type=str, default='False')
        self.parser.add_argument("--tap_th", type=float, default=0.1, help='threshold avg pooling')
        self.parser.add_argument("--tap_start", type=float, default=0)
        self.parser.add_argument("--watch_cam", action='store_true', help='save cam each iteration')

        self.parser.add_argument("--snapshot_dir", type=str, default='')
        self.parser.add_argument("--log_dir", type=str, default='../log')
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--decay_points", type=str, default='80')
        self.parser.add_argument("--epoch", type=int, default=100)
        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--sos_lr", type=float, default=0.001)
        self.parser.add_argument("--sos_gt_seg", type=str, default='True', help='True / False')
        self.parser.add_argument("--sos_seg_method", type=str, default='BC', help='BC / TC')
        self.parser.add_argument("--sos_loss_method", type=str, default='BCE', help='BCE / MSE')
        self.parser.add_argument("--sos_fg_th", type=float, default=0.01, help='segment pseudo gt scm')
        self.parser.add_argument("--sos_bg_th", type=float, default=0.01, help='segment pseudo gt scm')
        self.parser.add_argument("--sos_loss_weight", type=float, default=0.1, help='loss weight for the sos loss.')
        self.parser.add_argument("--sos_start", type=float, default=10, help='the start epoch to introduce sos.')
        self.parser.add_argument("--sa_lr", type=float, default=0.001)
        self.parser.add_argument("--sa_use_edge", type=str, default='False', help='Add edge encoding or not')
        self.parser.add_argument("--sa_edge_stage", type=str, default='5', help='4 for feat4, etc.')
        self.parser.add_argument("--sa_start", type=float, default=0, help='the start epoch to introduce sa module.')
        self.parser.add_argument("--sa_head", type=float, default=1, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, default=512, help='size of SA linear input')

        self.parser.add_argument("--cls_or_hinge", type=str, default='cls')
        self.parser.add_argument("--hinge_norm", type=str, default='softmax', help='norm/softmax')
        self.parser.add_argument("--hinge_lr", type=float, default=0.001)
        self.parser.add_argument("--hinge_loss_weight", type=float, default=0.1)
        self.parser.add_argument("--hinge_p", type=float, default=1)
        self.parser.add_argument("--hinge_m", type=float, default=0.9)

        self.parser.add_argument("--mode", type=str, default='sos+sa', help='spa/spa+hinge/sos/spa+sa/sos+sa/mc_sos')

        # self.parser.add_argument("--rcst_lr", type=float, default=0.005)
        # self.parser.add_argument("--rcst_signal", type=str, default='scm', help='sos / scm')
        # self.parser.add_argument("--rcst_loss_weight", type=float, default=0.1, help='loss weight for the ra loss.')
        # self.parser.add_argument("--rcst_start", type=float, default=10, help='the start epoch to introduce sos.')
        # self.parser.add_argument("--rcst_ratio", type=float, default=700, help='the ratio to scale of sos map.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        return opt


def get_scheduler(optim):
    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()


def get_model(args):
    model = eval(args.arch).model(pretrained=True, num_classes=args.num_classes, args=args)
    model.to(args.device)

    lr = args.lr
    sos_lr = args.sos_lr if ('sos' in args.mode) else None
    sa_lr = args.sa_lr if 'sa' in args.mode else None
    hg_lr = args.hinge_lr if 'hinge' in args.mode else None

    cls_layer = ['cls']
    cls_weight_list = []
    cls_bias_list = []

    sos_layer = ['sos'] if ('sos' in args.mode) else []
    sos_weight_list = []
    sos_bias_list = []

    sa_layer = ['sa'] if 'sa' in args.mode else []
    sa_weight_list = []
    sa_bias_list = []

    hg_layer = ['hinge'] if 'hinge' in args.mode else []
    hg_weight_list = []
    hg_bias_list = []

    other_weight_list = []
    other_bias_list = []

    # if 'sos' in args.mode:
    #     sos_lr = args.sos_lr
    # if 'sa' in args.mode:
    #     sa_lr = args.sa_lr
    # if 'rcst' in args.mode or 'sst' in args.mode:
    #     rcst_lr = args.rcst_lr
    #     fpn_layers = ['fpn', 'maxpool', 'rcst']
    #     fpn_weight_list = []
    #     fpn_bias_list = []

    print('\n Following parameters will be assigned different learning rate:')
    for name, value in model.named_parameters():
        if cls_layer[0] in name:
            print("cls-layer's learning rate:", lr*10, " => ", name)
            if 'weight' in name:
                cls_weight_list.append(value)
            elif 'bias' in name:
                cls_bias_list.append(value)
        elif ('sos' in args.mode) and sos_layer[0] in name:
            print("sos-layer's learning rate:", sos_lr, " => ", name)
            if 'weight' in name:
                sos_weight_list.append(value)
            elif 'bias' in name:
                sos_bias_list.append(value)
        elif ('sa' in args.mode) and sa_layer[0] in name:
            print("sa-module's learning rate:", sa_lr, " => ", name)
            if 'weight' in name:
                sa_weight_list.append(value)
            elif 'bias' in name:
                sa_bias_list.append(value)
        elif ('hinge' in args.mode) and hg_layer[0] in name:
            print("hinge-layer's learning rate:", hg_lr*10, " => ", name)
            if 'weight' in name:
                hg_weight_list.append(value)
            elif 'bias' in name:
                hg_bias_list.append(value)
        else:
            print("other layer's learning rate:", lr, " => ", name)
            if 'weight' in name:
                other_weight_list.append(value)
            elif 'bias' in name:
                other_bias_list.append(value)
        # if ('rcst' in args.mode or 'sst' in args.mode) and any([x in name for x in fpn_layers]):
        #     print("rcst learning rate:", name)
        #     if 'weight' in name:
        #         fpn_weight_list.append(value)
        #     elif 'bias' in name:
        #         fpn_bias_list.append(value)
        #     continue
    op_params_list = [{'params': other_weight_list, 'lr': lr}, {'params': other_bias_list, 'lr': lr * 2},
                      {'params': cls_weight_list, 'lr': lr * 10}, {'params': cls_bias_list, 'lr': lr * 20}]

    if 'sos' in args.mode:
        op_params_list.append({'params': sos_weight_list, 'lr': sos_lr})
        op_params_list.append({'params': sos_bias_list, 'lr': sos_lr * 2})
    if 'sa' in args.mode:
        op_params_list.append({'params': sa_weight_list, 'lr': sa_lr})
        op_params_list.append({'params': sa_bias_list, 'lr': sa_lr * 2})
    if 'hinge' in args.mode:
        op_params_list.append({'params': hg_weight_list, 'lr': hg_lr * 10})
        op_params_list.append({'params': hg_bias_list, 'lr': hg_lr * 20})
    # if 'rcst' in args.mode or 'sst' in args.mode:
    #     op_params_list.append({'params': fpn_weight_list, 'lr': rcst_lr})
    #     op_params_list.append({'params': fpn_bias_list, 'lr': rcst_lr * 2})
    optimizer = optim.SGD(op_params_list, momentum=0.9, weight_decay=0.0005, nesterov=True)
    model = torch.nn.DataParallel(model, args.gpus)
    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)

    return model, optimizer


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def reproducibility_set(args):
    # for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True


def log_init(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    return_params = [batch_time, losses, top1, top5]

    log_head = '#epoch \t loss \t pred@1 \t pred@5 \t'

    losses_so = None
    losses_hg = None
    losses_ra = None

    if 'sos' in args.mode:
        losses_so = AverageMeter()
        log_head += 'loss_so \t'
    if 'hinge' in args.mode:
        losses_hg = AverageMeter()
        log_head += 'loss_hg \t'
    # if 'rcst' in args.mode or 'sst' in args.mode:
    #     losses_rcst = AverageMeter()
    #     return_params.append(losses_rcst)
    #     log_head += 'loss_rcst \t '
    if args.ram:
        losses_ra = AverageMeter()
        log_head += 'loss_ra \t'

    return_params.append(losses_so)
    return_params.append(losses_hg)
    return_params.append(losses_ra)

    log_head += '\n'
    return_params.append(log_head)

    return return_params
