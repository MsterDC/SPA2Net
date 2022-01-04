import sys
sys.path.append('../')
import argparse
import os
import shutil
import numpy as np
import warnings
import random

import torch
import torch.nn.functional as F
import torch.backends.cuda as cudnn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.meters import AverageMeter
import engine.engine_optim as my_optim
from engine.engine_scheduler import GradualWarmupScheduler
from models import *

# default settings
ROOT_DIR = os.getcwd()
LR = 0.001
CLS_LR = 0.01
EPOCH = 21
DISP_INTERVAL = 20


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SPA-Net')
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

        self.parser.add_argument("--tencrop", type=str, default='False')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--mixp", action='store_true', help='turn on amp training.')
        self.parser.add_argument("--seed", default=0, type=int, help='seed for initializing training.')
        self.parser.add_argument("--increase_lr", type=str, default='False', help='only used on ILSVRC for now.')
        self.parser.add_argument("--increase_points", type=str, default='10', help='only used on ILSVRC for now.')

        self.parser.add_argument("--snapshot_dir", type=str, default='')
        self.parser.add_argument("--log_dir", type=str, default='../log')
        self.parser.add_argument("--load_finetune", type=str, default='False', help='use fine-tune model or pretrained')
        self.parser.add_argument("--pretrained_model_dir", type=str, default='../pretrained_models')
        self.parser.add_argument("--pretrained_model", type=str, default='vgg16.pth')
        self.parser.add_argument("--resume", type=str, default='False')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--epoch", type=int, default=100)

        self.parser.add_argument("--decay_points", type=str, default='none', help='using none on CUB as default. On ILSVRC, 12,14 is default setting.')
        self.parser.add_argument("--decay_module", type=str, default='bb,cls,sa;bb,cls,sa', help='using on ILSVRC as default, decay-LR modules')

        self.parser.add_argument("--warmup", type=str, default='False', help='switch use warmup training strategy.')
        self.parser.add_argument("--warmup_fun", type=str, default='gra', help='using on ILSVRC, op: gra / cos')
        self.parser.add_argument("--aba_params", type=str, default='', help='excluded modules to warmup. Using [,] for split')

        self.parser.add_argument("--scg_order", type=int, default=2, help='the order of similarity of HSC.')
        self.parser.add_argument("--scg_com", action='store_true', help='switch on second order supervised.')
        self.parser.add_argument("--scg_blocks", type=str, default='2,3,4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.2)
        self.parser.add_argument("--scg_sosc_th", type=float, default=1)
        self.parser.add_argument("--scg_so_weight", type=float, default=1)

        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--cls_lr", type=float, default=CLS_LR)
        self.parser.add_argument("--sos_lr", type=float, default=0.00005)
        self.parser.add_argument("--sa_lr", type=float, default=0.001)

        self.parser.add_argument("--ram", action='store_true', help='switch on restricted activation module.')
        self.parser.add_argument("--ra_loss_weight", type=float, default=0.1, help='loss weight for the ra loss.')
        self.parser.add_argument("--ram_start", type=float, default=10, help='the start epoch to introduce ra loss.')
        self.parser.add_argument("--ram_th_bg", type=float, default=0.2, help='the variance threshold for back ground.')
        self.parser.add_argument("--ram_bg_fg_gap", type=float, default=0.5, help='gap between fg & bg in ram.')

        self.parser.add_argument("--sos_gt_seg", type=str, default='True', help='True / False')
        self.parser.add_argument("--sos_seg_method", type=str, default='BC', help='BC / TC')
        self.parser.add_argument("--sos_loss_method", type=str, default='BCE', help='BCE / MSE')
        self.parser.add_argument("--sos_fg_th", type=float, default=0.01, help='threshold for segment pseudo gt scm')
        self.parser.add_argument("--sos_bg_th", type=float, default=0.01, help='threshold for segment pseudo gt scm')
        self.parser.add_argument("--sos_loss_weight", type=float, default=0.1, help='loss weight for the sos loss.')
        self.parser.add_argument("--sos_start", type=float, default=10, help='the start epoch to introduce sos.')

        self.parser.add_argument("--sa_use_edge", type=str, default='False', help='Add edge encoding or not')
        self.parser.add_argument("--sa_edge_weight", type=float, default=1)
        self.parser.add_argument("--sa_edge_stage", type=str, default='5', help='4 for feat4, etc.')
        self.parser.add_argument("--sa_start", type=float, default=0, help='the start epoch to introduce sa module.')
        self.parser.add_argument("--sa_head", type=float, default=1, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, default=512, help='size of SA linear input')

        self.parser.add_argument("--spa_loss", type=str, default='False', help='True or False for sparse loss of gt_scm.')
        self.parser.add_argument("--spa_loss_weight", type=float, default=0.001, help='loss weight for sparse loss.')
        self.parser.add_argument("--spa_loss_start", type=int, default=3, help='spa loss start point.')

        self.parser.add_argument("--mode", type=str, default='sos+sa', help='spa/sos/spa+sa/sos+sa_v3')
        self.parser.add_argument("--watch_cam", action='store_true', help='save cam each iteration')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        # eval the training mode
        mode = ['spa','sos','spa+sa','sos+sa_v3']
        if opt.mode not in mode:
            raise Exception('[Error] Invalid training mode, please check.')
        return opt


def get_model(args):

    if args.load_finetune == 'True':
        model = eval(args.arch).load_finetune(num_classes=args.num_classes, args=args)
    else:
        model = eval(args.arch).model(pretrained=True, num_classes=args.num_classes, args=args)
    model.to(args.device)

    lr = args.lr
    cls_lr = args.cls_lr
    sos_lr = args.sos_lr if ('sos' in args.mode) else None
    sa_lr = args.sa_lr if ('sa' in args.mode) else None

    # test for denoising cls loss
    # denoise_layer = ['denoise']
    # denoise_weight_list = []
    # denoise_bias_list = []

    cls_layer = ['cls']
    cls_weight_list = []
    cls_bias_list = []
    other_weight_list = []
    other_bias_list = []

    sos_layer = ['sos'] if ('sos' in args.mode) else []
    sos_weight_list = []
    sos_bias_list = []

    sa_layer = ['sa'] if ('sa' in args.mode) else []
    sa_weight_list = []
    sa_bias_list = []

    print('\n Following parameters will be assigned different learning rate:')
    for name, value in model.named_parameters():
        if cls_layer[0] in name:
            print("cls-layer's learning rate:", cls_lr, " => ", name)
            if 'weight' in name:
                cls_weight_list.append(value)
            elif 'bias' in name:
                cls_bias_list.append(value)
        # test for denoising cls loss
        # elif denoise_layer[0] in name:
        #     print("denoise-layer's learning rate:", cls_lr, " => ", name)
        #     if 'weight' in name:
        #         denoise_weight_list.append(value)
        #     elif 'bias' in name:
        #         denoise_bias_list.append(value)
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
        else:
            print("other layer's learning rate:", lr, " => ", name)
            if 'weight' in name:
                other_weight_list.append(value)
            elif 'bias' in name:
                other_bias_list.append(value)
    # set params list
    # test for denoising cls loss
    op_params_list = [{'params': other_weight_list, 'lr': lr}, {'params': other_bias_list, 'lr': lr * 2},
                      # {'params': denoise_weight_list, 'lr': cls_lr}, {'params': denoise_bias_list, 'lr': cls_lr * 2},
                      {'params': cls_weight_list, 'lr': cls_lr}, {'params': cls_bias_list, 'lr': cls_lr * 2}]
    optim_params_list = ['other_weight', 'other_bias', 'cls_weight', 'cls_bias']
    # optim_params_list = ['other_weight', 'other_bias', 'denoise_weight', 'denoise_bias', 'cls_weight', 'cls_bias']

    if 'sos' in args.mode:
        op_params_list.append({'params': sos_weight_list, 'lr': sos_lr})
        op_params_list.append({'params': sos_bias_list, 'lr': sos_lr * 2})
        optim_params_list.append('sos_weight'), optim_params_list.append('sos_bias')
    if 'sa' in args.mode:
        op_params_list.append({'params': sa_weight_list, 'lr': sa_lr})
        op_params_list.append({'params': sa_bias_list, 'lr': sa_lr * 2})
        optim_params_list.append('sa_weight'), optim_params_list.append('sa_bias')

    optimizer = optim.SGD(op_params_list, momentum=0.9, weight_decay=0.0005, nesterov=True)
    model = torch.nn.DataParallel(model, args.gpus)
    return model, optimizer, optim_params_list


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def set_decay_modules(decay_module):
    # set parameter_lr for decay and increasing
    # 0,1 / 2,3 / 4,5 / 6,7 => bb / cls / sos / sa, 'all' for all
    # only used for ILSVRC now.
    decay_field = decay_module.split(';')
    if decay_field == '':
        raise Exception("[Error] Must specify the decayed modules at first.")
    decay_idx = []
    for field in decay_field:
        if 'bb' in field:
            decay_idx += '0,1'
        if 'cls' in field:
            decay_idx += ',2,3'
        if 'sos' in field:
            decay_idx += ',4,5'
        if 'sa' in field:
            decay_idx += ',6,7'
        if field == 'all':
            decay_idx += 'all'
        decay_idx += ';'
    decay_params = ''.join(decay_idx).strip(';').split(';')
    decay_params.append(None)
    return decay_params


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
    # test for denoising loss
    # denoise_loss = AverageMeter()
    cls_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    return_params = [batch_time, losses, cls_loss, top1, top5]
    # test for denoising loss
    # return_params = [batch_time, losses, denoise_loss, cls_loss, top1, top5]
    log_head = '#epoch \t loss \t cls_loss \t pred@1 \t pred@5 \t'
    # log_head = '#epoch \t loss \t denoise_loss \t cls_loss \t pred@1 \t pred@5 \t'

    losses_so = None
    losses_ra = None
    losses_spa = None

    if 'sos' in args.mode:
        losses_so = AverageMeter()
        log_head += 'loss_so \t'
    if args.ram:
        losses_ra = AverageMeter()
        log_head += 'loss_ra \t'
    if args.spa_loss == 'True':
        losses_spa = AverageMeter()
        log_head += 'loss_spa \t'

    return_params.append(losses_so)
    return_params.append(losses_ra)
    return_params.append(losses_spa)

    log_head += '\n'
    return_params.append(log_head)

    return return_params


def warmup_init(args, optimizer, op_params_list):
    if args.warmup_fun == 'cos':
        cos_scheduler = None
        if args.dataset == 'ilsvrc':
            cos_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1)
        if args.dataset == 'cub':
            cos_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        return cos_scheduler

    # set gra warm-up module
    if args.warmup_fun == 'gra':

        wp_period, wp_node, wp_ps = None, None, None

        if args.dataset == 'ilsvrc':
            aba_params = []
            aba_list = args.aba_params.strip().split(',')
            if 'bb' in aba_list:
                aba_params.append('other_weight')
                aba_params.append('other_bias')
            if 'cls' in aba_list:
                aba_params.append('cls_weight')
                aba_params.append('cls_bias')
            if 'sa' in aba_list:
                aba_params.append('sa_weight')
                aba_params.append('sa_bias')

            fine_tune_params = list(set(op_params_list) - (set(aba_params)))
            wp_period = [2,2,2]
            wp_node = [0,3,6]
            wp_ps = [['sos_weight', 'sos_bias'],
                     ['sa_weight', 'sa_bias'], fine_tune_params]

        elif args.dataset == 'cub':
            raise Exception("For cub-200, gra LR warmup is unused.")

        gra_scheduler = GradualWarmupScheduler(optimizer=optimizer, warmup_period=wp_period,
                                                  warmup_node=wp_node, warmup_params=wp_ps,
                                                  optim_params_list=op_params_list)
        optimizer.zero_grad()
        optimizer.step()
        warmup_message = 'warmup_period:' + ','.join(list(map(str, wp_period))) + '\n'
        warmup_message += 'warmup_timeNode:' + ','.join(list(map(str, wp_node))) + '\n'
        for p_l in wp_ps:
            str_p = ','.join(p_l)
            warmup_message += 'warmup_params:' + str_p + '\n'
        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            fw.write(warmup_message)

        return gra_scheduler


def warmup_adjust(args, decay_params, decay_count, decay_flag, decay_once,
                  increase_once, increase_params, increase_count, optimizer,
                  current_epoch, gra_scheduler):
    return_list = []
    if args.warmup == 'True':  # warmup LR
        if args.dataset == 'cub' and args.warmup_fun == 'gra':
            raise Exception("On cub-200, warmup gra lr is unused.")
        if args.dataset == 'ilsvrc':
            decay_str = decay_params[decay_count]
            if my_optim.reduce_lr(args, optimizer, current_epoch, decay_params=decay_str):
                decay_count += 1
                if args.warmup_fun == 'gra':
                    gra_scheduler.update_optimizer(optimizer, current_epoch)
            if args.warmup_fun == 'gra':
                gra_scheduler.step(current_epoch)
    else:  # w/o warmup
        if args.dataset == 'cub':
            if args.decay_points == 'none':
                if decay_flag is True and decay_once is False:
                    decay_str = decay_params[decay_count]
                    if my_optim.reduce_lr(args, optimizer, current_epoch, decay_params=decay_str):
                        total_epoch = current_epoch + 20
                        decay_once = True
                        decay_count += 1
                        return_list.append(total_epoch)
            else:
                decay_str = decay_params[decay_count]
                if my_optim.reduce_lr(args, optimizer, current_epoch, decay_params=decay_str):
                    decay_once = True
                    decay_count += 1
        if args.dataset == 'ilsvrc':
            decay_str = decay_params[decay_count]
            if my_optim.reduce_lr(args, optimizer, current_epoch, decay_params=decay_str):
                decay_once = True
                decay_count += 1

    # Increasing learning rate only once.
    if args.increase_lr == 'True' and args.dataset == 'ilsvrc':
        if decay_once is True and increase_once is False:
            increase_str = increase_params[increase_count]
            if my_optim.increase_lr(args, optimizer, current_epoch, increase_params=increase_str):
                decay_once = False
                increase_once = True
                increase_count += 1

    return_list.append(optimizer)
    return_list.append(decay_count)
    return_list.append(decay_once)
    return_list.append(gra_scheduler)
    return_list.append(increase_once)
    return_list.append(increase_count)
    return return_list