import sys

sys.path.append('../')
import argparse
import os
import shutil
import numpy as np
import warnings
import datetime
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.backends.cuda as cudnn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.meters import AverageMeter
import engine.engine_optim as my_optim
from engine.engine_scheduler import GradualWarmupScheduler
from models import *


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SPA2Net')
        self.parser.add_argument("--arch", type=str, default='vgg_sst')
        self.parser.add_argument("--root_dir", type=str, default=os.getcwd(), help='Root dir for the project')
        self.parser.add_argument("--dataset", type=str, default='cub')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
        self.parser.add_argument("--train_list", type=str, default='')
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--in_norm", type=str, default='True', help='normalize input or not')
        self.parser.add_argument("--tencrop", type=str, default='False')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--seed", default=0, type=int, help='seed for initializing training.')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=64)
        self.parser.add_argument("--drop_last", type=bool, default=False)
        self.parser.add_argument("--pin_memory", type=bool, default=False)

        self.parser.add_argument("--snapshot_dir", type=str)
        self.parser.add_argument("--log_dir", type=str)

        self.parser.add_argument("--resume", type=str, default='False')
        self.parser.add_argument("--restore_from", type=str, default='', help='checkpoint id')

        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--epoch", type=int)

        self.parser.add_argument("--load_finetune", type=str, default='False', help='use fine-tune model or not')
        self.parser.add_argument("--finetuned_model_dir", type=str, default='')
        self.parser.add_argument("--finetuned_model", type=str, default='')

        self.parser.add_argument("--pretrained_model_dir", type=str, default='../pretrained_models')
        self.parser.add_argument("--pretrained_model", type=str, default='')

        self.parser.add_argument("--warmup", type=str, default='False', help='switch use warmup training strategy.')
        self.parser.add_argument("--warmup_fun", type=str, default='gra', help='using on ILSVRC, op: gra / cos')

        self.parser.add_argument("--freeze_module", type=str, default='None',help='freezed modules name.')

        self.parser.add_argument("--decay_module", type=str, default='', help='use @ as delimiter')
        self.parser.add_argument("--decay_node", type=str, default='', help='it can be dynamic, use @ as delimiter')
        self.parser.add_argument("--decay_scale", type=str, default='', help='use @ as delimiter')

        self.parser.add_argument("--lr", type=float)
        self.parser.add_argument("--cls_lr", type=float)
        self.parser.add_argument("--sos_lr", type=float)
        self.parser.add_argument("--sa_lr", type=float)

        self.parser.add_argument("--scg_com", action='store_true')
        self.parser.add_argument("--scg_blocks", type=str, default='4,5')
        self.parser.add_argument("--scg_fosc_th", type=float)
        self.parser.add_argument("--scg_sosc_th", type=float)

        self.parser.add_argument("--sa_use_edge", type=str, help='True / False')
        self.parser.add_argument("--sa_edge_stage", type=str, default='4,5', help='4 for feat4, etc.')
        self.parser.add_argument("--sa_start", type=float, help='the start epoch to introduce sa module.')
        self.parser.add_argument("--sa_head", type=float, default=8, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, help='size of SA linear input')

        self.parser.add_argument("--sos_start", type=float, default=0, help='the start epoch to introduce sos.')
        self.parser.add_argument("--sos_loss_weight", type=float, help='loss weight for the sos loss.')
        self.parser.add_argument("--sos_gt_seg", type=str, default='True', help='True / False')
        self.parser.add_argument("--sos_seg_method", type=str, default='TC', help='BC / TC')
        self.parser.add_argument("--sos_loss_method", type=str, default='BCE', help='BCE / MSE')
        self.parser.add_argument("--sos_fg_th", type=float, help='threshold for segment pseudo gt scm')
        self.parser.add_argument("--sos_bg_th", type=float, help='threshold for segment pseudo gt scm')

        self.parser.add_argument("--ram", action='store_true', help='switch on restricted activation module.')
        self.parser.add_argument("--ram_start", type=float, help='the start epoch to introduce ra loss.')
        self.parser.add_argument("--ram_loss_weight", type=float, help='loss weight for the ra loss.')
        self.parser.add_argument("--ram_th_bg", type=float, help='the variance threshold for back ground.')
        self.parser.add_argument("--ram_bg_fg_gap", type=float, help='gap between fg & bg in ram.')

        self.parser.add_argument("--spa_loss", type=str, default='False')
        self.parser.add_argument("--spa_loss_start", type=int, default=3, help='spa loss start point.')
        self.parser.add_argument("--spa_loss_weight", type=float, default=0.001, help='loss weight for sparse loss.')

        self.parser.add_argument("--mode", type=str, default='sos+sa_v3', help='spa/sos/spa+sa/sos+sa_v3')
        self.parser.add_argument("--watch_cam", action='store_true', help='save cam each iteration.')
        self.parser.add_argument("--wanted_im", type=str, help='the img u want to save during training.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        # eval dataset
        if opt.dataset not in ['cub', 'ilsvrc']:
            raise Exception('Wrong dataset, please check.')
        # eval decay nodes
        if opt.decay_node == '' or opt.decay_module == '' or opt.decay_scale == '':
            raise Exception('Please specify the decayed points.')
        if opt.decay_node == 'dynamic' and opt.dataset != 'cub':
            raise Exception('The decay node can not be dynamic, please check..')
        # eval pretrained model
        opt.pretrained_model = 'vgg16.pth' if 'vgg' in opt.arch else 'inception_v3_google.pth'
        # eval freezed modules
        freezed_m_list = opt.freeze_module.split(',')
        freezed_modules = OrderedDict()
        bakb_module_name = 'conv1_2,conv3,conv4,conv5' if 'vgg' in opt.arch else 'Conv2d_1a_3x3,Conv2d_2a_3x3,' \
                                                                                 'Conv2d_2b_3x3,Conv2d_3b_1x1,' \
                                                                                 'Conv2d_4a_3x3,Mixed_5b,Mixed_5c,' \
                                                                                 'Mixed_5d,Mixed_6a,Mixed_6b,' \
                                                                                 'Mixed_6c,Mixed_6d,Mixed_6e'
        if 'bakb' in freezed_m_list:
            freezed_modules['bakb'] = bakb_module_name
        if 'cls-h' in freezed_m_list:
            freezed_modules['cls-h'] = 'cls'
        if 'loc-h' in freezed_m_list:
            freezed_modules['loc-h'] = 'sos'
        if 'sAtt' in freezed_m_list:
            freezed_modules['sAtt'] = 'sa'
        opt.freeze_module = freezed_modules
        # eval self-attention channels
        if 'sa' in opt.mode:
            opt.sa_neu_num = 512 if 'vgg' in opt.arch else 768
        if opt.watch_cam:
            if opt.dataset == 'cub':
                opt.wanted_im = 'Scarlet_Tanager_0083_138500'
            elif opt.dataset == 'ilsvrc':
                opt.wanted_im = 'n01440764_2574'
            else:
                raise
        # eval training modes
        support_mode = ['spa', 'sos', 'spa+sa', 'sos+sa_v3']
        if opt.mode not in support_mode:
            raise Exception('[Error] Invalid training mode, please check.')
        return opt


def train_print(args, idx, losses, losses_cls, losses_so, losses_ra, losses_spa, top1, top5,
                total_epoch, current_epoch, steps_per_epoch, batch_time, global_counter):
    # Calculate ETA
    eta_seconds = ((total_epoch - current_epoch) * steps_per_epoch +
                   (steps_per_epoch - idx)) * batch_time.avg
    eta_str = "{:0>8}".format(str(datetime.timedelta(seconds=int(eta_seconds))))
    eta_seconds_epoch = steps_per_epoch * batch_time.avg
    eta_str_epoch = "{:0>8}".format(str(datetime.timedelta(seconds=int(eta_seconds_epoch))))
    log_output = 'Epoch: [{0}][{1}/{2}] \t ' \
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t ' \
                 'ETA {eta_str}({eta_str_epoch})\t ' \
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t ' \
                 'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t ' \
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t ' \
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t '.format(current_epoch,
                                                                    global_counter % steps_per_epoch,
                                                                    steps_per_epoch,
                                                                    batch_time=batch_time,
                                                                    eta_str=eta_str,
                                                                    eta_str_epoch=eta_str_epoch,
                                                                    loss=losses,
                                                                    loss_cls=losses_cls,
                                                                    top1=top1, top5=top5)
    if 'sos' in args.mode:
        log_output += 'Loss_so {loss_so.val:.4f} ({loss_so.avg:.4f})\t'.format(loss_so=losses_so)
    if args.ram:
        log_output += 'Loss_ra {loss_ra.val:.4f} ({loss_ra.avg:.4f})\t'.format(loss_ra=losses_ra)
    if args.spa_loss == 'True':
        log_output += 'Loss_spa {loss_spa.val:.4f} ({loss_spa.avg:.4f})\t'.format(loss_spa=losses_spa)
    print(log_output)
    pass


def train_recording(args, writer, losses, losses_cls, losses_ra, losses_so, losses_spa, top1, top5,
                    current_epoch, decay_flag=False, decay_once=False):
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        log_output = '{} \t {:.4f} \t {:.3f} \t {:.3f} \t {:.3f} \t'.format(current_epoch, losses.avg,
                                                                            losses_cls.avg,
                                                                            top1.avg, top5.avg)
        writer.add_scalar('loss_epoch', losses.avg, current_epoch)
        writer.add_scalar('cls_loss_epoch', losses_cls.avg, current_epoch)

        if args.ram:
            log_output += '{:.4f} \t'.format(losses_ra.avg)
            writer.add_scalar('ram_loss', losses_ra.avg, current_epoch)
            if args.dataset == 'cub':
                if args.decay_node == 'dynamic' \
                        and decay_flag is False \
                        and decay_once is False \
                        and losses_ra.avg >= 0.30 \
                        and current_epoch >= 80:
                    decay_flag = True
                elif args.decay_node == 'dynamic' \
                        and decay_flag is False \
                        and decay_once is False \
                        and current_epoch >= 99:
                    decay_flag = True
                pass
            pass
        pass

        if 'sos' in args.mode:
            log_output += '{:.4f} \t'.format(losses_so.avg)
            writer.add_scalar('sos_loss', losses_so.avg, current_epoch)
        pass
        if args.spa_loss == 'True':
            log_output += '{:.4f} \t'.format(losses_spa.avg)
            writer.add_scalar('spa_loss', losses_spa.avg, current_epoch)
        pass
        log_output += '\n'
        fw.write(log_output)
        fw.close()
    return decay_flag


def get_optimized_params(args, model, lrs):
    lr,cls_lr,sos_lr,sa_lr = lrs
    cls_weight_list, cls_bias_list, bb_weight_list, bb_bias_list = [],[],[],[]
    sos_weight_list, sos_bias_list = [], []
    sa_weight_list, sa_bias_list = [], []

    print('\n Following parameters will be assigned different learning rate:')
    for name, value in model.named_parameters():
        if 'cls' in name:
            print("cls-layer's learning rate:", cls_lr, " => ", name)
            if 'weight' in name:
                cls_weight_list.append(value)
            elif 'bias' in name:
                cls_bias_list.append(value)
        elif ('sos' in args.mode) and ('sos' in name):
            print("sos-layer's learning rate:", sos_lr, " => ", name)
            if 'weight' in name:
                sos_weight_list.append(value)
            elif 'bias' in name:
                sos_bias_list.append(value)
        elif ('sa' in args.mode) and ('sa' in name):
            print("sa-module's learning rate:", sa_lr, " => ", name)
            if 'weight' in name:
                sa_weight_list.append(value)
            elif 'bias' in name:
                sa_bias_list.append(value)
        else:
            print("backbone-layer's learning rate:", lr, " => ", name)
            if 'weight' in name:
                bb_weight_list.append(value)
            elif 'bias' in name:
                bb_bias_list.append(value)
    return cls_weight_list, cls_bias_list, bb_weight_list, bb_bias_list, \
           sos_weight_list, sos_bias_list, sa_weight_list, sa_bias_list


def modulesKey2Value(key):
    full_modules_key = ['bakb', 'sAtt', 'cls-h', 'loc-h']
    full_modules_value = ['bb', 'sa', 'cls', 'sos']
    dict_key2value = [full_modules_value[full_modules_key.index(k)] for k in key]
    return dict_key2value


def modulesValue2Key(value):
    full_modules_key = ['bakb', 'sAtt', 'cls-h', 'loc-h']
    full_modules_value = ['bb', 'sa', 'cls', 'sos']
    dict_value2key = [full_modules_key[full_modules_value.index(v)] for v in value]
    return dict_value2key


def set_optimizer(opt_modules, params_list, lrs):
    """Set the parameters of the optimizer, the order of parameters passed to the optimizer cannot be changed.
    Author: Kevin
    :param opt_modules: list of the optimized module names, eg: ['bb','cls','sa','sos']
    :param params_list: list of params values need to be optimized.
    :return: list of the optimized params with learning rate.
    """
    cls_weight_list, cls_bias_list, bb_weight_list, bb_bias_list, sos_weight_list, sos_bias_list, \
    sa_weight_list, sa_bias_list = params_list
    lr,cls_lr,sos_lr,sa_lr = lrs
    optim_params_list = []
    if 'bb' in opt_modules:
        optim_params_list.append({'params': bb_weight_list, 'lr': lr})
        optim_params_list.append({'params': bb_bias_list, 'lr': lr * 2})
    if 'cls' in opt_modules:
        optim_params_list.append({'params': cls_weight_list, 'lr': cls_lr})
        optim_params_list.append({'params': cls_bias_list, 'lr': cls_lr * 2})
    if 'sos' in opt_modules:
        optim_params_list.append({'params': sos_weight_list, 'lr': sos_lr})
        optim_params_list.append({'params': sos_bias_list, 'lr': sos_lr * 2})
    if 'sa' in opt_modules:
        optim_params_list.append({'params': sa_weight_list, 'lr': sa_lr})
        optim_params_list.append({'params': sa_bias_list, 'lr': sa_lr * 2})
    optimizer = optim.SGD(optim_params_list, momentum=0.9, weight_decay=0.0005, nesterov=True)
    return optimizer


def set_lr(args):
    lr = args.lr
    cls_lr = args.cls_lr
    sos_lr = args.sos_lr if ('sos' in args.mode) else 0
    sa_lr = args.sa_lr if ('sa' in args.mode) else 0
    return lr,cls_lr,sos_lr,sa_lr


def get_optimzed_params_name(args, model):
    """Get the list of parameter names that can be optimized.
    Author: Kevin
    :param args: training params
    :param model: training model
    :return: optim_params_name is with suffix '_weight or '_bias',
    but opt_modules_value without suffix.
    """
    model_modules_key = ['bakb', 'cls-h']
    if 'sos' in args.mode:
        model_modules_key.append('loc-h')
    if 'sa' in args.mode:
        model_modules_key.append('sAtt')

    freezed_module_key = list(args.freeze_module.keys()) if len(args.freeze_module) != 0 else []
    freezed_module_value = list(args.freeze_module.values()) if len(args.freeze_module) != 0 else []
    my_optim.freeze_by_names(model, freezed_module_value)
    opt_modules_key = list(set(model_modules_key).difference(freezed_module_key))

    # Resort 'opt_modules_key' as 'model_modules_key'
    opt_modules_key.sort(key=lambda x: model_modules_key.index(x))
    opt_modules_value = modulesKey2Value(opt_modules_key)

    optim_params_name = []
    for v in opt_modules_value:
        optim_params_name.append(v + '_weight')
        optim_params_name.append(v + '_bias')
    return optim_params_name, opt_modules_key, opt_modules_value


def get_model_optimizer(args):
    """Get the model and the optimizer.
    Author: Kevin
    :param args: training params
    :return: model, optimizer, optimized params' name(List)
    """
    if args.load_finetune == 'True':
        model = eval(args.arch).load_finetune(num_classes=args.num_classes, args=args)
    else:
        model = eval(args.arch).model(pretrained=True, num_classes=args.num_classes, args=args)
    model.to(args.device)

    # Get the list of parameter names that can be optimized.
    # 'optim_params_name' is composed of 'opt_modules_value' and a suffix, such as '_weight' and '_bias'.
    # 'opt_modules_key' looks like: ['bakb', 'sAtt', 'cls-h', 'loc-h']
    # 'opt_modules_value' looks like: ['bb', 'sa', 'cls', 'sos']
    optim_params_name, opt_modules_key, opt_modules_value = get_optimzed_params_name(args, model)

    # Set parameters with differenet learning rate for optimizer
    lrs = set_lr(args)
    params_list = get_optimized_params(args, model, lrs)
    optimizer = set_optimizer(opt_modules_value, params_list, lrs)
    model = torch.nn.DataParallel(model, args.gpus)
    return model, optimizer, optim_params_name, opt_modules_key


def save_checkpoint(args, state, filename, is_best=False):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def check_module_validity(module_name):
    pre_def_module_name = ['bakb', 'sAtt', 'cls-h', 'loc-h']
    for m in module_name:
        if isinstance(m, list):
            for mm in m:
                if mm not in pre_def_module_name:
                    raise
        if isinstance(m, str):
            if m not in pre_def_module_name:
                raise
    return


def check_module_scale_node_consistency(modules, scales, nodes=None):
    if nodes is not None:
        if len(nodes) != len(modules) or len(nodes) != len(scales):
            raise
    if len(modules) != len(scales):
        raise
    for i in range(len(modules)):
        if len(modules[i]) != len(scales[i]):
            raise
    pass


def decay_info_builder(args, optim_modules):
    """ Convert decayed module name(Str) to optimizated params index(List).
    Author: Kevin
    params: decay_modules, eg: 'bakb,sAtt,cls-h,loc-h@bakb,sAtt,cls-h,loc-h' => String
    params: decay_scales, eg: '0.1,0.1,0.1,0.1@0.1,0.1,0.1,0.1' => String
    params: decay_nodes, eg: '10,15' => String
    params: optim_modules, eg: ['bakb', 'sAtt', 'cls-h', 'loc-h'] => List
    return: decay_params_idx, eg: {10: [[0,1,0.1],[2,3,0.1]], 15:[[0,1,0.1],[2,3,0.1]]} => Dict
    """
    check_module_validity(optim_modules)
    # Set opt_params_idx according to optim_modules
    opt_params_idx = {}
    for i, opt_m in enumerate(optim_modules):
        opt_params_idx.update({opt_m: [2 * i, 2 * i + 1]})

    decay_modules, decay_scales, decay_nodes = args.decay_module, args.decay_scale, args.decay_node
    decay_m_list = decay_modules.strip().split('@')
    # [['bakb', 'sAtt', 'cls-h', 'loc-h'], ['bakb', 'sAtt', 'cls-h', 'loc-h']]
    decay_m_list = [d_p.split(',') for d_p in decay_m_list]
    check_module_validity(decay_m_list)

    decay_s_list = decay_scales.strip().split('@')
    # [['0.1', '0.1', '0.1', '0.1'], ['0.1', '0.1', '0.1', '0.1']]
    decay_s_list = [d_s.split(',') for d_s in decay_s_list]

    decay_params_idx = {}
    if decay_nodes != 'dynamic':
        decay_n_list = decay_nodes.strip().split('@')
        decay_n_list = [int(x) for x in decay_n_list]
        check_module_scale_node_consistency(decay_m_list, decay_s_list, decay_n_list)
        # Get the module index of decay_modules according to opt_params_idx
        for d_m, d_s, d_n in zip(decay_m_list, decay_s_list, decay_n_list):
            decay_idx_p = []
            for _m, _s in zip(d_m, d_s):
                if _m in opt_params_idx:
                    m_idx = opt_params_idx[_m].copy()  # [idx_0, idx_1]
                    m_idx.append(float(_s))  # [idx_0, idx_1, scale_0]
                    decay_idx_p.append(m_idx)
            decay_params_idx.update({d_n: decay_idx_p})
        print("Set Decayed Modules => ", decay_params_idx)
        return decay_params_idx
    else:
        check_module_scale_node_consistency(decay_m_list, decay_s_list)
        for d_m, d_s in zip(decay_m_list, decay_s_list):
            decay_idx_p = []
            for _m, _s in zip(d_m, d_s):
                if _m in opt_params_idx:
                    m_idx = opt_params_idx[_m].copy()  # [idx_0, idx_1]
                    m_idx.append(float(_s))  # [idx_0, idx_1, scale_0]
                    decay_idx_p.append(m_idx)
            decay_params_idx.update({'dynamic': decay_idx_p})
        print("Set Decayed Modules => ", decay_params_idx)
        return decay_params_idx


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
    cls_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    return_params = [batch_time, losses, cls_loss, top1, top5]
    log_head = '#epoch \t loss \t cls_loss \t pred@1 \t pred@5 \t'

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
            wp_period = [2, 2, 2]
            wp_node = [0, 3, 6]
            wp_ps = [['sos_weight', 'sos_bias'],
                     ['sa_weight', 'sa_bias'], op_params_list]
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
            fw.close()
        return gra_scheduler


def learning_rate_decay(decayed_info, optimizer, epoch):
    """Adjusting learning rate according to decayed_info.
    Author: Kevin
    :param decayed_info: Dict => {epoch0:[[idx_w0,idx_b0,decay_scale0],...], epoch1:[[...],[...]]}
    :param optimizer: optimizer for training
    :param epoch: current epoch
    :return: pass
    """
    decayed_node = decayed_info.keys()
    if epoch in decayed_node:
        decayed_idx_scale = decayed_info[epoch]
        if len(decayed_idx_scale) != 0:
            for item in decayed_idx_scale:
                adjust_learning_rate(optimizer, item)
            pass
        pass
    pass


def learning_rate_decay_dynamic(decayed_info, optimizer):
    """Only used for training on CUB-200-2011."""
    for item in list(decayed_info.values())[0]:
        adjust_learning_rate(optimizer, item)
    pass


def adjust_learning_rate(optimizer, idxs_scales):
    """Adjusting learning rate according to decayed_info.
    Author: Kevin
    :param optimizer: optimizer for training
    :param idxs_scales: List => [idx_weight, idx_bias, decay_scale]
    :return: pass
    """
    _idx_w, _idx_b, _s = idxs_scales
    optimizer.param_groups[_idx_w]['lr'] = optimizer.param_groups[_idx_w]['lr'] * _s
    optimizer.param_groups[_idx_b]['lr'] = optimizer.param_groups[_idx_b]['lr'] * _s
    pass
