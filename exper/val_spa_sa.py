# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
import argparse
import os
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F

from utils import AverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore
from utils.localization import get_topk_boxes_hier, get_topk_boxes_hier_scg, get_topk_boxes_scg_v2
from utils.vistools import save_im_heatmap_box, save_im_sim, save_sim_heatmap_box, vis_feature, vis_var
from models import *
import cv2

LR = 0.001
EPOCH = 200
DISP_INTERVAL = 50

# default settings
ROOT_DIR = os.getcwd()


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CVPR2021-SPA')
        self.parser.add_argument("--root_dir", type=str, default='')
        self.parser.add_argument("--img_dir", type=str, default='')
        self.parser.add_argument("--test_list", type=str, default='')
        self.parser.add_argument("--test_box", type=str, default='')
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='imagenet')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_v0')
        self.parser.add_argument("--threshold", type=str, default='0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45')
        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--decay_points", type=str, default='none')
        self.parser.add_argument("--epoch", type=int, default=EPOCH)
        self.parser.add_argument("--tencrop", type=str, default='True')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
        self.parser.add_argument("--snapshot_dir", type=str, default='')
        self.parser.add_argument("--resume", type=str, default='True')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--debug", action='store_true', help='.')
        self.parser.add_argument("--debug_detail", action='store_true', help='.')
        self.parser.add_argument("--vis_feat", action='store_true', help='.')
        self.parser.add_argument("--vis_var", action='store_true', help='.')
        self.parser.add_argument("--debug_dir", type=str, default='../debug', help='save visualization results.')
        self.parser.add_argument("--vis_dir", type=str, default='../vis_dir', help='save visualization results.')
        self.parser.add_argument("--scg", action='store_true', help='switch on the self-correlation generating module.')
        self.parser.add_argument("--scg_blocks", type=str, default='2,3,4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_com", action='store_true')
        self.parser.add_argument("--scg_fo", action='store_true')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.1)
        self.parser.add_argument("--scg_sosc_th", type=float, default=0.1)
        self.parser.add_argument("--scg_order", type=int, default=2)
        self.parser.add_argument("--scg_so_weight", type=float, default=1)
        self.parser.add_argument("--scg_fg_th", type=float, default=0.01)
        self.parser.add_argument("--scg_bg_th", type=float, default=0.01)
        self.parser.add_argument("--iou_th", type=float, default=0.5)
        self.parser.add_argument("--in_norm", type=str, default='True', help='normalize input or not')
        self.parser.add_argument("--sa", action='store_true', help='using sa module or not')
        self.parser.add_argument("--sa_edge_encode", type=str, default='False', help='Add edge encoding or not')
        self.parser.add_argument("--use_sa", type=str, default='False', help='using sa module or not')
        self.parser.add_argument("--sa_head", type=float, default=1, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, default=512, help='size of SA linear input')
        self.parser.add_argument("--use_thr_pool", type=str, default='False')
        self.parser.add_argument("--th_avgpool", type=float, default=0.1, help='threshold of avg pooling')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.threshold = list(map(float, opt.threshold.split(',')))
        return opt


def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args)

    model = torch.nn.DataParallel(model, args.gpus)
    model.cuda()

    if args.resume == 'True':
        restore(args, model, None, istrain=False)

    return model


def eval_loc(cls_logits, cls_map, img_path, label, gt_boxes, topk=(1, 5), threshold=None, mode='union', iou_th=0.5):
    # get topk_idx_boxes, topk_cams, gt_known_box, gt_known_cam
    top_boxes, top_maps, gt_known_box, gt_known_map = get_topk_boxes_hier(cls_logits[0], cls_map, img_path,
                                                                          label, topk=topk, threshold=threshold,
                                                                          mode=mode)
    top1_box, top5_boxes = top_boxes

    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(),
                                                              gt_boxes,
                                                              topk=(1, 5), iou_th=iou_th)
    locerr_gt_known, _ = evaluate.locerr((gt_known_box,), label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)

    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, gt_known_map, top1_wrong_detail


def eval_loc_scg(cls_logits, top_cams, gt_known_cams, aff_maps, img_path, label, gt_boxes,
                 topk=(1, 5), threshold=None, mode='union', fg_th=0.1, bg_th=0.01, iou_th=0.5, sc_maps_fo=None):
    top_boxes, top_maps = get_topk_boxes_hier_scg(cls_logits[0], top_cams, aff_maps, img_path, topk=topk,
                                                  threshold=threshold, mode=mode, fg_th=fg_th, bg_th=bg_th,
                                                  sc_maps_fo=sc_maps_fo)
    top1_box, top5_boxes = top_boxes
    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(),
                                                              gt_boxes,
                                                              topk=(1, 5), iou_th=iou_th)

    gt_known_boxes, gt_known_maps = get_topk_boxes_hier_scg(cls_logits[0], gt_known_cams, aff_maps, img_path, topk=(1,),
                                                            threshold=threshold, mode=mode, gt_labels=label,
                                                            fg_th=fg_th,
                                                            bg_th=bg_th, sc_maps_fo=sc_maps_fo)
    # update result record
    locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)

    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, top1_wrong_detail


def eval_loc_scg_v2(cls_logits, top_cams, gt_known_cams, aff_maps, img_path, label, gt_boxes,
                    topk=(1, 5), threshold=None, mode='union', fg_th=0.1, bg_th=0.01, iou_th=0.5, sc_maps_fo=None):
    top_boxes, top_maps = get_topk_boxes_scg_v2(cls_logits[0], top_cams, aff_maps, img_path, topk=topk,
                                                threshold=threshold, mode=mode,
                                                sc_maps_fo=sc_maps_fo)
    top1_box, top5_boxes = top_boxes

    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(),
                                                              gt_boxes,
                                                              topk=(1, 5), iou_th=iou_th)

    gt_known_boxes, gt_known_maps = get_topk_boxes_scg_v2(cls_logits[0], gt_known_cams, aff_maps, img_path, topk=(1,),
                                                          threshold=threshold, mode=mode, gt_labels=label,
                                                          sc_maps_fo=sc_maps_fo)

    # update result record
    locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)

    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, top1_wrong_detail


def val(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    test_record_name = args.debug_dir.split('/')[-1].split('_')[-1] + '.txt'
    with open(os.path.join(args.snapshot_dir, test_record_name), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)

    if args.dataset == 'ilsvrc':
        gt_boxes = []
        img_name = []
        with open(args.test_box, 'r') as f:
            for x in f.readlines():
                x = x.strip().split(' ')
                if len(x[1:]) % 4 == 0:
                    gt_boxes.append(list(map(float, x[1:])))
                    img_name.append(os.path.join(args.img_dir, x[0].replace('.xml', '.JPEG')))
                else:
                    print('Wrong gt bboxes.')
    elif args.dataset == 'cub':
        with open(args.test_box, 'r') as f:
            gt_boxes = [list(map(float, x.strip().split(' ')[2:])) for x in f.readlines()]
        gt_boxes = [(box[0], box[1], box[0] + box[2] - 1, box[1] + box[3] - 1) for box in gt_boxes]
    else:
        print('Wrong dataset.')
    # meters
    top1_clsacc = AverageMeter()
    top5_clsacc = AverageMeter()
    top1_clsacc.reset()
    top5_clsacc.reset()

    loc_err = {}
    for th in args.threshold:
        loc_err['top1_locerr_{}'.format(th)] = AverageMeter()
        loc_err['top1_locerr_{}'.format(th)].reset()
        loc_err['top5_locerr_{}'.format(th)] = AverageMeter()
        loc_err['top5_locerr_{}'.format(th)].reset()
        loc_err['gt_known_locerr_{}'.format(th)] = AverageMeter()
        loc_err['gt_known_locerr_{}'.format(th)].reset()
        for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
            loc_err['top1_locerr_{}_{}'.format(err, th)] = AverageMeter()
            loc_err['top1_locerr_{}_{}'.format(err, th)].reset()
        if args.scg:
            loc_err['top1_locerr_scg_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_scg_{}'.format(th)].reset()
            loc_err['top5_locerr_scg_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_scg_{}'.format(th)].reset()
            loc_err['gt_known_locerr_scg_{}'.format(th)] = AverageMeter()
            loc_err['gt_known_locerr_scg_{}'.format(th)].reset()
            for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
                loc_err['top1_locerr_scg_{}_{}'.format(err, th)] = AverageMeter()
                loc_err['top1_locerr_scg_{}_{}'.format(err, th)].reset()
    # get model
    model = get_model(args)
    model.eval()

    test_loader = data_loader(args, train=False)

    # testing
    if args.debug:
        # show_idxs = np.arange(20)
        np.random.seed(2333)
        # show_idxs = np.arange(len(test_loader))
        show_idxs = np.arange(10)
        np.random.shuffle(show_idxs)
        show_idxs = show_idxs[:]

    # evaluation classification task

    for idx, dat_test in tqdm(enumerate(test_loader)):
        # 读取用于分类的测试数据
        img_path, img, label_in = dat_test
        img_cls, img_loc = img
        img_cls_norm, img_cls_wo_norm = img_cls
        img_loc_norm, img_loc_wo_norm = img_loc

        input_cls_img = img_cls_norm if args.in_norm == 'True' else img_cls_wo_norm
        input_loc_img = img_loc_norm if args.in_norm == 'True' else img_loc_wo_norm

        if args.tencrop == 'True':
            bs, ncrops, c, h, w = input_cls_img.size()
            input_cls_img = input_cls_img.view(-1, c, h, w)

        # forward pass
        args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')
        input_cls_img, input_loc_img = input_cls_img.to(args.device), input_loc_img.to(args.device)

        # 分类任务前向
        with torch.no_grad():
            logits, _, _ = model(input_cls_img, train_flag=False)  # shape of logits:(10,200,14,14) = (10, 200, w, h)

            if args.use_thr_pool == 'True':
                # threshold avg pool
                cls_logits = model.module.thr_avg_pool(logits)
            else:
                # avg pool
                cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)  # (10,200)

            cls_logits = F.softmax(cls_logits, dim=1)  # (10,200)

            if args.tencrop == 'True':
                cls_logits = cls_logits.view(1, ncrops, -1).mean(1)  # (10,200) -> (1,10,200) -> (1, 200)

            prec1_1, prec5_1 = evaluate.accuracy(cls_logits.cpu().data, label_in.long(), topk=(1, 5))
            top1_clsacc.update(prec1_1[0].numpy(), input_cls_img.size()[0])
            top5_clsacc.update(prec5_1[0].numpy(), input_cls_img.size()[0])

        # 定位任务前向
        with torch.no_grad():
            logits, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False, scg_flag=args.scg)

            loc_map = F.relu(logits)

        for th in args.threshold:
            """
            cls_logits: (1,200)
            loc_map: (n,c,w,h)
            gt_boxes: list -> len=testset_size, gt_boxes[idx] = (xmin,ymin,xmax,ymax)
            top_maps: list -> len=5, top_maps[0].shape=(224,224), cam generated by loc_map
                      loc_map -> 
            top5_boxes: list -> len=5, top5_boxes[0]=(idx,xmin,ymin,xmax,ymax)
            """
            locerr_1, locerr_5, gt_known_locerr, top_maps, top5_boxes, gt_known_maps, top1_wrong_detail = \
                eval_loc(cls_logits, loc_map, img_path[0], label_in, gt_boxes[idx], topk=(1, 5), threshold=th,
                         mode='union', iou_th=args.iou_th)

            loc_err['top1_locerr_{}'.format(th)].update(locerr_1, input_loc_img.size()[0])
            loc_err['top5_locerr_{}'.format(th)].update(locerr_5, input_loc_img.size()[0])
            loc_err['gt_known_locerr_{}'.format(th)].update(gt_known_locerr, input_loc_img.size()[0])

            cls_wrong, multi_instances, region_part, region_more, region_wrong = top1_wrong_detail
            right = 1 - (cls_wrong + multi_instances + region_part + region_more + region_wrong)
            loc_err['top1_locerr_right_{}'.format(th)].update(right, input_loc_img.size()[0])
            loc_err['top1_locerr_cls_wrong_{}'.format(th)].update(cls_wrong, input_loc_img.size()[0])
            loc_err['top1_locerr_mins_wrong_{}'.format(th)].update(multi_instances, input_loc_img.size()[0])
            loc_err['top1_locerr_part_wrong_{}'.format(th)].update(region_part, input_loc_img.size()[0])
            loc_err['top1_locerr_more_wrong_{}'.format(th)].update(region_more, input_loc_img.size()[0])
            loc_err['top1_locerr_other_{}'.format(th)].update(region_wrong, input_loc_img.size()[0])
            if args.debug and idx in show_idxs and (th == args.threshold[3]):
                top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}'.format(cls_wrong,
                                                                                            multi_instances,
                                                                                            region_part,
                                                                                            region_more,
                                                                                            region_wrong)
                debug_dir = os.path.join(args.debug_dir, top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
                save_im_heatmap_box(img_path[0], top_maps, top5_boxes, debug_dir,
                                    gt_label=label_in.data.long().numpy(), gt_box=gt_boxes[idx],
                                    epoch=args.current_epoch, threshold=th)

            if args.scg:
                sc_maps = []
                if args.scg_com:
                    for sc_map_fo_i, sc_map_so_i in zip(sc_maps_fo, sc_maps_so):
                        if (sc_map_fo_i is not None) and (sc_map_so_i is not None):
                            sc_map_so_i = sc_map_so_i.to(args.device)
                            sc_map_i = torch.max(sc_map_fo_i, args.scg_so_weight * sc_map_so_i)
                            sc_map_i = sc_map_i / (torch.sum(sc_map_i, dim=1, keepdim=True) + 1e-10)
                            sc_maps.append(sc_map_i)
                elif args.scg_fo:
                    sc_maps = sc_maps_fo
                else:
                    sc_maps = sc_maps_so

                # locerr_1_scg, locerr_5_scg, gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
                #     eval_loc_scg(cls_logits, top_maps, gt_known_maps, sc_maps[-1] + sc_maps[-2], img_path[0], label_in,
                #                  gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                #                  fg_th=args.scg_fg_th, bg_th=args.scg_bg_th, iou_th=args.iou_th,
                #                  sc_maps_fo=None)

                locerr_1_scg, \
                locerr_5_scg, \
                gt_known_locerr_scg, \
                top_maps_scg, \
                top5_boxes_scg, \
                top1_wrong_detail_scg = \
                    eval_loc_scg_v2(cls_logits, top_maps, gt_known_maps, sc_maps[-2] + sc_maps[-1], img_path[0],
                                    label_in,
                                    gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                                    fg_th=args.scg_fg_th, bg_th=args.scg_bg_th, iou_th=args.iou_th,
                                    sc_maps_fo=None)

                loc_err['top1_locerr_scg_{}'.format(th)].update(locerr_1_scg, input_loc_img.size()[0])
                loc_err['top5_locerr_scg_{}'.format(th)].update(locerr_5_scg, input_loc_img.size()[0])
                loc_err['gt_known_locerr_scg_{}'.format(th)].update(gt_known_locerr_scg, input_loc_img.size()[0])

                cls_wrong_scg, multi_instances_scg, region_part_scg, region_more_scg, region_wrong_scg = top1_wrong_detail_scg
                right_scg = 1 - (
                        cls_wrong_scg + multi_instances_scg + region_part_scg + region_more_scg + region_wrong_scg)
                loc_err['top1_locerr_scg_right_{}'.format(th)].update(right_scg, input_loc_img.size()[0])
                loc_err['top1_locerr_scg_cls_wrong_{}'.format(th)].update(cls_wrong_scg, input_loc_img.size()[0])
                loc_err['top1_locerr_scg_mins_wrong_{}'.format(th)].update(multi_instances_scg, input_loc_img.size()[0])
                loc_err['top1_locerr_scg_part_wrong_{}'.format(th)].update(region_part_scg, input_loc_img.size()[0])
                loc_err['top1_locerr_scg_more_wrong_{}'.format(th)].update(region_more_scg, input_loc_img.size()[0])
                loc_err['top1_locerr_scg_other_{}'.format(th)].update(region_wrong_scg, input_loc_img.size()[0])

                if args.debug and idx in show_idxs and (th == args.threshold[3]):
                    top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_scg'.format(cls_wrong_scg,
                                                                                                    multi_instances_scg,
                                                                                                    region_part_scg,
                                                                                                    region_more_scg,
                                                                                                    region_wrong_scg)
                    debug_dir = os.path.join(args.debug_dir,
                                             top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
                    save_im_heatmap_box(img_path[0], top_maps_scg, top5_boxes_scg, debug_dir,
                                        gt_label=label_in.data.long().numpy(), gt_box=gt_boxes[idx],
                                        epoch=args.current_epoch, threshold=th, suffix='scg45')
                    save_im_sim(img_path[0], sc_maps_fo[-2] + sc_maps_fo[-1], debug_dir,
                                gt_label=label_in.data.long().numpy(),
                                epoch=args.current_epoch, suffix='fo_45')
                    save_im_sim(img_path[0], sc_maps_so[-2] + sc_maps_so[-1], debug_dir,
                                gt_label=label_in.data.long().numpy(),
                                epoch=args.current_epoch, suffix='so_45')
                    save_im_sim(img_path[0], sc_maps[-2] + sc_maps[-1], debug_dir,
                                gt_label=label_in.data.long().numpy(),
                                epoch=args.current_epoch, suffix='com45')

    print('== cls err')
    print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
    for th in args.threshold:
        print('=========== threshold: {} ==========='.format(th))
        print('== loc err')
        print('CAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                       loc_err['top5_locerr_{}'.format(th)].avg))
        print('CAM-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_cls_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_mins_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_part_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_more_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_other_{}'.format(th)].sum))
        if args.scg:
            print('SCG-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_{}'.format(th)].avg,
                                                           loc_err['top5_locerr_scg_{}'.format(th)].avg))
            print('SCG-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_scg_right_{}'.format(th)].sum,
                                                             loc_err[
                                                                 'top1_locerr_scg_cls_wrong_{}'.format(th)].sum,
                                                             loc_err[
                                                                 'top1_locerr_scg_mins_wrong_{}'.format(th)].sum,
                                                             loc_err[
                                                                 'top1_locerr_scg_part_wrong_{}'.format(th)].sum,
                                                             loc_err[
                                                                 'top1_locerr_scg_more_wrong_{}'.format(th)].sum,
                                                             loc_err['top1_locerr_scg_other_{}'.format(th)].sum))
        print('== Gt-Known loc err')
        print('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg))
        if args.scg:
            print('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))

    setting = args.debug_dir.split('/')[-1]
    results_log_name = '{}_results.log'.format(setting)
    result_log = os.path.join(args.snapshot_dir, results_log_name)
    with open(result_log, 'a') as fw:
        fw.write('== cls err ')
        fw.write('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
        for th in args.threshold:
            fw.write('=========== threshold: {} ===========\n'.format(th))
            fw.write('== loc err ')
            fw.write('CAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                              loc_err['top5_locerr_{}'.format(th)].avg))
            fw.write('CAM-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_cls_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_mins_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_part_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_more_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_other_{}'.format(th)].sum))
            if args.scg:
                fw.write('SCG-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_{}'.format(th)].avg,
                                                                  loc_err['top5_locerr_scg_{}'.format(th)].avg))
                fw.write('SCG-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_scg_right_{}'.format(th)].sum,
                                                                    loc_err[
                                                                        'top1_locerr_scg_cls_wrong_{}'.format(th)].sum,
                                                                    loc_err[
                                                                        'top1_locerr_scg_mins_wrong_{}'.format(th)].sum,
                                                                    loc_err[
                                                                        'top1_locerr_scg_part_wrong_{}'.format(th)].sum,
                                                                    loc_err[
                                                                        'top1_locerr_scg_more_wrong_{}'.format(th)].sum,
                                                                    loc_err['top1_locerr_scg_other_{}'.format(th)].sum))
            fw.write('== Gt-Known loc err ')
            fw.write('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg))
            if args.scg:
                fw.write('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))


if __name__ == '__main__':
    args = opts().parse()
    val(args)
