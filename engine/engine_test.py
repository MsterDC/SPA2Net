import sys
sys.path.append('../')
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from utils.meters import AverageMeter
from utils.restore import restore
from engine.engine_locate import get_topk_boxes_hier, get_topk_boxes_hier_scg, get_topk_boxes_scg_v2, get_box_sos, \
    get_topk_boxes_sos
from utils import evaluate
from utils.vistools import debug_vis_loc, debug_vis_sc
from models import *

LR = 0.001
EPOCH = 200
DISP_INTERVAL = 50

# default settings
ROOT_DIR = os.getcwd()


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SPA-Net')
        self.parser.add_argument("--root_dir", type=str, default='')
        self.parser.add_argument("--img_dir", type=str, default='')
        self.parser.add_argument("--test_list", type=str, default='')
        self.parser.add_argument("--test_box", type=str, default='')
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='cub')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_sst')
        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--decay_points", type=str, default='none')
        self.parser.add_argument("--epoch", type=int, default=EPOCH)
        self.parser.add_argument("--tencrop", type=str, default='True')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
        self.parser.add_argument("--resume", type=str, default='True')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--debug_detail", action='store_true', help='.')
        self.parser.add_argument("--vis_dir", type=str, default='../vis_dir', help='save visualization results.')
        self.parser.add_argument("--in_norm", type=str, default='True', help='normalize input or not')
        self.parser.add_argument("--iou_th", type=float, default=0.5, help='the threshold for iou.')

        self.parser.add_argument("--scg_version", type=str, default='v2', help='v1 / v2')
        self.parser.add_argument("--scg_blocks", type=str, default='2,3,4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_com", action='store_true')
        self.parser.add_argument("--scg_fo", action='store_true')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.2)
        self.parser.add_argument("--scg_sosc_th", type=float, default=1)
        self.parser.add_argument("--scg_order", type=int, default=2, help='the order of similarity of HSC.')
        self.parser.add_argument("--scg_so_weight", type=float, default=1)

        self.parser.add_argument("--snapshot_dir", type=str, default='../snapshots')
        self.parser.add_argument("--debug_dir", type=str, default='../debug', help='save visualization results.')
        self.parser.add_argument("--threshold", type=str, default='0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5')
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')

        self.parser.add_argument("--sos_seg_method", type=str, default='TC', help='BC / TC')
        self.parser.add_argument("--sos_loss_method", type=str, default='BCE', help='BCE / MSE')

        self.parser.add_argument("--sa_use_edge", type=str, default='True', help='Add edge encoding or not')
        self.parser.add_argument("--sa_edge_weight", type=float, default=1, help='weight for edge-encoding.')
        self.parser.add_argument("--sa_edge_stage", type=str, default='4,5', help='4 for feat4, etc.')
        self.parser.add_argument("--sa_head", type=float, default=8, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, default=512, help='channel num')

        self.parser.add_argument("--mode", type=str, default='sos+sa_v3')
        self.parser.add_argument("--debug", action='store_true', help='.')
        self.parser.add_argument("--debug_num", type=int, default=10, help='visualization number, eg, 100')
        self.parser.add_argument("--debug_only", action='store_true', help='debug until debug_num')

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


def eval_loc(args, cls_logits, cls_map, label, gt_boxes, crop_size, topk=5, threshold=None, mode='union', iou_th=0.5):
    """
    @ written by Kevin
    :param cls_logits: (20, 200)
    :param cls_map: (20, 200, 14, 14)
    :param label: [20]
    :param gt_boxes: tuple => (20,)
    :param crop_size: 224
    :return: recording loc_err
    """
    _, topk_idx = cls_logits.topk(topk, 1, True, True)  # 20 * 5
    topk_idx = topk_idx.tolist()  # 20

    locerr_1_batch = []
    locerr_5_batch = []
    locerr_gt_known_batch = []
    top_maps_batch = []
    top5_boxes_batch = []
    gt_known_map_batch = []
    top1_wrong_detail_batch = []

    batch = cls_logits.shape[0]
    for ind in range(batch):
        top_boxes, top_maps, gt_known_box, gt_known_map = get_topk_boxes_hier(args, topk_idx[ind], cls_map[ind],
                                                                              label[ind], crop_size=crop_size,
                                                                              threshold=threshold,
                                                                              mode=mode)
        top1_box, top5_boxes = top_boxes
        # update result record
        gt_bbox_ind = list(map(float, gt_boxes[ind].split()))
        (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label[ind],
                                                                        gt_bbox_ind, topk=(1, 5), iou_th=iou_th)
        locerr_gt_known, _ = evaluate.locerr((gt_known_box,), label[ind], gt_bbox_ind, topk=(1,), iou_th=iou_th)

        locerr_1_batch.append(locerr_1)
        locerr_5_batch.append(locerr_5)
        locerr_gt_known_batch.append(locerr_gt_known[0])
        top_maps_batch.append(top_maps)
        top5_boxes_batch.append(top5_boxes)
        gt_known_map_batch.append(gt_known_map)
        top1_wrong_detail_batch.append(list(top1_wrong_detail))
    top1_wrong_detail_batch = np.mean(np.array(top1_wrong_detail_batch), axis=0)
    return np.mean(locerr_1_batch), np.mean(locerr_5_batch), np.mean(locerr_gt_known_batch), \
           top_maps_batch, top5_boxes_batch, gt_known_map_batch, top1_wrong_detail_batch


def eval_loc_sos(args, cls_logits, pred_scm, label, gt_boxes, topk=5, threshold=None, mode='union', iou_th=0.5):
    _, topk_idx = cls_logits.topk(topk, 1, True, True)
    topk_idx = topk_idx.tolist()
    batch = cls_logits.shape[0]
    locerr_1_batch = []
    locerr_5_batch = []
    locerr_gt_known_batch = []
    top_maps_batch = []
    top5_boxes_batch = []
    top1_wrong_detail_batch = []

    # evaluate gt-known loc error
    for ind in range(batch):
        gt_bbox_ind = list(map(float, gt_boxes[ind].split()))
        if len(pred_scm.shape) > 3:
            gt_known_sos = pred_scm[:, label[ind], ...]
        else:
            gt_known_sos = pred_scm[ind]
        if args.sos_loss_method == 'BCE':
            gt_known_sos = torch.sigmoid(gt_known_sos)
        gt_known_box, gt_known_map = get_box_sos(gt_known_sos, args.crop_size, threshold=threshold, gt_labels=label[ind])
        gt_known_locerr, gt_known_wrong_detail = evaluate.locerr(gt_known_box, label[ind], gt_bbox_ind,
                                                                       topk=(1,), iou_th=iou_th)
        # evaluate top-k loc error
        top_boxes, top_maps = get_topk_boxes_sos(topk_idx[ind], pred_scm[ind], args.crop_size, topk=(1, 5), threshold=threshold,
                                                    mode=mode, loss_method=args.sos_loss_method)
        top1_box, top5_boxes = top_boxes
        (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label[ind],
                                                                        gt_bbox_ind, topk=(1, 5), iou_th=iou_th)
        locerr_1_batch.append(locerr_1)
        locerr_5_batch.append(locerr_5)
        locerr_gt_known_batch.append(gt_known_locerr[0])
        top_maps_batch.append(top_maps)
        top5_boxes_batch.append(top5_boxes)
        top1_wrong_detail_batch.append(list(top1_wrong_detail))
    top1_wrong_detail_batch = np.mean(np.array(top1_wrong_detail_batch), axis=0)
    return np.mean(locerr_1_batch), np.mean(locerr_5_batch), np.mean(locerr_gt_known_batch), \
           top_maps_batch, top5_boxes_batch, top1_wrong_detail_batch


def eval_loc_scg_v2(cls_logits, top_cams, gt_known_cams, aff_maps, label, gt_boxes, crop_size,
                    topk=(1, 5), threshold=None, mode='union', iou_th=0.5, sc_maps_fo=None):
    _, topk_idx = cls_logits.topk(topk, 1, True, True)
    topk_idx = topk_idx.tolist()

    locerr_1_batch = []
    locerr_5_batch = []
    locerr_gt_known_batch = []
    top_maps_batch = []
    top5_boxes_batch = []
    top1_wrong_detail_batch = []

    batch = cls_logits.shape[0]
    for ind in range(batch):
        top_boxes, top_maps = get_topk_boxes_scg_v2(topk_idx[ind], top_cams[ind], aff_maps[ind], crop_size=crop_size,
                                                    threshold=threshold, mode=mode, sc_maps_fo=sc_maps_fo)
        top1_box, top5_boxes = top_boxes
        # update result record
        gt_bbox_ind = list(map(float, gt_boxes[ind].split()))
        (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label[ind],
                                                                        gt_bbox_ind, topk=(1, 5), iou_th=iou_th)
        # gt-known bbox evaluate
        gt_known_boxes, gt_known_maps = get_topk_boxes_scg_v2(topk_idx[ind], gt_known_cams[ind], aff_maps[ind],
                                                              crop_size=crop_size, topk=(1,), threshold=threshold,
                                                              mode=mode, gt_labels=label[ind], sc_maps_fo=sc_maps_fo)
        # update result record
        locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label[ind], gt_bbox_ind, topk=(1,), iou_th=iou_th)

        locerr_1_batch.append(locerr_1)
        locerr_5_batch.append(locerr_5)
        locerr_gt_known_batch.append(locerr_gt_known[0])
        top_maps_batch.append(top_maps)
        top5_boxes_batch.append(top5_boxes)
        top1_wrong_detail_batch.append(list(top1_wrong_detail))
    top1_wrong_detail_batch = np.mean(np.array(top1_wrong_detail_batch), axis=0)
    return np.mean(locerr_1_batch), np.mean(locerr_5_batch), np.mean(locerr_gt_known_batch), top_maps_batch, \
           top5_boxes_batch, top1_wrong_detail_batch


def eval_loc_scg(cls_logits, top_cams, gt_known_cams, aff_maps, label, gt_boxes, crop_size, topk=(1, 5), threshold=None,
                 mode='union', fg_th=0.1, bg_th=0.2, iou_th=0.5, sc_maps_fo=None):
    _, topk_idx = cls_logits.topk(topk, 1, True, True)
    topk_idx = topk_idx.tolist()

    locerr_1_batch = []
    locerr_5_batch = []
    locerr_gt_known_batch = []
    top_maps_batch = []
    top5_boxes_batch = []
    top1_wrong_detail_batch = []

    batch = cls_logits.shape[0]
    for ind in range(batch):
        top_boxes, top_maps = get_topk_boxes_hier_scg(topk_idx[ind], top_cams[ind], aff_maps[ind], crop_size=crop_size,
                                                      threshold=threshold, mode=mode, fg_th=fg_th, bg_th=bg_th, sc_maps_fo=sc_maps_fo)
        top1_box, top5_boxes = top_boxes
        # update result record
        gt_bbox_ind = list(map(float, gt_boxes[ind].split()))
        (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label[ind],
                                                                        gt_bbox_ind, topk=(1, 5), iou_th=iou_th)
        # gt-known bbox evaluate
        gt_known_boxes, gt_known_maps = get_topk_boxes_hier_scg(topk_idx[ind], gt_known_cams[ind], aff_maps[ind],
                                                                crop_size=crop_size, topk=(1,), threshold=threshold,
                                                                mode=mode, fg_th=fg_th, bg_th=bg_th, gt_labels=label[ind], sc_maps_fo=sc_maps_fo)
        # update result record
        locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label[ind], gt_bbox_ind, topk=(1,), iou_th=iou_th)

        locerr_1_batch.append(locerr_1)
        locerr_5_batch.append(locerr_5)
        locerr_gt_known_batch.append(locerr_gt_known[0])
        top_maps_batch.append(top_maps)
        top5_boxes_batch.append(top5_boxes)
        top1_wrong_detail_batch.append(list(top1_wrong_detail))
    top1_wrong_detail_batch = np.mean(np.array(top1_wrong_detail_batch), axis=0)
    return np.mean(locerr_1_batch), np.mean(locerr_5_batch), np.mean(locerr_gt_known_batch), top_maps_batch, \
           top5_boxes_batch, top1_wrong_detail_batch


def init_meters(args):
    # meters
    top1_clsacc = AverageMeter()
    top5_clsacc = AverageMeter()
    top1_clsacc.reset()
    top5_clsacc.reset()
    meters_params = {'top1_clsacc': top1_clsacc, 'top5_clsacc': top5_clsacc}
    if 'hinge' in args.mode:
        top1_clsacc_hg = AverageMeter()
        top5_clsacc_hg = AverageMeter()
        top1_clsacc_hg.reset()
        top5_clsacc_hg.reset()
        meters_params.update({'top1_clsacc_hg': top1_clsacc_hg, 'top5_clsacc_hg': top5_clsacc_hg})

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

        loc_err['top1_locerr_scg_{}'.format(th)] = AverageMeter()
        loc_err['top1_locerr_scg_{}'.format(th)].reset()
        loc_err['top5_locerr_scg_{}'.format(th)] = AverageMeter()
        loc_err['top5_locerr_scg_{}'.format(th)].reset()
        loc_err['gt_known_locerr_scg_{}'.format(th)] = AverageMeter()
        loc_err['gt_known_locerr_scg_{}'.format(th)].reset()
        for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
            loc_err['top1_locerr_scg_{}_{}'.format(err, th)] = AverageMeter()
            loc_err['top1_locerr_scg_{}_{}'.format(err, th)].reset()

        if 'sos' in args.mode:
            loc_err['top1_locerr_sos_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_sos_{}'.format(th)].reset()
            loc_err['top5_locerr_sos_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_sos_{}'.format(th)].reset()
            loc_err['gt_known_locerr_sos_{}'.format(th)] = AverageMeter()
            loc_err['gt_known_locerr_sos_{}'.format(th)].reset()
            for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
                loc_err['top1_locerr_{}_sos_{}'.format(err, th)] = AverageMeter()
                loc_err['top1_locerr_{}_sos_{}'.format(err, th)].reset()

        if 'hinge' in args.mode:
            loc_err['top1_locerr_hinge_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_hinge_{}'.format(th)].reset()
            loc_err['top5_locerr_hinge_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_hinge_{}'.format(th)].reset()
            loc_err['gt_known_locerr_hinge_{}'.format(th)] = AverageMeter()
            loc_err['gt_known_locerr_hinge_{}'.format(th)].reset()
            for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
                loc_err['top1_locerr_{}_hinge_{}'.format(err, th)] = AverageMeter()
                loc_err['top1_locerr_{}_hinge_{}'.format(err, th)].reset()

            loc_err['top1_locerr_scg_hinge_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_scg_hinge_{}'.format(th)].reset()
            loc_err['top5_locerr_scg_hinge_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_scg_hinge_{}'.format(th)].reset()
            loc_err['gt_known_locerr_scg_hinge_{}'.format(th)] = AverageMeter()
            loc_err['gt_known_locerr_scg_hinge_{}'.format(th)].reset()
            for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
                loc_err['top1_locerr_scg_{}_hinge_{}'.format(err, th)] = AverageMeter()
                loc_err['top1_locerr_scg_{}_hinge_{}'.format(err, th)].reset()

    return meters_params, loc_err


def eval_loc_all(args, loc_params):
    loc_err, cls_logits, loc_map, img_path, label_in, gt_boxes, input_loc_img, idx, show_idxs, sc_maps_fo, sc_maps_so = \
        loc_params.get('loc_err'), loc_params.get('cls_logits'), loc_params.get('loc_map'), loc_params.get('img_path'), \
        loc_params.get('label_in'), loc_params.get('gt_boxes'), loc_params.get('input_loc_img'), loc_params.get('idx'), \
        loc_params.get('show_idxs'), loc_params.get('sc_maps_fo'), loc_params.get('sc_maps_so')
    pred_sos = loc_params.get('pred_sos') if 'sos' in args.mode else None

    # cls_logits: 20 * 200
    # loc_map: 20 * 200 * 14 * 14
    # input_loc_img: 20 * 200 * 224 * 224
    # idx: start from 0
    # sc_maps_fo = [none, none, [20,196,196], [20,196,196]]
    # sc_maps_so = [none, none, [20,196,196], [20,196,196]]
    # pred_sos: 20 * 14 * 14

    for th in args.threshold:

        locerr_1, locerr_5, gt_known_locerr, top_maps, top5_boxes, gt_known_maps, top1_wrong_detail = \
            eval_loc(args, cls_logits, loc_map, label_in, gt_boxes, crop_size=args.crop_size, topk=5,
                     threshold=th, mode='union', iou_th=args.iou_th)

        # record CAM localization error
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

        # Visualization
        detail_cam = {'cls_wrong': cls_wrong, 'multi_instances': multi_instances,
                  'region_part': region_part, 'region_more': region_more,
                  'region_wrong': region_wrong}
        debug_vis_loc(args, idx, show_idxs, img_path, top_maps, top5_boxes, label_in.data.long().numpy(),
                      gt_boxes, detail_cam, suffix='cam')

        # SCG
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

        if args.scg_blocks == '4,5':
            sc_com = sc_maps[-2] + sc_maps[-1]
        elif args.scg_blocks == '4':
            sc_com = sc_maps[-2]
        elif args.scg_blocks == '5':
            sc_com = sc_maps[-1]
        else:
            raise Exception("[Error] HSC must be calculated by 4 or 5 stage feature of backbone.")

        locerr_1_scg, locerr_5_scg, gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
            None, None, None, None, None, None,
        if args.scg_version == 'v1':
            locerr_1_scg, locerr_5_scg, gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
                eval_loc_scg(cls_logits, top_maps, gt_known_maps, sc_com, label_in,
                             gt_boxes, crop_size=args.crop_size, topk=5, threshold=th, mode='union',
                             fg_th=0.05, bg_th=0.05, iou_th=args.iou_th, sc_maps_fo=None)
        if args.scg_version == 'v2':
            locerr_1_scg, locerr_5_scg, gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
                eval_loc_scg_v2(cls_logits, top_maps, gt_known_maps, sc_com, label_in,
                                gt_boxes, crop_size=args.crop_size, topk=5, threshold=th, mode='union',
                                iou_th=args.iou_th, sc_maps_fo=None)

        # record SCG localization error
        loc_err['top1_locerr_scg_{}'.format(th)].update(locerr_1_scg, input_loc_img.size()[0])
        loc_err['top5_locerr_scg_{}'.format(th)].update(locerr_5_scg, input_loc_img.size()[0])
        loc_err['gt_known_locerr_scg_{}'.format(th)].update(gt_known_locerr_scg, input_loc_img.size()[0])
        cls_wrong_scg, multi_instances_scg, region_part_scg, region_more_scg, region_wrong_scg = \
            top1_wrong_detail_scg
        right_scg = 1 - (
                cls_wrong_scg + multi_instances_scg + region_part_scg + region_more_scg + region_wrong_scg)
        loc_err['top1_locerr_scg_right_{}'.format(th)].update(right_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_cls_wrong_{}'.format(th)].update(cls_wrong_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_mins_wrong_{}'.format(th)].update(multi_instances_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_part_wrong_{}'.format(th)].update(region_part_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_more_wrong_{}'.format(th)].update(region_more_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_other_{}'.format(th)].update(region_wrong_scg, input_loc_img.size()[0])

        # Visualization
        detail_scg = {'cls_wrong': cls_wrong_scg, 'multi_instances': multi_instances_scg,
                      'region_part': region_part_scg, 'region_more': region_more_scg,
                      'region_wrong': region_wrong_scg}

        if args.scg_blocks == '4,5':
            sc_maps_fo_fuse = sc_maps_fo[-2] + sc_maps_fo[-1]  # add fo from stage 4 and 5
            sc_maps_so_fuse = sc_maps_so[-2] + sc_maps_so[-1]  # add so from stage 4 and 5
            suffix_sc = ['fo_45', 'so_45', 'com45']
        elif args.scg_blocks == '4':
            sc_maps_fo_fuse = sc_maps_fo[-2]
            sc_maps_so_fuse = sc_maps_so[-2]
            suffix_sc = ['fo_4', 'so_4', 'com4']
        elif args.scg_blocks == '5':
            sc_maps_fo_fuse = sc_maps_fo[-1]
            sc_maps_so_fuse = sc_maps_so[-1]
            suffix_sc = ['fo_5', 'so_5', 'com5']
        else:
            raise Exception("[Error] HSC must be calculated by 4 or 5 stage feature of backbone.")

        # Visualization for localization map and self-correlation map.
        debug_vis_loc(args, idx, show_idxs, img_path, top_maps_scg, top5_boxes_scg, label_in.data.long().numpy(), gt_boxes, detail_scg, suffix='scg')
        # debug_vis_sc(args, idx, show_idxs, img_path, sc_maps_fo_fuse, sc_maps_so_fuse, sc_com, label_in.data.long().numpy(), detail_scg, suffix=suffix_sc)

        # SOS localization
        if 'sos' in args.mode:
            locerr_1_sos, locerr_5_sos, gt_known_locerr_sos, top_sos_maps, top5_sos_boxes, top1_wrong_detail_sos = \
                eval_loc_sos(args, cls_logits, pred_sos, label_in, gt_boxes, threshold=th, iou_th=args.iou_th)
            # record SOS location error
            loc_err['top1_locerr_sos_{}'.format(th)].update(locerr_1_sos, input_loc_img.size()[0])
            loc_err['top5_locerr_sos_{}'.format(th)].update(locerr_5_sos, input_loc_img.size()[0])
            loc_err['gt_known_locerr_sos_{}'.format(th)].update(gt_known_locerr_sos, input_loc_img.size()[0])

            cls_wrong_sos, multi_instances_sos, region_part_sos, region_more_sos, region_wrong_sos = top1_wrong_detail_sos
            right_sos = 1 - (
                    cls_wrong_sos + multi_instances_sos + region_part_sos + region_more_sos + region_wrong_sos)

            loc_err['top1_locerr_right_sos_{}'.format(th)].update(right_sos, input_loc_img.size()[0])
            loc_err['top1_locerr_cls_wrong_sos_{}'.format(th)].update(cls_wrong_sos, input_loc_img.size()[0])
            loc_err['top1_locerr_mins_wrong_sos_{}'.format(th)].update(multi_instances_sos, input_loc_img.size()[0])
            loc_err['top1_locerr_part_wrong_sos_{}'.format(th)].update(region_part_sos, input_loc_img.size()[0])
            loc_err['top1_locerr_more_wrong_sos_{}'.format(th)].update(region_more_sos, input_loc_img.size()[0])
            loc_err['top1_locerr_other_sos_{}'.format(th)].update(region_wrong_sos, input_loc_img.size()[0])

            # Visualization
            detail_sos = {'cls_wrong': cls_wrong_sos, 'multi_instances': multi_instances_sos,
                          'region_part': region_part_sos, 'region_more': region_more_sos,
                          'region_wrong': region_wrong_sos}

            debug_vis_loc(args, idx, show_idxs, img_path, top_sos_maps, top5_sos_boxes, label_in.data.long().numpy(),
                          gt_boxes, detail_sos, suffix='sos')

    base_num = (args.debug_num // args.batch_size) if (args.debug_num % args.batch_size) == 0 \
        else (args.debug_num // args.batch_size) + 1
    if args.debug_only and idx >= base_num:
        print("Debug-Only Mode: Mission Complete.")
        sys.exit(0)

    return loc_err


def print_fun(args, print_params):
    top1_clsacc, top5_clsacc, loc_err = print_params.get('top1_clsacc'), print_params.get(
        'top5_clsacc'), print_params.get('loc_err')
    print('== cls err ==\n')
    print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
    for th in args.threshold:
        print('=========== threshold: {} ===========\n'.format(th))
        print('== loc err ==\n')
        print('CAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                       loc_err['top5_locerr_{}'.format(th)].avg))
        print('CAM-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_cls_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_mins_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_part_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_more_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_other_{}'.format(th)].sum))
        print('SCG-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_{}'.format(th)].avg,
                                                       loc_err['top5_locerr_scg_{}'.format(th)].avg))
        print('SCG-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_scg_right_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_scg_cls_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_scg_mins_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_scg_part_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_scg_more_wrong_{}'.format(th)].sum,
                                                         loc_err['top1_locerr_scg_other_{}'.format(th)].sum))

        if 'sos' in args.mode:
            print('SOS-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_sos_{}'.format(th)].avg,
                                                           loc_err['top5_locerr_sos_{}'.format(th)].avg))
            print('SOS-err_detail: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_sos_{}'.format(th)].sum,
                                                               loc_err['top1_locerr_cls_wrong_sos_{}'.format(th)].sum,
                                                               loc_err['top1_locerr_mins_wrong_sos_{}'.format(th)].sum,
                                                               loc_err['top1_locerr_part_wrong_sos_{}'.format(th)].sum,
                                                               loc_err['top1_locerr_more_wrong_sos_{}'.format(th)].sum,
                                                               loc_err['top1_locerr_other_sos_{}'.format(th)].sum))

        print('== Gt-Known loc err ==\n')
        print('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg))
        print('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))
        if 'sos' in args.mode:
            print('SOS-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_sos_{}'.format(th)].avg))

def res_record(args, params):
    top1_clsacc, top5_clsacc, loc_err = params.get('top1_clsacc'), params.get(
        'top5_clsacc'), params.get('loc_err')

    setting = args.debug_dir.split('/')[-1]
    results_log_name = '{}_results.log'.format(setting)
    result_log = os.path.join(args.snapshot_dir, results_log_name)
    with open(result_log, 'a') as fw:
        fw.write('== cls err ')
        fw.write('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
        for th in args.threshold:
            fw.write('=========== threshold: {} ===========\n'.format(th))
            fw.write('== loc err ==\n')
            fw.write('CAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                              loc_err['top5_locerr_{}'.format(th)].avg))
            fw.write('CAM-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_cls_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_mins_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_part_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_more_wrong_{}'.format(th)].sum,
                                                                loc_err['top1_locerr_other_{}'.format(th)].sum))
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
            if 'sos' in args.mode:
                fw.write('SOS-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_sos_{}'.format(th)].avg,
                                                                  loc_err['top5_locerr_sos_{}'.format(th)].avg))
                fw.write(
                    'SOS-err_detail: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_sos_{}'.format(th)].sum,
                                                                 loc_err[
                                                                     'top1_locerr_cls_wrong_sos_{}'.format(th)].sum,
                                                                 loc_err[
                                                                     'top1_locerr_mins_wrong_sos_{}'.format(
                                                                         th)].sum,
                                                                 loc_err[
                                                                     'top1_locerr_part_wrong_sos_{}'.format(
                                                                         th)].sum,
                                                                 loc_err[
                                                                     'top1_locerr_more_wrong_sos_{}'.format(
                                                                         th)].sum,
                                                                 loc_err[
                                                                     'top1_locerr_other_sos_{}'.format(th)].sum))

            fw.write('== Gt-Known loc err ==\n')
            fw.write('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg))
            fw.write('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))
            if 'sos' in args.mode:
                fw.write('SOS-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_sos_{}'.format(th)].avg))


def sc_format(sc_maps_fo_merge, sc_maps_so_merge, ncrops):
    sc_maps_fo = []
    sc_maps_so = []
    for sc_map_fo in sc_maps_fo_merge:
        if sc_map_fo is None:
            sc_maps_fo.append(sc_map_fo)
            continue
        n, h, w = sc_map_fo.shape
        sc_map_fo = sc_map_fo.reshape(-1, (ncrops + 1), h, w)  # 20 * 11 * 196 * 196
        sc_maps_fo.append(sc_map_fo[:, -1, :, :])  # 20 * 196 * 196
    for sc_map_so in sc_maps_so_merge:
        if sc_map_so is None:
            sc_maps_so.append(sc_map_so)
            continue
        n, h, w = sc_map_so.shape
        sc_map_so = sc_map_so.reshape(-1, (ncrops + 1), h, w)
        sc_maps_so.append(sc_map_so[:, -1, :, :])
    return sc_maps_fo, sc_maps_so
