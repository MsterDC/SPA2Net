import os
import sys

sys.path.append('../')
import argparse

import torch
import torch.nn.functional as F

from utils import AverageMeter
from utils import evaluate
from utils.restore import restore
from utils.localization import get_topk_boxes_hier, get_topk_boxes_hier_scg, get_topk_boxes_scg_v2, get_box_sos
from utils.vistools import save_im_heatmap_box, save_im_sim
from models import *


LR = 0.001
EPOCH = 200
DISP_INTERVAL = 50

# default settings
ROOT_DIR = os.getcwd()


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='TPAMI2022-SST')
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
        self.parser.add_argument("--debug_dir", type=str, default='../debug', help='save visualization results.')
        self.parser.add_argument("--vis_dir", type=str, default='../vis_dir', help='save visualization results.')
        self.parser.add_argument("--in_norm", type=str, default='True', help='normalize input or not')
        self.parser.add_argument("--scg_blocks", type=str, default='2,3,4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_com", action='store_true')
        self.parser.add_argument("--scg_fo", action='store_true')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.2)
        self.parser.add_argument("--scg_sosc_th", type=float, default=1)
        self.parser.add_argument("--scg_order", type=int, default=2, help='the order of similarity of HSC.')
        self.parser.add_argument("--scg_so_weight", type=float, default=1)
        self.parser.add_argument("--sa_use_edge", type=str, default='True', help='Add edge encoding or not')
        self.parser.add_argument("--sa_edge_stage", type=str, default='4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--sa_head", type=float, default=1, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, default=512, help='size of SA linear input')
        self.parser.add_argument("--iou_th", type=float, default=0.5, help='the threshold for iou.')
        self.parser.add_argument("--use_tap", type=str, default='False')
        self.parser.add_argument("--tap_th", type=float, default=0.1, help='threshold avg pooling')
        self.parser.add_argument("--mode", type=str, default='sos+sa')

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
    top_boxes, top_maps, gt_known_box, gt_known_map = get_topk_boxes_hier(cls_logits[0], cls_map, img_path, label,
                                                                          topk=topk, threshold=threshold, mode=mode)
    top1_box, top5_boxes = top_boxes
    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(),
                                                              gt_boxes, topk=(1, 5), iou_th=iou_th)
    locerr_gt_known, _ = evaluate.locerr((gt_known_box,), label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)

    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, gt_known_map, top1_wrong_detail


def eval_loc_sos(pred_scm, img_path, label, gt_boxes, threshold, iou_th=0.5):
    pred_box, maps = get_box_sos(pred_scm, img_path, threshold=threshold, gt_labels=label)
    # update result record
    locerr, wrong_detail = evaluate.locerr(pred_box, label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)
    return locerr[0], wrong_detail, pred_box, maps


def eval_loc_scg_v2(cls_logits, top_cams, gt_known_cams, aff_maps, img_path, label, gt_boxes,
                    topk=(1, 5), threshold=None, mode='union', iou_th=0.5, sc_maps_fo=None):
    top_boxes, top_maps = get_topk_boxes_scg_v2(cls_logits[0], top_cams, aff_maps, img_path, topk=topk,
                                                threshold=threshold, mode=mode, sc_maps_fo=sc_maps_fo)
    top1_box, top5_boxes = top_boxes
    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(),
                                                              gt_boxes, topk=(1, 5), iou_th=iou_th)

    # gt-known
    gt_known_boxes, gt_known_maps = get_topk_boxes_scg_v2(cls_logits[0], gt_known_cams, aff_maps, img_path, topk=(1,),
                                                          threshold=threshold, mode=mode, gt_labels=label,
                                                          sc_maps_fo=sc_maps_fo)
    # update result record
    locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)
    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, top1_wrong_detail


def eval_loc_scg(cls_logits, top_cams, gt_known_cams, aff_maps, img_path, label, gt_boxes,
                 topk=(1, 5), threshold=None, mode='union', fg_th=0.1, bg_th=0.01, iou_th=0.5, sc_maps_fo=None):
    top_boxes, top_maps = get_topk_boxes_hier_scg(cls_logits[0], top_cams, aff_maps, img_path, topk=topk,
                                                  threshold=threshold, mode=mode, fg_th=fg_th, bg_th=bg_th,
                                                  sc_maps_fo=sc_maps_fo)
    top1_box, top5_boxes = top_boxes
    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(),
                                                              gt_boxes, topk=(1, 5), iou_th=iou_th)
    gt_known_boxes, gt_known_maps = get_topk_boxes_hier_scg(cls_logits[0], gt_known_cams, aff_maps, img_path, topk=(1,),
                                                            threshold=threshold, mode=mode, gt_labels=label,
                                                            fg_th=fg_th, bg_th=bg_th, sc_maps_fo=sc_maps_fo)
    # update result record
    locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)
    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, top1_wrong_detail


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
            loc_err['locerr_sos_{}'.format(th)] = AverageMeter()
            loc_err['locerr_sos_{}'.format(th)].reset()
            for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
                loc_err['locerr_sos_{}_{}'.format(err, th)] = AverageMeter()
                loc_err['locerr_sos_{}_{}'.format(err, th)].reset()
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

    if 'hinge' in args.mode:
        hg_norm_logits, loc_map_hg = loc_params.get('hg_norm_logits'), loc_params.get('loc_map_hg')

    if 'sos' in args.mode:
        pred_sos = loc_params.get('pred_sos')

    for th in args.threshold:
        # CAM localization: top_maps与原始图像尺寸相同
        locerr_1, locerr_5, gt_known_locerr, top_maps, top5_boxes, gt_known_maps, top1_wrong_detail = \
            eval_loc(cls_logits, loc_map, img_path[0], label_in, gt_boxes[idx], topk=(1, 5), threshold=th,
                     mode='union', iou_th=args.iou_th)
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
        if args.debug and idx in show_idxs and (th == args.threshold[1]):
            top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}'.format(cls_wrong, multi_instances,
                                                                                        region_part, region_more,
                                                                                        region_wrong)
            debug_dir = os.path.join(args.debug_dir, top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
            save_im_heatmap_box(img_path[0], top_maps, top5_boxes, debug_dir, gt_label=label_in.data.long().numpy(),
                                gt_box=gt_boxes[idx], epoch=args.current_epoch, threshold=th)

        # SCG localization: SCM + CAM
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

        # SCGv1 algorithm
        # locerr_1_scg, locerr_5_scg, gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
        #     eval_loc_scg(cls_logits, top_maps, gt_known_maps, sc_maps[-1] + sc_maps[-2], img_path[0], label_in,
        #                  gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
        #                  fg_th=args.scg_fg_th, bg_th=args.scg_bg_th, iou_th=args.iou_th,
        #                  sc_maps_fo=None)

        # SCGv2 algorithm: top_maps_scg 与原始图像相同尺寸
        locerr_1_scg, locerr_5_scg, gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
            eval_loc_scg_v2(cls_logits, top_maps, gt_known_maps, sc_maps[-2] + sc_maps[-1], img_path[0],
                            label_in, gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                            iou_th=args.iou_th,
                            sc_maps_fo=None)

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
        loc_err['top1_locerr_scg_mins_wrong_{}'.format(th)].update(multi_instances_scg,
                                                                   input_loc_img.size()[0])
        loc_err['top1_locerr_scg_part_wrong_{}'.format(th)].update(region_part_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_more_wrong_{}'.format(th)].update(region_more_scg, input_loc_img.size()[0])
        loc_err['top1_locerr_scg_other_{}'.format(th)].update(region_wrong_scg, input_loc_img.size()[0])

        # Visualization
        if args.debug and idx in show_idxs and (th == args.threshold[1]):
            top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_scg'.format(cls_wrong_scg,
                                                                                            multi_instances_scg,
                                                                                            region_part_scg,
                                                                                            region_more_scg,
                                                                                            region_wrong_scg)
            debug_dir = os.path.join(args.debug_dir,
                                     top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
            save_im_heatmap_box(img_path[0], top_maps_scg, top5_boxes_scg, debug_dir,
                                gt_label=label_in.data.long().numpy(), gt_box=gt_boxes[idx],
                                epoch=args.current_epoch, threshold=th, suffix='scg')
            save_im_sim(img_path[0], sc_maps_fo[-2] + sc_maps_fo[-1], debug_dir,
                        gt_label=label_in.data.long().numpy(), epoch=args.current_epoch, suffix='fo_45')
            save_im_sim(img_path[0], sc_maps_so[-2] + sc_maps_so[-1], debug_dir,
                        gt_label=label_in.data.long().numpy(), epoch=args.current_epoch, suffix='so_45')
            save_im_sim(img_path[0], sc_maps[-2] + sc_maps[-1], debug_dir,
                        gt_label=label_in.data.long().numpy(),
                        epoch=args.current_epoch, suffix='com45')

        ################################ Hinge Branch ###################################
        if 'hinge' in args.mode:
            # Hinge branch localization: top_maps与原始图像尺寸相同
            locerr_1_hg, locerr_5_hg, gt_known_locerr_hg, top_maps_hg, top5_boxes_hg, gt_known_maps_hg, top1_wrong_detail_hg = \
                eval_loc(hg_norm_logits, loc_map_hg, img_path[0], label_in, gt_boxes[idx], topk=(1, 5),
                         threshold=th, mode='union', iou_th=args.iou_th)
            # Hinge branch localization error
            loc_err['top1_locerr_hinge_{}'.format(th)].update(locerr_1_hg, input_loc_img.size()[0])
            loc_err['top5_locerr_hinge_{}'.format(th)].update(locerr_5_hg, input_loc_img.size()[0])
            loc_err['gt_known_locerr_hinge_{}'.format(th)].update(gt_known_locerr_hg, input_loc_img.size()[0])
            cls_wrong_hg, multi_instances_hg, region_part_hg, region_more_hg, region_wrong_hg = top1_wrong_detail_hg
            right_hg = 1 - (cls_wrong_hg + multi_instances_hg + region_part_hg + region_more_hg + region_wrong_hg)
            loc_err['top1_locerr_right_hinge_{}'.format(th)].update(right_hg, input_loc_img.size()[0])
            loc_err['top1_locerr_cls_wrong_hinge_{}'.format(th)].update(cls_wrong_hg, input_loc_img.size()[0])
            loc_err['top1_locerr_mins_wrong_hinge_{}'.format(th)].update(multi_instances_hg,
                                                                         input_loc_img.size()[0])
            loc_err['top1_locerr_part_wrong_hinge_{}'.format(th)].update(region_part_hg, input_loc_img.size()[0])
            loc_err['top1_locerr_more_wrong_hinge_{}'.format(th)].update(region_more_hg, input_loc_img.size()[0])
            loc_err['top1_locerr_other_hinge_{}'.format(th)].update(region_wrong_hg, input_loc_img.size()[0])
            if args.debug and idx in show_idxs and (th == args.threshold[1]):
                top1_wrong_detail_dir_hg = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_hinge_{}'.format(cls_wrong_hg,
                                                                                                     multi_instances_hg,
                                                                                                     region_part_hg,
                                                                                                     region_more_hg,
                                                                                                     region_wrong_hg)
                debug_dir_hg = os.path.join(args.debug_dir,
                                            top1_wrong_detail_dir_hg) if args.debug_detail else args.debug_dir
                save_im_heatmap_box(img_path[0], top_maps_hg, top5_boxes_hg, debug_dir_hg,
                                    gt_label=label_in.data.long().numpy(),
                                    gt_box=gt_boxes[idx], epoch=args.current_epoch, threshold=th, suffix='cam_hg')
            # SCGv1 algorithm
            # locerr_1_scg_hg, locerr_5_scg_hg, gt_known_locerr_scg_hg, top_maps_scg_hg, top5_boxes_scg_hg, top1_wrong_detail_scg_hg = \
            #     eval_loc_scg(hg_norm_logits, top_maps_hg, gt_known_maps_hg, sc_maps[-1] + sc_maps[-2], img_path[0], label_in,
            #                  gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
            #                  fg_th=args.scg_fg_th, bg_th=args.scg_bg_th, iou_th=args.iou_th,
            #                  sc_maps_fo=None)

            # SCGv2 algorithm: top_maps_scg 与原始图像相同尺寸
            locerr_1_scg_hg, locerr_5_scg_hg, gt_known_locerr_scg_hg, top_maps_scg_hg, top5_boxes_scg_hg, top1_wrong_detail_scg_hg = \
                eval_loc_scg_v2(hg_norm_logits, top_maps_hg, gt_known_maps_hg, sc_maps[-2] + sc_maps[-1],
                                img_path[0],
                                label_in, gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                                iou_th=args.iou_th, sc_maps_fo=None)

            # record SCG localization error
            loc_err['top1_locerr_scg_hinge_{}'.format(th)].update(locerr_1_scg_hg, input_loc_img.size()[0])
            loc_err['top5_locerr_scg_hinge_{}'.format(th)].update(locerr_5_scg_hg, input_loc_img.size()[0])
            loc_err['gt_known_locerr_scg_hinge_{}'.format(th)].update(gt_known_locerr_scg_hg,
                                                                      input_loc_img.size()[0])
            cls_wrong_scg_hg, multi_instances_scg_hg, region_part_scg_hg, region_more_scg_hg, region_wrong_scg_hg = \
                top1_wrong_detail_scg_hg
            right_scg_hg = 1 - (
                    cls_wrong_scg_hg + multi_instances_scg_hg + region_part_scg_hg + region_more_scg_hg + region_wrong_scg_hg)
            loc_err['top1_locerr_scg_right_hinge_{}'.format(th)].update(right_scg_hg, input_loc_img.size()[0])
            loc_err['top1_locerr_scg_cls_wrong_hinge_{}'.format(th)].update(cls_wrong_scg_hg,
                                                                            input_loc_img.size()[0])
            loc_err['top1_locerr_scg_mins_wrong_hinge_{}'.format(th)].update(multi_instances_scg_hg,
                                                                             input_loc_img.size()[0])
            loc_err['top1_locerr_scg_part_wrong_hinge_{}'.format(th)].update(region_part_scg_hg,
                                                                             input_loc_img.size()[0])
            loc_err['top1_locerr_scg_more_wrong_hinge_{}'.format(th)].update(region_more_scg_hg,
                                                                             input_loc_img.size()[0])
            loc_err['top1_locerr_scg_other_hinge_{}'.format(th)].update(region_wrong_scg_hg,
                                                                        input_loc_img.size()[0])

            # Visualization
            if args.debug and idx in show_idxs and (th == args.threshold[1]):
                top1_wrong_detail_dir_hg = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_scg_hinge'.format(
                    cls_wrong_scg_hg,
                    multi_instances_scg_hg,
                    region_part_scg_hg,
                    region_more_scg_hg,
                    region_wrong_scg_hg)
                debug_dir = os.path.join(args.debug_dir,
                                         top1_wrong_detail_dir_hg) if args.debug_detail else args.debug_dir
                save_im_heatmap_box(img_path[0], top_maps_scg_hg, top5_boxes_scg_hg, debug_dir,
                                    gt_label=label_in.data.long().numpy(), gt_box=gt_boxes[idx],
                                    epoch=args.current_epoch, threshold=th, suffix='scg_hg')
                save_im_sim(img_path[0], sc_maps_fo[-2] + sc_maps_fo[-1], debug_dir,
                            gt_label=label_in.data.long().numpy(), epoch=args.current_epoch, suffix='fo_45_hg')
                save_im_sim(img_path[0], sc_maps_so[-2] + sc_maps_so[-1], debug_dir,
                            gt_label=label_in.data.long().numpy(), epoch=args.current_epoch, suffix='so_45_hg')
                save_im_sim(img_path[0], sc_maps[-2] + sc_maps[-1], debug_dir,
                            gt_label=label_in.data.long().numpy(),
                            epoch=args.current_epoch, suffix='com45_hg')

        # SOS localization
        if 'sos' in args.mode:
            output = F.sigmoid(pred_sos)
            # pred_sos = torch.where(output >= 0.9, torch.ones_like(output), torch.zeros_like(output))
            locerr_sos, wrong_detail_sos, pred_box, maps_sos = eval_loc_sos(output, img_path[0], label_in,
                                                                            gt_boxes[idx], threshold=th,
                                                                            iou_th=args.iou_th)
            # record SOS location error
            loc_err['locerr_sos_{}'.format(th)].update(locerr_sos, input_loc_img.size()[0])
            cls_wrong_sos, multi_instances_sos, region_part_sos, region_more_sos, region_wrong_sos = \
                wrong_detail_sos
            right_sos = 1 - (
                    cls_wrong_sos + multi_instances_sos + region_part_sos + region_more_sos + region_wrong_sos)
            loc_err['locerr_sos_right_{}'.format(th)].update(right_sos, input_loc_img.size()[0])
            loc_err['locerr_sos_cls_wrong_{}'.format(th)].update(cls_wrong_sos, input_loc_img.size()[0])
            loc_err['locerr_sos_mins_wrong_{}'.format(th)].update(multi_instances_sos, input_loc_img.size()[0])
            loc_err['locerr_sos_part_wrong_{}'.format(th)].update(region_part_sos, input_loc_img.size()[0])
            loc_err['locerr_sos_more_wrong_{}'.format(th)].update(region_more_sos, input_loc_img.size()[0])
            loc_err['locerr_sos_other_{}'.format(th)].update(region_wrong_sos, input_loc_img.size()[0])

            # Visualization
            if args.debug and idx in show_idxs and (th == args.threshold[-1]):
                wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_sos'.format(cls_wrong_sos,
                                                                                           multi_instances_sos,
                                                                                           region_part_sos,
                                                                                           region_more_sos,
                                                                                           region_wrong_sos)

                debug_dir = os.path.join(args.debug_dir, wrong_detail_dir) if args.debug_detail else args.debug_dir
                save_im_heatmap_box(img_path[0], maps_sos, pred_box[0], debug_dir,
                                    gt_label=label_in.data.long().numpy(),
                                    gt_box=gt_boxes[idx], epoch=args.current_epoch, threshold=th, suffix='sos')
                # 采样测试待重新开发
    return loc_err


def print_fun(args, print_params):
    top1_clsacc, top5_clsacc, loc_err = print_params.get('top1_clsacc'), print_params.get(
        'top5_clsacc'), print_params.get('loc_err')
    if 'hinge' in args.mode:
        top1_clsacc_hg, top5_clsacc_hg = print_params.get('top1_clsacc_hg'), print_params.get('top5_clsacc_hg')
    print('== cls err ==\n')
    print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
    if 'hinge' in args.mode:
        print('== hinge cls err ==\n')
        print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc_hg.avg, 100.0 - top5_clsacc_hg.avg))
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
        if 'hinge' in args.mode:
            print('CAM-Hinge-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_hinge_{}'.format(th)].avg,
                                                                 loc_err['top5_locerr_hinge_{}'.format(th)].avg))
            print('CAM-Hinge-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_hinge_{}'.format(th)].sum,
                                                                   loc_err[
                                                                       'top1_locerr_cls_wrong_hinge_{}'.format(th)].sum,
                                                                   loc_err['top1_locerr_mins_wrong_hinge_{}'.format(
                                                                       th)].sum,
                                                                   loc_err['top1_locerr_part_wrong_hinge_{}'.format(
                                                                       th)].sum,
                                                                   loc_err['top1_locerr_more_wrong_hinge_{}'.format(
                                                                       th)].sum,
                                                                   loc_err[
                                                                       'top1_locerr_other_hinge_{}'.format(th)].sum))
            print('SCG-Hinge-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_hinge_{}'.format(th)].avg,
                                                                 loc_err['top5_locerr_scg_hinge_{}'.format(th)].avg))
            print('SCG-Hinge-Top1_err: {} {} {} {} {} {}\n'.format(
                loc_err['top1_locerr_scg_right_hinge_{}'.format(th)].sum,
                loc_err['top1_locerr_scg_cls_wrong_hinge_{}'.format(th)].sum,
                loc_err['top1_locerr_scg_mins_wrong_hinge_{}'.format(th)].sum,
                loc_err['top1_locerr_scg_part_wrong_hinge_{}'.format(th)].sum,
                loc_err['top1_locerr_scg_more_wrong_hinge_{}'.format(th)].sum,
                loc_err['top1_locerr_scg_other_hinge_{}'.format(th)].sum))

        if 'sos' in args.mode:
            print('SOS-loc_err: {:.2f}\n'.format(loc_err['locerr_sos_{}'.format(th)].avg))
            print('SOS-err_detail: {} {} {} {} {} {}\n'.format(loc_err['locerr_sos_right_{}'.format(th)].sum,
                                                               loc_err['locerr_sos_cls_wrong_{}'.format(th)].sum,
                                                               loc_err['locerr_sos_mins_wrong_{}'.format(th)].sum,
                                                               loc_err['locerr_sos_part_wrong_{}'.format(th)].sum,
                                                               loc_err['locerr_sos_more_wrong_{}'.format(th)].sum,
                                                               loc_err['locerr_sos_other_{}'.format(th)].sum))

        print('== Gt-Known loc err ==\n')
        print('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg))
        print('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))
        if 'hinge' in args.mode:
            print('== Hinge Gt-Known loc err ==\n')
            print('CAM-Hinge-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_hinge_{}'.format(th)].avg))
            print('SCG-Hinge-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_hinge_{}'.format(th)].avg))
