"""
Copyright (c) 2021-present CVM Group of JLU-AI, Kevin.
"""

import sys

sys.path.append('../../')

import argparse
import numpy as np
from tqdm import tqdm
from os.path import join as ospj
import matplotlib.pyplot as plt

from LocToolBox import get_image_ids
from LocToolBox import t2n
from LocToolBox import _get_loc_map_loader
from MaskEva import MaskEvaluator


def eval_single_map(image_ids, name, image_mask_ids,
                    data_root, ret_root, mask_root, cam_bbox_threshold_list, cam_mask_threshold_list, topk=(1,)):
    """
    Evaluate single image.
    :param mask_root: ../../data/mask_val
    :param image_ids: image id list
    :param name: 'cam' or others
    :param image_mask_ids: image mask id list
    :param data_root: data root dir
    :param ret_root: predicted map path
    :param cam_bbox_threshold_list: bbox threshold list from 0 to 1.
    :param cam_mask_threshold_list: mask threshold list from 0 to 255.
    :param topk: (1,5).
    :return: evaluation result.
    """
    loc_map_path = ospj(ret_root, name)  # pred map's save path
    loc_map_loader = _get_loc_map_loader(image_ids, loc_map_path)
    evaluator_mask = MaskEvaluator(data_root=data_root,
                                   ret_root=ret_root,
                                   mask_root=mask_root,
                                   cam_bbox_threshold_list=cam_bbox_threshold_list,
                                   cam_mask_threshold_list=cam_mask_threshold_list,
                                   topk=topk)
    for loc_maps, image_ids_ in tqdm(loc_map_loader):  # for iteration
        for loc_map, image_id in zip(loc_maps, image_ids_):  # for batch
            loc_map = t2n(loc_map)  # format to numpy and detach
            if image_id in image_mask_ids:  # if match to mask id
                evaluator_mask.accumulate(loc_map, image_id)
    mask_perf = evaluator_mask.compute()
    return mask_perf


def evaluate_wsol(args, topk=(1,)):
    """
    For different activation maps, the saving dirs should be same as <name>
    :param args: arguments
    :param topk: default is (1,5)
    :return: evaluation results.
    """
    print("Loading and evaluating localization maps.")
    data_root = args.data_dir  # list path of root data
    mask_root = args.mask_root  # gt mask path
    cam_curve_interval = args.cam_curve_interval
    image_ids = get_image_ids(ospj(data_root, 'val_list.txt'))  # test data ids
    image_mask_ids = get_image_ids(ospj(mask_root, 'val_mask.txt'))  # mask data ids
    cam_bbox_threshold_list = list(np.arange(0, 1, cam_curve_interval))
    cam_mask_threshold_list = list(np.arange(0, 255))

    ret = {}
    if args.eval_cam:
        print("===========evaluate cam==============")
        mask_perf = eval_single_map(image_ids=image_ids,
                                    name="cam",
                                    image_mask_ids=image_mask_ids,
                                    data_root=data_root,
                                    ret_root=args.ret_dir,  # ../results/exp_id
                                    mask_root=args.mask_root,
                                    cam_bbox_threshold_list=cam_bbox_threshold_list,
                                    cam_mask_threshold_list=cam_mask_threshold_list,
                                    topk=topk)
        ret['cam'] = [mask_perf]

        print("===========evaluate cam-gt==============")
        mask_perf = eval_single_map(image_ids=image_ids,
                                    name="cam/gt_known",
                                    image_mask_ids=image_mask_ids,
                                    data_root=data_root,
                                    ret_root=args.ret_dir,  # ../results/exp_id
                                    mask_root=args.mask_root,
                                    cam_bbox_threshold_list=cam_bbox_threshold_list,
                                    cam_mask_threshold_list=cam_mask_threshold_list,
                                    topk=topk)
        ret['cam_gt'] = [mask_perf]

    if args.eval_scg:
        print("============evaluate scg==============")
        mask_perf = eval_single_map(image_ids=image_ids,
                                    name='scg',
                                    image_mask_ids=image_mask_ids,
                                    data_root=data_root,
                                    ret_root=args.ret_dir,  # ../results/exp_id
                                    mask_root=args.mask_root,
                                    cam_bbox_threshold_list=cam_bbox_threshold_list,
                                    cam_mask_threshold_list=cam_mask_threshold_list,
                                    topk=topk)
        ret['scg'] = [mask_perf]

        print("============evaluate scg-gt==============")
        mask_perf = eval_single_map(image_ids=image_ids,
                                    name='scg/gt_known',
                                    image_mask_ids=image_mask_ids,
                                    data_root=data_root,
                                    ret_root=args.ret_dir,  # ../results/exp_id
                                    mask_root=args.mask_root,
                                    cam_bbox_threshold_list=cam_bbox_threshold_list,
                                    cam_mask_threshold_list=cam_mask_threshold_list,
                                    topk=topk)
        ret['scg_gt'] = [mask_perf]

    if args.eval_sos:
        print("============evaluate sos==============")
        mask_perf = eval_single_map(image_ids=image_ids,
                                    name='sos',
                                    image_mask_ids=image_mask_ids,
                                    data_root=data_root,
                                    ret_root=args.ret_dir,  # ../results/exp_id
                                    mask_root=args.mask_root,
                                    cam_bbox_threshold_list=cam_bbox_threshold_list,
                                    cam_mask_threshold_list=cam_mask_threshold_list,
                                    topk=topk)
        ret['sos'] = [mask_perf]

        print("============evaluate sos-gt==============")
        mask_perf = eval_single_map(image_ids=image_ids,
                                    name='sos/gt_known',
                                    image_mask_ids=image_mask_ids,
                                    data_root=data_root,
                                    ret_root=args.ret_dir,  # ../results/exp_id
                                    mask_root=args.mask_root,
                                    cam_bbox_threshold_list=cam_bbox_threshold_list,
                                    cam_mask_threshold_list=cam_mask_threshold_list,
                                    topk=topk)
        ret['sos_gt'] = [mask_perf]

    return ret


def save_ret(ret, name, save_dir):
    mask_perf = ret[name]
    peak_iou = np.max(mask_perf[0][1]) * 100.0
    peak_th = np.argmax(mask_perf[0][1])
    f = open(ospj(save_dir, 'perf_{}.log'.format(name)), 'w')
    f.write("========={} evaluation=========\n".format(name))
    f.write("auc: {}\n".format(mask_perf[0]))
    f.write("peak iou: {}\n".format(peak_iou))
    f.write("peak threshold: {}\n".format(peak_th))
    f.close()


def save_pr_curve(fmt_map, ret, save_dir):
    plt.figure()
    plt.title("Precison-Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    for key in ret:
        precision, recall = ret[key][0][2:]
        plt.plot(recall[2:], precision[2:], fmt_map[key], label=key)
    plt.legend()
    plt.savefig(ospj(save_dir, 'PRCurve.jpg'))


def save_iouth_curve(fmt_map, ret, save_dir):
    plt.figure()
    plt.title("IoU-Threshold Curve")
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    cam_mask_threshold_list = list(np.arange(0, 255))
    th = np.append(cam_mask_threshold_list, [258, 260])
    for key in ret:
        iou = ret[key][0][1]
        plt.plot(th, iou, fmt_map[key], label=key)
    plt.legend()
    plt.savefig(ospj(save_dir, 'IoU-ThCurve.jpg'))


def main():
    """
    root_dir => results
    data_root => data
    mask_root => data/mask_val
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../../results', help="evaluation result path")
    parser.add_argument('--exp_id', type=str, default='', help="exp id with same as training")
    parser.add_argument('--data_root', type=str, default='../../data/', help="data root path, including mask path")
    parser.add_argument('--dataset', type=str, default='ilsvrc', help="default is ilsvrc")
    parser.add_argument('--mask_root', type=str, default='../../data/mask_val', help="mask gt root path")
    parser.add_argument('--cam_curve_interval', type=float, default=0.05, help="interval for evaluation")

    parser.add_argument('--eval_cam', type=int, default=1)
    parser.add_argument('--eval_scg', type=int, default=1)
    parser.add_argument('--eval_sos', type=int, default=1)
    parser.add_argument('--save_pr_curve', type=int, default=0)
    parser.add_argument('--save_iouth_curve', type=int, default=0)

    args = parser.parse_args()
    if args.dataset == 'cub':
        args.data_dir = ospj(args.data_root, 'CUB_200_2011/list')  # list path of root data
    elif args.dataset == 'ilsvrc':
        args.data_dir = ospj(args.data_root, 'ILSVRC/list')  # list path of root data

    args.ret_dir = ospj(args.root_dir, args.exp_id)  # ../results/exp_id
    args.save_dir = ospj(args.root_dir, args.exp_id)  # ../results/exp_id

    # Take the evaluation example, cams(14x14 or 28x28)
    # It needs to be saved in path <root_dir / exp_id / cam /> and the format is '{img_id}.npy'
    ret = evaluate_wsol(args, topk=(1, 5))

    fmt_map = {}

    if args.eval_cam:
        fmt_map['cam'] = 'r-.'
        save_ret(ret, 'cam', args.save_dir)
        save_ret(ret, 'cam_gt', args.save_dir)
    if args.eval_scg:
        fmt_map['scg'] = 'm-.'
        save_ret(ret, 'scg', args.save_dir)
        save_ret(ret, 'scg_gt', args.save_dir)
    if args.eval_sos:
        fmt_map['sos'] = 'c-.'
        save_ret(ret, 'sos', args.save_dir)
        save_ret(ret, 'sos_gt', args.save_dir)

    if args.save_pr_curve:
        save_pr_curve(fmt_map, ret, args.save_dir)

    if args.save_iouth_curve:
        save_iouth_curve(fmt_map, ret, args.save_dir)


if __name__ == "__main__":
    main()
