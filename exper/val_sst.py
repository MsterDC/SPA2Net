import sys

sys.path.append('../')
import os
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F

from utils import evaluate
from utils.loader import data_loader
from utils.init_val import opts, get_model, init_meters, eval_loc_all, print_fun
from utils.vistools import save_im_heatmap_box, save_im_sim, save_sim_heatmap_box, vis_feature, vis_var
from models import *


def init_dataset(args):
    gt_boxes = []
    img_name = []
    if args.dataset == 'ilsvrc':
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

    return gt_boxes, img_name


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

    gt_boxes, img_name = init_dataset(args)
    params_meters, loc_err = init_meters(args)

    top1_clsacc, top5_clsacc = params_meters.get('top1_clsacc'), params_meters.get('top5_clsacc')
    if 'hinge' in args.mode:
        top1_clsacc_hg, top5_clsacc_hg = params_meters.get('top1_clsacc_hg'), params_meters.get('top5_clsacc_hg')

    # get model
    model = get_model(args)
    model.eval()

    # get data
    test_loader = data_loader(args, train=False)

    show_idxs = None
    if args.debug:
        # show_idxs = np.arange(20)
        np.random.seed(2333)
        # show_idxs = np.arange(len(test_loader))
        show_idxs = np.arange(10)
        np.random.shuffle(show_idxs)
        show_idxs = show_idxs[:]

    for idx, dat_test in tqdm(enumerate(test_loader)):
        # 读取用于分类的测试数据
        img_path, img, label_in = dat_test
        img_cls, img_loc = img
        input_cls_img = img_cls
        input_loc_img = img_loc

        if args.tencrop == 'True':
            bs, ncrops, c, h, w = input_cls_img.size()
            input_cls_img = input_cls_img.view(-1, c, h, w)

        # forward pass
        args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')
        input_cls_img, input_loc_img = input_cls_img.to(args.device), input_loc_img.to(args.device)

        # classification task
        with torch.no_grad():
            if args.mode == 'spa':
                logits, _, _ = model(input_cls_img, train_flag=False)
            if args.mode == 'spa+sa':
                logits, _, _ = model(input_cls_img, train_flag=False)
            if args.mode == 'sos':
                logits, _, _, _ = model(input_cls_img, train_flag=False)
            if args.mode == 'sos+sa':
                logits, _, _, _ = model(input_cls_img, train_flag=False)
            if args.mode == 'spa+hinge':
                logits, hg_logits, _, _ = model(input_cls_img, train_flag=False)
            if args.mode == 'mc_sos':
                logits, _, _, _ = model(input_cls_img, train_flag=False)
            # if args.mode == 'rcst':
            #     logits, _, _, _ = model(input_cls_img, train_flag=False)
            # if args.mode == 'sst':
            #     logits, _, _, _, _ = model(input_cls_img, train_flag=False)
            # if args.mode == 'rcst+sa':
            #     logits, _, _, _ = model(input_cls_img, train_flag=False)
            # if args.mode == 'sst+sa':
            #     pass

            # global average pooling
            if args.use_tap == 'True':
                cls_logits = model.module.thr_avg_pool(logits)
            else:
                cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)  # (n, 200)

            cls_logits = F.softmax(cls_logits, dim=1)  # shape of cls_logits:(10,200)
            if args.tencrop == 'True':
                cls_logits = cls_logits.view(1, ncrops, -1).mean(1)

            # record acc for classification
            prec1_1, prec5_1 = evaluate.accuracy(cls_logits.cpu().data, label_in.long(), topk=(1, 5))
            top1_clsacc.update(prec1_1[0].numpy(), input_cls_img.size()[0])
            top5_clsacc.update(prec5_1[0].numpy(), input_cls_img.size()[0])

            if 'hinge' in args.mode:
                hg_cls_logits = torch.mean(torch.mean(hg_logits, dim=2), dim=2)
                min_val, _ = torch.min(hg_cls_logits, dim=-1, keepdim=True)
                max_val, _ = torch.max(hg_cls_logits, dim=-1, keepdim=True)
                hg_norm_logits = (hg_cls_logits - min_val) / (max_val - min_val + 1e-15)
                hg_norm_logits = F.softmax(hg_norm_logits, dim=1)  # shape of cls_logits:(10,200)
                if args.tencrop == 'True':
                    hg_norm_logits = hg_norm_logits.view(1, ncrops, -1).mean(1)
                # record acc for classification
                prec1_1_hg, prec5_1_hg = evaluate.accuracy(hg_norm_logits.cpu().data, label_in.long(), topk=(1, 5))
                top1_clsacc_hg.update(prec1_1_hg[0].numpy(), input_cls_img.size()[0])
                top5_clsacc_hg.update(prec5_1_hg[0].numpy(), input_cls_img.size()[0])

        # localization task
        with torch.no_grad():
            if args.mode == 'spa':
                logits, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False)
            if args.mode == 'spa+sa':
                logits, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False)
            if args.mode == 'sos':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False)
            if args.mode == 'sos+sa':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False)
            if args.mode == 'spa+hinge':
                logits, hg_logits, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False)
            if args.mode == 'mc_sos':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(input_loc_img, train_flag=False)
            # if args.mode == 'rcst':
            #     logits, sc_maps_fo, sc_maps_so, rcst_map = model(input_loc_img, train_flag=False)
            # if args.mode == 'sst':
            #     logits, pred_sos, sc_maps_fo, sc_maps_so, rcst_map = model(input_loc_img, train_flag=False)
            # if args.mode == 'rcst+sa':
            #     logits, sc_maps_fo, sc_maps_so, rcst_map = model(input_loc_img, train_flag=False)
            # if args.mode == 'sst+sa':
            #     pass

            loc_map = F.relu(logits)
            if 'hinge' in args.mode:
                loc_map_hg = F.relu(hg_logits)

        # Eval Loc Error with CAM / SCG / SOS
        params_loc = {'cls_logits': cls_logits, 'input_cls_img': input_cls_img, 'logits': logits,
                      'sc_maps_fo': sc_maps_fo, 'sc_maps_so': sc_maps_so, 'loc_map': loc_map,
                      'loc_err': loc_err, 'img_path': img_path, 'input_loc_img': input_loc_img,
                      'label_in': label_in, 'gt_boxes': gt_boxes, 'idx': idx, 'show_idxs': show_idxs, }
        if 'hinge' in args.mode:
            params_loc.update({'hg_logits': hg_logits, 'loc_map_hg': loc_map_hg, 'hg_norm_logits': hg_norm_logits})
        if 'sos' in args.mode:
            params_loc.update({'pred_sos': pred_sos})

        loc_err = eval_loc_all(args, params_loc)

    # validation process end #
    # print val messages and record
    print_params = {'loc_err': loc_err, 'top1_clsacc': top1_clsacc, 'top5_clsacc': top5_clsacc}
    if 'hinge' in args.mode:
        print_params.update({'top1_clsacc_hg': top1_clsacc_hg, 'top5_clsacc_hg': top5_clsacc_hg})
    print_fun(args, print_params)

    # record results with .txt file
    setting = args.debug_dir.split('/')[-1]
    results_log_name = '{}_results.log'.format(setting)
    result_log = os.path.join(args.snapshot_dir, results_log_name)
    with open(result_log, 'a') as fw:
        fw.write('== cls err ')
        fw.write('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
        if 'hinge' in args.mode:
            fw.write('== Hinge cls err ')
            fw.write('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc_hg.avg, 100.0 - top5_clsacc_hg.avg))
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
            if 'hinge' in args.mode:
                fw.write('== Hinge loc err ==\n')
                fw.write('CAM-Hinge-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_hinge_{}'.format(th)].avg,
                                                                        loc_err['top5_locerr_hinge_{}'.format(th)].avg))
                fw.write('CAM-Hinge-Top1_err: {} {} {} {} {} {}\n'.format(
                    loc_err['top1_locerr_right_hinge_{}'.format(th)].sum,
                    loc_err['top1_locerr_cls_wrong_hinge_{}'.format(th)].sum,
                    loc_err['top1_locerr_mins_wrong_hinge_{}'.format(th)].sum,
                    loc_err['top1_locerr_part_wrong_hinge_{}'.format(th)].sum,
                    loc_err['top1_locerr_more_wrong_hinge_{}'.format(th)].sum,
                    loc_err['top1_locerr_other_hinge_{}'.format(th)].sum))
                fw.write(
                    'SCG-Hinge-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_hinge_{}'.format(th)].avg,
                                                                   loc_err['top5_locerr_scg_hinge_{}'.format(th)].avg))
                fw.write('SCG-Hinge-Top1_err: {} {} {} {} {} {}\n'.format(
                    loc_err['top1_locerr_scg_right_hinge_{}'.format(th)].sum,
                    loc_err[
                        'top1_locerr_scg_cls_wrong_hinge_{}'.format(th)].sum,
                    loc_err[
                        'top1_locerr_scg_mins_wrong_hinge_{}'.format(th)].sum,
                    loc_err[
                        'top1_locerr_scg_part_wrong_hinge_{}'.format(th)].sum,
                    loc_err[
                        'top1_locerr_scg_more_wrong_hinge_{}'.format(th)].sum,
                    loc_err['top1_locerr_scg_other_hinge_{}'.format(th)].sum))
            if 'sos' in args.mode:
                fw.write('SOS-loc_err: {:.2f}\n'.format(loc_err['locerr_sos_{}'.format(th)].avg))
                fw.write('SOS-err_detail: {} {} {} {} {} {}\n'.format(loc_err['locerr_sos_right_{}'.format(th)].sum,
                                                                      loc_err['locerr_sos_cls_wrong_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'locerr_sos_mins_wrong_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'locerr_sos_part_wrong_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'locerr_sos_more_wrong_{}'.format(th)].sum,
                                                                      loc_err['locerr_sos_other_{}'.format(th)].sum))
            fw.write('== Gt-Known loc err ==\n')
            fw.write('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg))
            fw.write('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))
            if 'hinge' in args.mode:
                fw.write('== Hinge Gt-Known loc err ==\n')
                fw.write('CAM-Hinge-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_hinge_{}'.format(th)].avg))
                fw.write('SCG-Hinge-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_hinge_{}'.format(th)].avg))


if __name__ == '__main__':
    args = opts().parse()
    val(args)
