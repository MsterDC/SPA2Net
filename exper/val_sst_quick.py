import sys

sys.path.append('../')
import os
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F

from utils import evaluate_quick
from utils.loader_quick import data_loader
from utils.init_val_quick import opts, get_model, init_meters, eval_loc_all, print_fun
from models import *


def val(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    # test parameters recording
    test_record_name = args.debug_dir.split('/')[-1].split('_')[-1] + '.txt'
    with open(os.path.join(args.snapshot_dir, test_record_name), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)

    # meters initial
    params_meters, loc_err = init_meters(args)

    top1_clsacc, top5_clsacc = params_meters.get('top1_clsacc'), params_meters.get('top5_clsacc')

    # get model
    model = get_model(args)
    model.eval()

    # get data
    test_loader = data_loader(args, train=False)

    show_idxs = None
    if args.debug:
        np.random.seed(2333)
        show_idxs = np.arange(10)
        np.random.shuffle(show_idxs)
        show_idxs = show_idxs[:]

    for idx, dat_test in tqdm(enumerate(test_loader)):
        img_path, img, label_in, gt_bbox = dat_test
        img_cls, img_loc = img

        ncrops = 1
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img_cls.size()  # 20 * 10 * 200 * 224 * 224
            img_loc = img_loc.unsqueeze(1)  # 20 * 200 * 224 * 224
            img_merge = torch.cat((img_cls, img_loc), dim=1)  # 20 * 11 * 200 * 224 * 224
            img_merge = img_merge.contiguous().view(-1, c, h, w)  # 220 * 200 * 224 * 224
        else:
            bs, c, h, w = img_loc.size()
            img_cls = img_cls.unsqueeze(1)
            img_loc = img_loc.unsqueeze(1)
            img_merge = torch.cat((img_cls, img_loc), dim=1)
            img_merge = img_merge.contiguous().view(-1, c, h, w)

        # forward pass
        args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')
        img_merge = img_merge.to(args.device)

        with torch.no_grad():
            if args.mode == 'spa':
                logits, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)
            if args.mode == 'spa+sa':
                logits, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)
            if args.mode == 'sos':
                logits, pred_sos, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)
            if 'sos+sa' in args.mode:
                logits, pred_sos, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)
            if args.mode == 'spa+hinge':
                logits, hg_logits, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)
            if args.mode == 'mc_sos':
                logits, pred_sos, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)

            # shape of logits: [220,200,14,14]
            # shape of sc_maps_fo_merge: (sc_fo_2, sc_fo_3, sc_fo_4, sc_fo_5)->(none,none,[220,196,196],[220,196,196])
            # shape of sc_maps_so_merge: (sc_so_2, sc_so_3, sc_so_4, sc_so_5)->(none,none,[220,196,196],[220,196,196])
            # shape of sc_maps_fo_merge[-1]: [220, 196, 196]

            b, c, h, w = logits.size()
            logits = logits.reshape(-1, ncrops + 1, c, h, w)  # 20 * 11 * 200 * 14 * 14
            logits_cls = logits[:, :ncrops, ...]  # 20 * 10 * 200 * 14 * 14
            logits_loc = logits[:, -1, ...]  # 20 * 200 * 14 * 14
            cls_logits = torch.mean(torch.mean(logits_cls, dim=-1), dim=-1)  # 20 * 10 * 200
            cls_logits = F.softmax(cls_logits, dim=-1)  # 20 * 10 * 200 <--OR--> 20 * 200
            if args.tencrop == 'True':
                cls_logits = cls_logits.mean(1)  # 20 * 200
                if 'sos' in args.mode:
                    pred_sos = pred_sos.reshape(-1, ncrops + 1, c, h, w) if 'mc_sos' in args.mode else pred_sos.reshape(-1, ncrops + 1, h, w)
                    pred_sos = pred_sos[:, -1, ...]

            # record acc for classification
            prec1_1, prec5_1 = evaluate_quick.accuracy(cls_logits.cpu().data, label_in.long(), topk=(1, 5))
            top1_clsacc.update(prec1_1[0].numpy(), img_merge.size()[0])
            top5_clsacc.update(prec5_1[0].numpy(), img_merge.size()[0])

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

            loc_map = F.relu(logits_loc)  # 20 * 200 * 14 * 14

        params_loc = {'cls_logits': cls_logits, 'input_cls_img': img_cls, 'logits': logits,
                      'sc_maps_fo': sc_maps_fo, 'sc_maps_so': sc_maps_so, 'loc_map': loc_map,
                      'loc_err': loc_err, 'img_path': img_path, 'input_loc_img': img_loc,
                      'label_in': label_in, 'gt_boxes': gt_bbox, 'idx': idx, 'show_idxs': show_idxs, }
        if 'sos' in args.mode:
            params_loc.update({'pred_sos': pred_sos})

        loc_err = eval_loc_all(args, params_loc)

    # print val messages and record
    print_params = {'loc_err': loc_err, 'top1_clsacc': top1_clsacc, 'top5_clsacc': top5_clsacc}
    print_fun(args, print_params)

    # record results
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


if __name__ == '__main__':
    args = opts().parse()
    val(args)
