import sys
sys.path.append('../')
sys.path.append('../../')
import warnings
warnings.filterwarnings("ignore")
import os
from tqdm import tqdm
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
import argparse
from matplotlib import pyplot as plt
import seaborn as sns

from engine.engine_loader import data_loader
from engine.engine_test import sc_format
from engine.engine_locate import extract_bbox_from_map
from utils import evaluate, norm_atten_map

from utils.restore import restore
from models import *


def cal_ori_box_coor(boxes, w, h):
    if isinstance(boxes, tuple):
        boxes = list(boxes)
    else:
        boxes = boxes.split()
    coor_list = []
    box_cnt = len(boxes) // 4
    boxes = np.asarray(boxes, int)
    for i in range(box_cnt):
        bbox = boxes[i * 4:(i + 1) * 4]
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = bbox
        ori_left_top_x = int(left_top_x * w / args.crop_size)  # [Attention] 224 is the cropped size
        ori_left_top_y = int(left_top_y * h / args.crop_size)
        ori_right_bottom_x = int(right_bottom_x * w / args.crop_size)
        ori_right_bottom_y = int(right_bottom_y * h / args.crop_size)
        coor_list.append([(ori_left_top_x, ori_left_top_y), (ori_right_bottom_x, ori_right_bottom_y)])
    pass
    return coor_list


def vis_bbox_heatmap(args, img_file, maps, pred_boxes, gt_box, path):
    im = cv2.imread(img_file)
    h, w, _ = np.shape(im)
    ratio_color = 255

    # initial the numpy array to save the map temporary.
    draw_im = 255 * np.ones((h + 5, w, 3), np.uint8)
    draw_cam = 255 * np.ones((h + 5, w, 3), np.uint8)
    draw_scg = 255 * np.ones((h + 5, w, 3), np.uint8)
    if 'sos' in args.mode:
        draw_sos = 255 * np.ones((h + 5, w, 3), np.uint8)

    # parse the input params
    if 'sos' in args.mode:
        cam, scg, sos = maps
        cam_box, scg_box, sos_box = pred_boxes
        cam_box = cal_ori_box_coor(cam_box, w, h)
        scg_box = cal_ori_box_coor(scg_box, w, h)
        sos_box = cal_ori_box_coor(sos_box, w, h)
    else:
        cam, scg = maps
        cam_box, scg_box = pred_boxes
        cam_box = cal_ori_box_coor(cam_box, w, h)
        scg_box = cal_ori_box_coor(scg_box, w, h)

    draw_im[:h, :, :] = im
    # mapping to original bbox size
    gt_boxes = cal_ori_box_coor(gt_box, w, h)
    # draw gt bbox on image
    for box in gt_boxes:
        cv2.rectangle(draw_im, box[0], box[1], color=(0, 0, 255), thickness=2)

    # save original image to numpy array
    im_to_save = [draw_im.copy()]
    # re-initial the original image
    draw_im = 255 * np.ones((h + 5, w, 3), np.uint8)  # (h+15, w, 3)
    draw_im[:h, :, :] = im

    cam_ori_size = cv2.resize(cam, dsize=(w, h))
    cam_gray_map = np.uint8(ratio_color * cam_ori_size)
    cam_heatmap = cv2.applyColorMap(cam_gray_map, cv2.COLORMAP_JET)
    draw_cam[:h, :, :] = cam_heatmap * 0.7 + draw_im[:h, :, :] * 0.3
    for box, pred_box in zip(gt_boxes, cam_box):
        cv2.rectangle(draw_cam, box[0], box[1], color=(0, 0, 255), thickness=2)
        cv2.rectangle(draw_cam, pred_box[0], pred_box[1], color=(0, 255, 0), thickness=2)
    im_to_save.append(draw_cam.copy())  # save cam to numpy array

    scg_ori_size = cv2.resize(scg, dsize=(w, h))
    scg_gray_map = np.uint8(ratio_color * scg_ori_size)
    scg_heatmap = cv2.applyColorMap(scg_gray_map, cv2.COLORMAP_JET)
    draw_scg[:h, :, :] = scg_heatmap * 0.7 + draw_im[:h, :, :] * 0.3
    for box, pred_box in zip(gt_boxes, scg_box):
        cv2.rectangle(draw_scg, box[0], box[1], color=(0, 0, 255), thickness=2)
        cv2.rectangle(draw_scg, pred_box[0], pred_box[1], color=(0, 255, 0), thickness=2)
    im_to_save.append(draw_scg.copy())  # save scg to numpy array

    if 'sos' in args.mode:
        sos_ori_size = cv2.resize(sos, dsize=(w, h))
        sos_gray_map = np.uint8(ratio_color * sos_ori_size)
        sos_heatmap = cv2.applyColorMap(sos_gray_map, cv2.COLORMAP_JET)
        draw_sos[:h, :, :] = sos_heatmap * 0.7 + draw_im[:h, :, :] * 0.3
        for box, pred_box in zip(gt_boxes, sos_box):
            cv2.rectangle(draw_sos, box[0], box[1], color=(0, 0, 255), thickness=2)
            cv2.rectangle(draw_sos, pred_box[0], pred_box[1], color=(0, 255, 0), thickness=2)
        im_to_save.append(draw_sos.copy())  # save sos map to numpy array

    # concatenate all heatmap and save image
    im_to_save = np.concatenate(im_to_save, axis=1)
    cv2.imwrite(path, im_to_save)
    pass


def eval_cam(args, cam_map, gt_label):
    cam_map = cam_map.data.cpu().numpy()
    cam_map_ = cam_map[int(gt_label), :, :]
    cam_map_ = norm_atten_map(cam_map_)
    cam_map_gt_known = cv2.resize(cam_map_, dsize=(args.crop_size, args.crop_size))
    # segment the foreground
    fg_map = cam_map_gt_known >= args.threshold[0]
    box = extract_bbox_from_map(fg_map)
    return box, cam_map_gt_known


def parse_scm(args, sc_maps_fo, sc_maps_so):
    sc_maps = []
    if args.scg_com:
        for sc_map_fo_i, sc_map_so_i in zip(sc_maps_fo, sc_maps_so):
            if (sc_map_fo_i is not None) and (sc_map_so_i is not None):
                sc_map_so_i = sc_map_so_i.to(args.device)
                sc_map_i = torch.max(sc_map_fo_i, sc_map_so_i)
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
    return sc_com


def eval_scg(args, cam, sc_map):
    sc_map = sc_map.squeeze().data.cpu().numpy()
    wh_sc = sc_map.shape[0]
    h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))
    cam_map_cls = cv2.resize(cam, dsize=(w_sc, h_sc))
    cam_map_cls_vector = cam_map_cls.reshape(1, -1)
    cam_sc_dot = cam_map_cls_vector.dot(sc_map)
    cam_sc_map = cam_sc_dot.reshape(w_sc, h_sc)
    scm = cam_sc_map * (cam_sc_map >= 0)
    scm = (scm - np.min(scm)) / (np.max(scm) - np.min(scm) + 1e-10)
    scm = cv2.resize(scm, dsize=(args.crop_size, args.crop_size))
    scm = np.maximum(0, scm)
    fg_map = scm >= args.threshold[1]
    box = extract_bbox_from_map(fg_map)
    return box, scm


def eval_sos(args, loc_map):
    loc_map = torch.sigmoid(loc_map)
    pred_scm = loc_map.data.cpu().numpy()
    loc_map = cv2.resize(pred_scm, dsize=(args.crop_size, args.crop_size))
    loc_map = np.maximum(0, loc_map)
    fg_map = loc_map >= args.threshold[2]
    box = extract_bbox_from_map(fg_map)
    return box, loc_map


def save_statis(args, dict_iou):
    cam_iou = dict_iou.get('cam_iou')
    str_cam_iou = [str(i) for i in cam_iou]
    write_str_cam = ','.join(str_cam_iou)
    cam_to_save = os.path.join(args.save_dir, 'cam_iou.txt')
    write_iou(write_str_cam, cam_to_save)

    scg_iou = dict_iou.get('scg_iou')
    str_scg_iou = [str(i) for i in scg_iou]
    write_str_scg = ','.join(str_scg_iou)
    scg_to_save = os.path.join(args.save_dir, 'scg_iou.txt')
    write_iou(write_str_scg, scg_to_save)

    if 'sos' in args.mode:
        sos_iou = dict_iou.get('sos_iou')
        str_sos_iou = [str(i) for i in sos_iou]
        write_str_sos = ','.join(str_sos_iou)
        sos_to_save = os.path.join(args.save_dir, 'sos_iou.txt')
        write_iou(write_str_sos, sos_to_save)
    print("Write IoU success!")
    pass


def write_iou(iou_str, path):
    f = open(path, "w")
    f.write(iou_str)
    f.close()
    pass


def init_meters():
    pass


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SPA-Net')
        self.parser.add_argument("--root_dir", type=str, default='')
        self.parser.add_argument("--img_dir", type=str, default='')
        self.parser.add_argument("--test_list", type=str, default='')
        self.parser.add_argument("--test_box", type=str, default='')
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='cub')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_sst')
        self.parser.add_argument("--tencrop", type=str, default='True')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--in_norm", type=str, default='True', help='normalize input or not')
        self.parser.add_argument("--iou_th", type=float, default=0.5, help='the threshold for iou.')

        self.parser.add_argument("--scg_version", type=str, default='v2', help='v1 / v2')
        self.parser.add_argument("--scg_blocks", type=str, default='4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_com", action='store_true', help='fuse fo and so or not.')
        self.parser.add_argument("--scg_fo", action='store_true', help='only use fo.')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.2)
        self.parser.add_argument("--scg_sosc_th", type=float, default=1)
        self.parser.add_argument("--scgv1_bg_th", type=float, default=0.05)
        self.parser.add_argument("--scgv1_fg_th", type=float, default=0.05)

        self.parser.add_argument("--snapshot_dir", type=str, default='../snapshots')
        self.parser.add_argument("--save_dir", type=str, default='../evalbox', help='save results.')
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--threshold", type=str, help='value range of threshold')

        self.parser.add_argument("--sos_seg_method", type=str, default='TC', help='BC / TC')
        self.parser.add_argument("--sos_loss_method", type=str, default='BCE', help='BCE / MSE')

        self.parser.add_argument("--sa_use_edge", type=str, default='True', help='Add edge encoding or not')
        self.parser.add_argument("--sa_edge_stage", type=str, default='4,5', help='4 for feat4, etc.')
        self.parser.add_argument("--sa_head", type=float, default=8, help='number of SA heads')
        self.parser.add_argument("--sa_neu_num", type=float, help='channel num')

        self.parser.add_argument("--vis_bbox", action='store_true')
        self.parser.add_argument("--vis_bbox_num", type=int, default=10, help='sample number for visualization.')
        self.parser.add_argument("--vis_attention", action='store_true')
        self.parser.add_argument("--statis_bbox", action='store_true')

        self.parser.add_argument("--mode", type=str, default='sos+sa_v3')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        # eval the training mode
        mode = ['spa', 'sos', 'spa+sa', 'sos+sa_v3']
        if opt.mode not in mode:
            raise Exception('[Error] Invalid training mode, please check.')
        if 'sa' in opt.mode:
            opt.sa_neu_num = 512 if 'vgg' in opt.arch else 768
        # sparse the thresholds
        opt.threshold = list(map(float, opt.threshold.split(',')))
        return opt


def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args)
    model = torch.nn.DataParallel(model, args.gpus)
    model.cuda()
    restore(args, model, None, istrain=False)
    return model

def val(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        raise Exception("[Error] The snapshot dir is wrong, please check.")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # test parameters recording
    test_record_name = args.save_dir.split('/')[-1].split('_')[-1] + '.txt'
    with open(os.path.join(args.save_dir, test_record_name), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)

    # meters initial
    # params_meters, loc_err = init_meters(args)

    # get model
    model = get_model(args)
    model.eval()

    # get data
    test_loader = data_loader(args, train=False)
    cam_iou, scg_iou, sos_iou = [], [], []

    if args.vis_bbox:
        batch_num = args.vis_bbox_num // args.batch_size  # total epoch nums for visualization
        left_num = args.vis_bbox_num % args.batch_size  # left samples nums for visualization

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
            if args.mode == 'sos+sa_v3':
                logits, pred_sos, sc_maps_fo_merge, sc_maps_so_merge = model(img_merge, train_flag=False)

            b, c, h, w = logits.size()
            logits = logits.reshape(-1, ncrops + 1, c, h, w)  # 20 * 11 * 200 * 14 * 14
            logits_cls = logits[:, :ncrops, ...]  # 20 * 10 * 200 * 14 * 14
            logits_loc = logits[:, -1, ...]  # 20 * 200 * 14 * 14
            cls_logits = torch.mean(torch.mean(logits_cls, dim=-1), dim=-1)  # 20 * 10 * 200
            cls_logits = F.softmax(cls_logits, dim=-1)  # 20 * 10 * 200 <--OR--> 20 * 200
            if args.tencrop == 'True':
                cls_logits = cls_logits.mean(1)  # 20 * 200
                if 'sos' in args.mode:
                    pred_sos = pred_sos.reshape(-1, ncrops + 1, h, w)
                    pred_sos = pred_sos[:, -1, ...]

            # reformat sc maps
            sc_maps_fo, sc_maps_so = sc_format(sc_maps_fo_merge, sc_maps_so_merge, ncrops)
            sc_maps = parse_scm(args, sc_maps_fo, sc_maps_so)
            # normalize logits
            loc_map = F.relu(logits_loc)  # 20 * 200 * 14 * 14

        # params_loc = {'cls_logits': cls_logits, 'input_cls_img': img_cls, 'logits': logits,
        #               'sc_maps_fo': sc_maps_fo, 'sc_maps_so': sc_maps_so, 'loc_map': loc_map,
        #               'img_path': img_path, 'input_loc_img': img_loc,
        #               'label_in': label_in, 'gt_boxes': gt_bbox, 'idx': idx}
        # if 'sos' in args.mode:
        #     params_loc.update({'pred_sos': pred_sos})

        if args.vis_bbox:
            if idx + 1 <= batch_num:
                pass
            else:
                if left_num == 0:
                    break
                else:
                    left_num = 0

        for sample_idx in range(bs):
            # evaluation for CAM/SCG/SOS
            cam_bbox, cam = eval_cam(args, loc_map[sample_idx], label_in[sample_idx])
            scg_bbox, sgm = eval_scg(args,  cam, sc_maps[sample_idx])
            if 'sos' in args.mode:
                sos_bbox, ssm = eval_sos(args, pred_sos[sample_idx])

            img_id = img_path[sample_idx].split('/')[-1].split('.')[0]
            gt_bbox_idx = list(map(float, gt_bbox[sample_idx].split()))

            iou_cam = float(evaluate.cal_iou(cam_bbox, gt_bbox_idx))
            iou_scg = float(evaluate.cal_iou(scg_bbox, gt_bbox_idx))
            if 'sos' in args.mode:
                iou_sos = float(evaluate.cal_iou(sos_bbox, gt_bbox_idx))

            # save iou to list
            if args.statis_bbox:
                cam_iou.append(iou_cam)
                scg_iou.append(iou_scg)
                if 'sos' in args.mode:
                    sos_iou.append(iou_sos)

            # save attention map as numpy files
            if args.vis_attention:
                hsc_file_dir = os.path.join(args.save_dir, 'HSC')
                if not os.path.exists(hsc_file_dir):
                    os.mkdir(hsc_file_dir)
                hsc_save_path = os.path.join(hsc_file_dir, img_id+'.npy')
                hsc_file =  sc_maps[sample_idx].cpu().numpy()
                np.save(hsc_save_path, hsc_file)

                for h in range(int(args.sa_head)):
                    sa_file_dir = os.path.join(args.save_dir, 'SA')
                    if not os.path.exists(sa_file_dir):
                        os.mkdir(sa_file_dir)
                    sa_file_id = img_id + '_sa_h' + str(h) + '.npy'
                    sa_save_path = os.path.join(sa_file_dir, sa_file_id)
                    sa_file = model.module.sa.get_attention()[sample_idx][h].cpu().numpy()
                    np.save(sa_save_path, sa_file)

                for h in range(int(args.sa_head)):
                    hpsa_file_dir = os.path.join(args.save_dir, 'HPSA')
                    if not os.path.exists(hpsa_file_dir):
                        os.mkdir(hpsa_file_dir)
                    hpsa_file_id = img_id + '_hpsa_h' + str(h) + '.npy'
                    hpsa_save_path = os.path.join(hpsa_file_dir, hpsa_file_id)
                    hpsa_file = model.module.sa.get_enhanced()[sample_idx][h].cpu().numpy()
                    np.save(hpsa_save_path, hpsa_file)
                # print("Save attention map success!")

            if args.vis_bbox:
                heatmap_save_dir = os.path.join(args.save_dir, 'heatmap')
                if not os.path.exists(heatmap_save_dir):
                    os.makedirs(heatmap_save_dir)
                heatmap_save_path = os.path.join(heatmap_save_dir, img_id+'_htmap.png')
                map_save_list = []
                box_save_list = []

                map_save_list.append(cam)
                box_save_list.append(cam_bbox)

                map_save_list.append(sgm)
                box_save_list.append(scg_bbox)

                if 'sos' in args.mode:
                    map_save_list.append(ssm)
                    box_save_list.append(sos_bbox)

                vis_bbox_heatmap(args, img_path[sample_idx], map_save_list, box_save_list, gt_bbox[sample_idx], heatmap_save_path)
                pass

    # save iou to text files
    if args.statis_bbox:
        dict_iou = {'cam_iou': cam_iou, 'scg_iou': scg_iou}
        if 'sos' in args.mode:
            dict_iou.update({'sos_iou': sos_iou})
        save_statis(args, dict_iou)


if __name__ == '__main__':
    args = opts().parse()
    val(args)
    print("Congratulation! Mission Success.")

