# -*- coding: UTF-8 -*-
"""
@ author: chendong
Merging the 'split_result' files which generated by paralleled validation into a full result file.
"""
import os
import sys
import numpy as np
from pathlib import Path


def read_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.log':
                f_1 = open(os.path.join(root, file), 'r')
                L.append(f_1.readlines())
                f_1.close()
    return L


def parse_log(sum_data):
    th_v = 'fuck you'
    sum_dic = {}
    cls_list = []
    for id, log in enumerate(sum_data):
        sum_dic[id] = {}
        for i in range(len(log)):
            if 'cls err' in log[i]:
                cls_err_top1, cls_err_top5 = float(log[i].strip().split(' ')[-3]), float(log[i].strip().split(' ')[-1])
                cls_list.append((cls_err_top1, cls_err_top5))
            if 'threshold' in log[i]:
                th_v = log[i].strip().split(' ')[-2]
                sum_dic[id][th_v] = []
            if '=' in log[i] and 'Gt-Known' not in log[i]:
                continue
            line = log[i].strip().split(' ')
            if 'CAM' in log[i] and 'Top1' in log[i] and 'Top5' in log[i]:
                cam_loc_top1, cam_loc_top5 = float(line[1]), float(line[-1])
                sum_dic[id][th_v].append(cam_loc_top1)
                sum_dic[id][th_v].append(cam_loc_top5)
            if 'CAM-Top1_err' in log[i]:
                cam_err_detail = np.array([int(i) for i in line[1:]])
                sum_dic[id][th_v].append(cam_err_detail)
            if 'SCG' in log[i] and 'Top1' in log[i] and 'Top5' in log[i]:
                scg_loc_top1, scg_loc_top5 = float(line[1]), float(line[-1])
                sum_dic[id][th_v].append(scg_loc_top1)
                sum_dic[id][th_v].append(scg_loc_top5)
            if 'SCG-Top1_err' in log[i]:
                scg_err_detail = np.array([int(i) for i in line[1:]])
                sum_dic[id][th_v].append(scg_err_detail)
            if 'SOS' in log[i] and 'Top1' in log[i] and 'Top5' in log[i]:
                sos_loc_top1, sos_loc_top5 = float(line[1]), float(line[-1])
                sum_dic[id][th_v].append(sos_loc_top1)
                sum_dic[id][th_v].append(sos_loc_top5)
            if 'SOS-err_detail' in log[i]:
                sos_err_detail = np.array([int(i) for i in line[1:]])
                sum_dic[id][th_v].append(sos_err_detail)
            if 'Gt-Known' in log[i]:
                cam_gt_known_loc_err = float(log[i + 1].strip().split(' ')[-1])
                scg_gt_known_loc_err = float(log[i + 2].strip().split(' ')[-1])
                sos_gt_known_loc_err = float(log[i + 3].strip().split(' ')[-1])
                sum_dic[id][th_v].append(cam_gt_known_loc_err)
                sum_dic[id][th_v].append(scg_gt_known_loc_err)
                sum_dic[id][th_v].append(sos_gt_known_loc_err)
    return cls_list, sum_dic


def merge_data(parse_cls, parse_dic, nums):
    result = []
    sum_cls_err_top1, sum_cls_err_top5 = 0, 0

    for (top1, top5) in parse_cls:
        sum_cls_err_top1 += top1
        sum_cls_err_top5 += top5
    cls_err_top1, cls_err_top5 = round(sum_cls_err_top1 / nums, 2), round(sum_cls_err_top5 / nums, 2)
    print('[CLS Error] Top-1:', cls_err_top1, 'Top-5:', cls_err_top5)
    result.append('[CLS Error] Top-1: ' + str(cls_err_top1) + ' | Top-5: ' + str(cls_err_top5) + '\n')

    th_list = list(parse_dic.get(0).keys())
    for th in th_list:
        sum_cam_loc_err_top1, sum_cam_loc_err_top5 = 0, 0
        sum_scg_loc_err_top1, sum_scg_loc_err_top5 = 0, 0
        sum_sos_loc_err_top1, sum_sos_loc_err_top5 = 0, 0
        sum_cam_err_detail, sum_scg_err_detail, sum_sos_err_detail = 0, 0, 0
        sum_gt_known_cam, sum_gt_known_scg, sum_gt_known_sos = 0, 0, 0
        for id, ct in parse_dic.items():
            sum_cam_loc_err_top1 += ct.get(th)[0]
            sum_cam_loc_err_top5 += ct.get(th)[1]
            sum_cam_err_detail += ct.get(th)[2]

            sum_scg_loc_err_top1 += ct.get(th)[3]
            sum_scg_loc_err_top5 += ct.get(th)[4]
            sum_scg_err_detail += ct.get(th)[5]

            sum_sos_loc_err_top1 += ct.get(th)[6]
            sum_sos_loc_err_top5 += ct.get(th)[7]
            sum_sos_err_detail += ct.get(th)[8]

            sum_gt_known_cam += ct.get(th)[9]
            sum_gt_known_scg += ct.get(th)[10]
            sum_gt_known_sos += ct.get(th)[11]

        print('<<======>> ', th, ' <<======>>')
        print('CAM-Top1 =>', round(sum_cam_loc_err_top1/nums, 2), ' | ', 'CAM-Top5 =>', round(sum_cam_loc_err_top5/nums, 2))
        print('SCG-Top1 =>', round(sum_scg_loc_err_top1/nums, 2), ' | ', 'SCG-Top5 =>', round(sum_scg_loc_err_top5/nums, 2))
        print('SOS-Top1 =>', round(sum_sos_loc_err_top1/nums, 2), ' | ', 'SOS-Top5 =>', round(sum_sos_loc_err_top5/nums, 2))
        print('CAM_Error_Detail =>', sum_cam_err_detail.tolist())
        print('SCG_Error_Detail =>', sum_scg_err_detail.tolist())
        print('SOS_Error_Detail =>', sum_sos_err_detail.tolist())
        print('CAM-GT-Known =>', round(sum_gt_known_cam/nums, 2))
        print('SCG-GT-Known =>', round(sum_gt_known_scg/nums, 2))
        print('SOS-GT-Known =>', round(sum_gt_known_sos/nums, 2))

        result.append('<<======>> ' + th + ' <<======>>\n')
        result.append('CAM-Top1: ' + str(round(sum_cam_loc_err_top1/nums, 2)) + ' | ' + 'CAM-Top5: ' + str(round(sum_cam_loc_err_top5/nums, 2)) + '\n')
        result.append('SCG-Top1: ' + str(round(sum_scg_loc_err_top1/nums, 2)) + ' | ' + 'SCG-Top5: ' + str(round(sum_scg_loc_err_top5/nums, 2)) + '\n')
        result.append('SOS-Top1: ' + str(round(sum_sos_loc_err_top1/nums, 2)) + ' | ' + 'SOS-Top5: ' + str(round(sum_sos_loc_err_top5/nums, 2)) + '\n')
        result.append('CAM_Error_Detail: ' + ', '.join(list(map(str, sum_cam_err_detail.tolist()))) + '\n')
        result.append('SCG_Error_Detail: ' + ', '.join(list(map(str, sum_scg_err_detail.tolist()))) + '\n')
        result.append('SOS_Error_Detail: ' + ', '.join(list(map(str, sum_sos_err_detail.tolist()))) + '\n')
        result.append('CAM-GT-Known: ' + str(round(sum_gt_known_cam/nums, 2)) + '\n')
        result.append('SCG-GT-Known: ' + str(round(sum_gt_known_scg/nums, 2)) + '\n')
        result.append('SOS-GT-Known: ' + str(round(sum_gt_known_sos/nums, 2)) + '\n')
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please input the split_result_log files path!\n(Tips: Absolute path better.)')
        sys.exit()
    root = str(sys.argv[1])  # txt file dict
    sum_data = read_file(root)
    cls_err, sum_dict = parse_log(sum_data)
    res = merge_data(cls_err, sum_dict, len(sum_data))
    save_file = Path(os.path.join(root, 'result.txt'))
    if not save_file.is_file():
        with open(save_file, 'a') as f:
            for l in res:
                f.write(l)
        f.close()
    else:
        raise Exception('There is same file in the path, please check!')
