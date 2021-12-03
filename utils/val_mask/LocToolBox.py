"""
Copyright (c) 2021-present CVM Group of JLU-AI, Kevin.
"""

import sys
sys.path.append('../../')
import numpy as np
import os
import cv2
import argparse
from os.path import join as ospj

import torch.utils.data as torchdata


class LocMapDataset(torchdata.Dataset):
    def __init__(self, loc_map_path, image_ids):
        """
        load the saved activation maps.
        :param loc_map_path: activation maps' saving path.
        :param image_ids: image_ids list.
        """
        self.loc_map_path = loc_map_path
        self.image_ids = image_ids

    def _load_loc_map(self, image_id):
        loc_map_file = ospj(self.loc_map_path, image_id + '.npy')
        return np.load(loc_map_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        loc_map = self._load_loc_map(image_id)
        return loc_map, image_id

    def __len__(self):
        return len(self.image_ids)


def _get_loc_map_loader(image_ids, loc_map_path):
    return torchdata.DataLoader(
        LocMapDataset(loc_map_path, image_ids),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


def get_image_ids(img_list_file):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    with open(img_list_file, 'r') as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n').split('.')[0])
    return image_ids


def get_class_labels(img_list_file):
    img_id_to_label = {}
    with open(img_list_file, 'r') as f:
        for line in f.readlines():
            img_name, label = line.strip().split(' ')
            img_id = img_name.split('.')[0]
            label = int(label)
            img_id_to_label[img_id] = label

    return img_id_to_label


def get_boxes(box_list_file):
    img_id_to_boxes = {}
    with open(box_list_file, 'r') as f:
        for line in f.readlines():
            boxes = []
            info = line.strip().split(' ')
            img_id = info[0].split('.')[0]
            box_info = list(map(float, info[1:]))
            box_cnt = len(box_info) // 4
            for i in range(box_cnt):
                boxes.append(box_info[i * 4:(i + 1) * 4])
            img_id_to_boxes[img_id] = boxes

    return img_id_to_boxes


def get_image_sizes(size_list_file):
    """
    image_sizes.txt has the structure
    <path>,<h>,<w>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(size_list_file, 'r') as f:
        for line in f.readlines():
            image_name, hs, ws = line.strip('\n').split(',')
            image_id = image_name.split('.')[0]
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def get_pred_labels(pred_dir):
    img_id_to_pred = {}
    for img_name in os.listdir(pred_dir):
        img_id = img_name.split('.')[0]
        preds = np.load(os.path.join(pred_dir, img_name))
        img_id_to_pred[img_id] = preds

    return img_id_to_pred


def get_masks(mask_dir):
    img_id_to_mask = {}
    for img_name in os.listdir(mask_dir):
        img_id = img_name.split('.')[0]
        mask = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        img_id_to_mask[img_id] = mask

    return img_id_to_mask


def get_loc_maps(loc_map_dir):
    img_id_to_loc_map = {}
    for file_name in os.listdir(loc_map_dir):
        img_id = file_name.split('.')[0]
        loc_map = np.load(os.path.join(loc_map_dir, file_name))
        img_id_to_loc_map[img_id] = loc_map

    return img_id_to_loc_map


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))
