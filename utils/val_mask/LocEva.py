"""
Copyright (c) 2021-present CVM Group of JLU-AI, Kevin.
"""

import sys

sys.path.append('../../')
import os
from scipy.ndimage import label
from os.path import join as ospj
import cv2
import numpy as np

from LocToolBox import get_boxes, get_class_labels
from LocToolBox import get_image_sizes, get_pred_labels
from LocToolBox import check_scoremap_validity
from LocToolBox import check_box_convention


def extract_bbox_from_map(boolen_map):
    assert boolen_map.ndim == 2, 'Invalid input shape'
    rows = np.any(boolen_map, axis=1)
    cols = np.any(boolen_map, axis=0)
    if rows.max() == False or cols.max() == False:
        return 0, 0, 0, 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  mode='union'):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable, value of between 0 and 1
        mode: union or max

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
    """
    scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min() + 1e-10)
    check_scoremap_validity(scoremap)

    def scoremap2bbox(threshold):
        fg_map = scoremap >= threshold
        if mode == 'max':
            objects, count = label(fg_map)
            max_area = 0
            max_box = None
            for idx in range(1, count + 1):
                obj = (objects == idx)
                box = extract_bbox_from_map(obj)
                area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                if area > max_area:
                    max_area = area
                    max_box = box
            if max_box is None:
                max_box = (0, 0, 0, 0)
            return max_box
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            return box

    estimated_box_at_each_thr = []
    for threshold in scoremap_threshold_list:
        box = scoremap2bbox(threshold)
        estimated_box_at_each_thr.append(box)

    return estimated_box_at_each_thr


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, data_root, ret_root, mask_root, cam_bbox_threshold_list,
                 cam_mask_threshold_list, mode='union', topk=(1, 5)):
        self.data_root = data_root
        self.img_list_file = ospj(data_root, 'val_list.txt')
        self.box_list_file = ospj(data_root, 'val_bboxes.txt')
        self.size_list_file = ospj(data_root, 'val_sizes.txt')
        self.mask_dir = ospj(mask_root, 'images')  # ../data/mask_val/images

        # 'pred_dir' is only used in bbox evaluation.
        pred_dir = ospj(ret_root, 'preds')
        if not os.path.exists(pred_dir):
            pred_dir = ospj(ret_root, '../preds')
        self.pred_dir = pred_dir

        self.cam_bbox_threshold_list = cam_bbox_threshold_list
        self.cam_mask_threshold_list = cam_mask_threshold_list

        self.mode = mode  # only used in bbox evaluation
        self.topk = topk
        self.image_sizes = get_image_sizes(self.size_list_file)

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.gt_labels = get_class_labels(self.img_list_file)
        self.cnt = 0
        self.num_correct_gt_known = np.zeros(len(self.cam_bbox_threshold_list))
        self.num_correct_top1 = np.zeros(len(self.cam_bbox_threshold_list))
        if len(self.topk) > 1:
            self.num_correct_topk = np.zeros(len(self.cam_bbox_threshold_list))
        self.gt_bboxes = get_boxes(self.box_list_file)
        self.cls_pred = get_pred_labels(self.pred_dir)

    def accumulate(self, scoremap, image_id):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        scoremap = cv2.resize(scoremap, self.image_sizes[image_id])
        boxes_at_thresholds = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_bbox_threshold_list,
            mode=self.mode)

        # boxes_at_thresholds = np.asarray(boxes_at_thresholds)
        # print(boxes_at_thresholds)
        # print(self.gt_bboxes[image_id])
        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        max_iou = multiple_iou.max(1)
        cls_correct_top1 = (self.gt_labels[image_id] in self.cls_pred[image_id][:1])
        if len(self.topk) > 1:
            cls_correct_topk = (self.gt_labels[image_id] in self.cls_pred[image_id][:self.topk[1]])
        self.num_correct_gt_known += (max_iou > 0.5)
        self.num_correct_top1 += (max_iou > 0.5) * cls_correct_top1
        self.num_correct_topk += (max_iou > 0.5) * cls_correct_topk
        self.cnt += 1

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        box_acc = {}
        box_acc['top1'] = self.num_correct_top1 * 100.0 / float(self.cnt)
        if len(self.topk) > 1:
            box_acc['top{}'.format(self.topk[1])] = self.num_correct_topk * 100.0 / float(self.cnt)
        box_acc['gt_known'] = self.num_correct_gt_known * 100.0 / float(self.cnt)

        return box_acc
