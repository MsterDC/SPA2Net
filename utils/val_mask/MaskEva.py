"""
Copyright (c) 2021-present CVM Group of JLU-AI, Kevin.
"""

import sys
sys.path.append('../../')
import cv2
import numpy as np
from os.path import join as ospj

from LocToolBox import check_scoremap_validity
from LocEva import LocalizationEvaluator


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)
        # cam_mask_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_mask_threshold_list) + 2  # 257
        self.threshold_list_right_edge = np.append(self.cam_mask_threshold_list,[255, 258, 260])  # [0,..255,258,260]

        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float)

    def accumulate(self, scoremap, image_id):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min() + 1e-10)
        check_scoremap_validity(scoremap)  # check the formulation of scoremap
        scoremap = cv2.resize(np.uint8(scoremap * 255), self.image_sizes[image_id])  # resize to ori size
        gt_mask = cv2.imread(ospj(self.mask_dir, image_id + '.png'), cv2.IMREAD_GRAYSCALE)  # load mask img
        gt_true_scores = scoremap[gt_mask > 128]
        gt_false_scores = scoremap[gt_mask <= 128]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        gt_true_hist = gt_true_hist.astype(np.float)
        self.gt_true_score_hist += gt_true_hist

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        gt_false_hist = gt_false_hist.astype(np.float)
        self.gt_false_score_hist += gt_false_hist

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        iou = (tp / (tp + fn + fp))[::-1]
        return auc, iou, precision, recall
