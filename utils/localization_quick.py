import numpy as np
import cv2
import torch
from scipy.ndimage import label
from .vistools_quick import norm_atten_map, NormalizationFamily
import torch.nn.functional as F


def get_topk_boxes_hier(args, cls_inds, cam_map, gt_label, crop_size, threshold, mode='union'):
    """
    @ author: Kevin
    :param cls_inds: list => [cls_idx_1, cls_idx_2, ... cls_idx_5]
    :param cam_map: [200, 14, 14]
    :param gt_label: [1]
    :param crop_size: 224 ???? correct?
    :param mode: union
    :param threshold: 0.1 ~ 0.5
    :return: result, maxk_maps, gt_known_boxes, gt_known_maps
    """
    topk = (1, 5)
    cam_map = cam_map.data.cpu().numpy()
    # get original image size and scale
    maxk_boxes = []
    maxk_maps = []
    for cls in cls_inds:
        cam_map_ = cam_map[cls, :, :]  # (14,14)

        # using different norm function
        norm_fun = NormalizationFamily()
        cam_map_ = norm_fun(args.norm_fun, cam_map_, args.percentile)
        # cam_map_ = norm_atten_map(cam_map_)  # (14,14)

        # TODO(Kevin): crop_size correctly ?
        cam_map_cls = cv2.resize(cam_map_, dsize=(crop_size, crop_size))

        maxk_maps.append(cam_map_.copy())
        # segment the foreground
        fg_map = cam_map_cls >= threshold

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
            max_box = (cls,) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            maxk_boxes.append((cls,) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]  # [[top1],[top5]]
    # gt_known
    gt_known_boxes = []
    gt_known_maps = []
    cam_map_ = cam_map[int(gt_label), :, :]

    # using different norm function
    norm_fun = NormalizationFamily()
    cam_map_ = norm_fun(args.norm_fun, cam_map_, args.percentile)
    # cam_map_ = norm_atten_map(cam_map_)

    # TODO(Kevin): crop_size correctly？
    cam_map_gt_known = cv2.resize(cam_map_, dsize=(crop_size, crop_size))

    gt_known_maps.append(cam_map_gt_known.copy())
    # segment the foreground
    fg_map = cam_map_gt_known >= threshold

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
        max_box = (int(gt_label),) + max_box
        gt_known_boxes.append(max_box)
    elif mode == 'union':
        box = extract_bbox_from_map(fg_map)
        gt_known_boxes.append((int(gt_label),) + box)
    else:
        raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    return result, maxk_maps, gt_known_boxes, gt_known_maps


def get_box_sos(pred_scm, crop_size, threshold, gt_labels):
    pred_scm = pred_scm.data.cpu().numpy()
    sc_map = cv2.resize(pred_scm, dsize=(crop_size, crop_size))  # (14,14) => ori size
    sc_map_cls = np.maximum(0, sc_map)
    maxk_maps = [sc_map_cls.copy()]
    fg_map = sc_map_cls >= threshold
    box = extract_bbox_from_map(fg_map)
    maxk_boxes = [(int(gt_labels),) + box]  # [(label, xmin, ymin, xmax, ymax)]
    result = [maxk_boxes[:k] for k in (1,)]  # [[(label, xmin, ymin, xmax, ymax)]]
    return result, maxk_maps


def get_topk_boxes_sos(cls_inds, sos_map, crop_size, topk=(1, 5), gt_labels=None,
                          threshold=0.2, mode='union', loss_method='BCE'):
    maxk_boxes = []
    maxk_maps = []
    for i in range(max(topk)):
        top_i_sos = sos_map[cls_inds[i]] if len(sos_map.shape) > 3 else sos_map
        sos_map_top_i = torch.sigmoid(top_i_sos) if loss_method == 'BCE' else top_i_sos
        sos_map_top_i = sos_map_top_i.data.cpu().numpy()
        sos_map_top_i = cv2.resize(sos_map_top_i, dsize=(crop_size, crop_size))
        sos_map_top_i_cls = np.maximum(0, sos_map_top_i)
        maxk_maps.append(sos_map_top_i_cls.copy())
        # extract boxes
        fg_map = sos_map_top_i_cls >= threshold
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
            if gt_labels is not None:
                max_box = (int(gt_labels[0]),) + max_box
            else:
                max_box = (cls_inds[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            if gt_labels is not None:
                maxk_boxes.append((int(gt_labels),) + box)
            else:
                maxk_boxes.append((cls_inds[i],) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]
    return result, maxk_maps


def get_topk_boxes_scg_v2(args, cls_inds, top_cams, sc_maps, crop_size, topk=(1, 5), gt_labels=None, threshold=0.2,
                          mode='union', sc_maps_fo=None):
    if isinstance(sc_maps, tuple) or isinstance(sc_maps, list):
        pass
    else:
        sc_maps = [sc_maps]
    if sc_maps_fo is not None:
        if isinstance(sc_maps_fo, tuple) or isinstance(sc_maps_fo, list):
            pass
        else:
            sc_maps_fo = [sc_maps_fo]

    maxk_boxes = []
    maxk_maps = []
    for i in range(max(topk)):
        sc_map_cls = 0
        for j, sc_map in enumerate(sc_maps):
            # shape of sc_map:(1,196,196)
            cam_map_cls = top_cams[i]  # The size of CAM is the same as ori image.
            sc_map = sc_map.squeeze().data.cpu().numpy()
            # shape of sc_map:(196,196)
            wh_sc = sc_map.shape[0]
            h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))  # 14,14
            cam_map_cls = cv2.resize(cam_map_cls, dsize=(w_sc, h_sc))
            cam_map_cls_vector = cam_map_cls.reshape(1, -1)  # (1,196)
            cam_sc_dot = cam_map_cls_vector.dot(sc_map)  # (1,196)
            cam_sc_map = cam_sc_dot.reshape(w_sc, h_sc)
            sc_map_cls_i = cam_sc_map * (cam_sc_map >= 0)

            # using different norm function
            norm_fun = NormalizationFamily()
            sc_map_cls_i = norm_fun(args.norm_fun, sc_map_cls_i, args.percentile)
            # sc_map_cls_i = (sc_map_cls_i - np.min(sc_map_cls_i)) / (np.max(sc_map_cls_i) - np.min(sc_map_cls_i) + 1e-10)

            sc_map_cls_i = cv2.resize(sc_map_cls_i, dsize=(crop_size, crop_size))
            sc_map_cls = np.maximum(sc_map_cls, sc_map_cls_i)

        maxk_maps.append(sc_map_cls.copy())
        # segment the foreground
        fg_map = sc_map_cls >= threshold

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
            if gt_labels is not None:
                max_box = (int(gt_labels),) + max_box
            else:
                max_box = (cls_inds[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            if gt_labels is not None:
                maxk_boxes.append((int(gt_labels),) + box)
            else:
                maxk_boxes.append((cls_inds[i],) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]
    return result, maxk_maps


def get_topk_boxes_hier_scg(cls_inds, top_cams, sc_maps, crop_size, topk=(1, 5), gt_labels=None, threshold=0.2,
                            mode='union', fg_th=0.1, bg_th=0.2, sc_maps_fo=None):

    if isinstance(sc_maps, tuple) or isinstance(sc_maps, list):
        pass
    else:
        sc_maps = [sc_maps]
    if sc_maps_fo is not None:
        if isinstance(sc_maps_fo, tuple) or isinstance(sc_maps_fo, list):
            pass
        else:
            sc_maps_fo = [sc_maps_fo]

    maxk_boxes = []
    maxk_maps = []
    for i in range(max(topk)):
        sc_map_cls = 0
        for j, sc_map in enumerate(sc_maps):
            # shape of sc_map:(1,196,196)
            cam_map_cls = top_cams[i]  # The size of CAM is the same as ori image.
            sc_map = sc_map.squeeze().data.cpu().numpy()  # (196,196)
            # shape of sc_map:(196,196)
            wh_sc = sc_map.shape[0]
            h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))  # 14,14
            cam_map_cls = cv2.resize(cam_map_cls, dsize=(w_sc, h_sc))
            cam_map_cls_vector = cam_map_cls.reshape(-1)  # (196,)
            # positive
            cam_map_cls_id = np.arange(wh_sc).astype(np.int)  # [0,1,2,...,195]
            cam_map_cls_th_ind_pos = cam_map_cls_id[cam_map_cls_vector >= fg_th]
            sc_map_sel_pos = sc_map[:, cam_map_cls_th_ind_pos]

            sc_map_sel_pos = (sc_map_sel_pos - np.min(sc_map_sel_pos, axis=0, keepdims=True)) / (
                    np.max(sc_map_sel_pos, axis=0, keepdims=True) - np.min(sc_map_sel_pos, axis=0,
                                                                           keepdims=True) + 1e-10)
            # shape of sc_map_sel_pos:(196,x), x is the selected point which >= fg_th.
            if sc_map_sel_pos.shape[1] > 0:
                sc_map_sel_pos = np.sum(sc_map_sel_pos, axis=1).reshape(h_sc, w_sc)  # (196,x) -> (196,) => (14,14)
                sc_map_sel_pos = (sc_map_sel_pos - np.min(sc_map_sel_pos)) / (
                        np.max(sc_map_sel_pos) - np.min(sc_map_sel_pos) + 1e-10)
                # Now, the shape of sc_map_sel_pos:(14,14)
            else:
                sc_map_sel_pos = 0
            # negtive
            cam_map_cls_th_ind_neg = cam_map_cls_id[cam_map_cls_vector <= bg_th]
            if sc_maps_fo is not None:
                sc_map_fo = sc_maps_fo[j]
                sc_map_fo = sc_map_fo.squeeze().data.cpu().numpy()
                sc_map_sel_neg = sc_map_fo[:, cam_map_cls_th_ind_neg]
            else:
                sc_map_sel_neg = sc_map[:, cam_map_cls_th_ind_neg]

            sc_map_sel_neg = (sc_map_sel_neg - np.min(sc_map_sel_neg, axis=0, keepdims=True)) / (
                    np.max(sc_map_sel_neg, axis=0, keepdims=True) - np.min(sc_map_sel_neg, axis=0,
                                                                           keepdims=True) + 1e-10)
            if sc_map_sel_neg.shape[1] > 0:
                sc_map_sel_neg = np.sum(sc_map_sel_neg, axis=1).reshape(h_sc, w_sc)
                sc_map_sel_neg = (sc_map_sel_neg - np.min(sc_map_sel_neg)) / (
                        np.max(sc_map_sel_neg) - np.min(sc_map_sel_neg) + 1e-10)
            else:
                sc_map_sel_neg = 0
            sc_map_cls_i = sc_map_sel_pos - sc_map_sel_neg
            sc_map_cls_i = sc_map_cls_i * (sc_map_cls_i >= 0)
            sc_map_cls_i = (sc_map_cls_i - np.min(sc_map_cls_i)) / (np.max(sc_map_cls_i) - np.min(sc_map_cls_i) + 1e-10)
            sc_map_cls_i = cv2.resize(sc_map_cls_i, dsize=(crop_size, crop_size))
            sc_map_cls = np.maximum(sc_map_cls, sc_map_cls_i)

        maxk_maps.append(sc_map_cls.copy())
        # segment the foreground
        fg_map = sc_map_cls >= threshold

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
            if gt_labels is not None:
                max_box = (int(gt_labels),) + max_box
            else:
                max_box = (cls_inds[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            if gt_labels is not None:
                maxk_boxes.append((int(gt_labels),) + box)
            else:
                maxk_boxes.append((cls_inds[i],) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]
    return result, maxk_maps


def get_masks(logits3, logits2, logits1, cam_map, parent_map, root_map, im_file, input_size, crop_size, topk=(1,),
              threshold=0.2, mode='union'):
    maxk = max(topk)
    species_cls = np.argsort(logits3)[::-1][:maxk]
    parent_cls = np.argsort(logits2)[::-1][:maxk]
    root_cls = np.argsort(logits1)[::-1][:maxk]

    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    maxk_maps = []
    for i in range(1):
        cam_map_ = cam_map[0, species_cls[i], :, :]
        parent_map_ = parent_map[0, parent_cls[i], :, :]
        root_map_ = root_map[0, root_cls[i], :, :]

        cam_map_cls = [cam_map_, parent_map_, root_map_]
        cam_map_ = (cam_map_ + parent_map_ + root_map_) / 3
        # cam_map_ = norm_atten_map(cam_map_)  # normalize cam map
        cam_map_cls.append(cam_map_)
        maxk_maps.append(np.array(cam_map_cls).copy())
    return maxk_maps


def extract_bbox_from_map(boolen_map):
    assert boolen_map.ndim == 2, 'Invalid input shape'
    rows = np.any(boolen_map, axis=1)
    cols = np.any(boolen_map, axis=0)
    if rows.max() == False or cols.max() == False:
        return 0, 0, 0, 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax
