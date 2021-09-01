import numpy as np
import cv2
import torch
from scipy.ndimage import label
from .vistools import norm_atten_map
import torch.nn.functional as F


def get_topk_boxes(logits, cam_map, im_file, input_size, crop_size, topk=(1,), threshold=0.2, mode='union', gt=None):
    maxk = max(topk)
    maxk_cls = np.argsort(logits)[::-1][:maxk]

    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    maxk_boxes = []
    maxk_maps = []
    for cls in maxk_cls:
        if gt:
            cls = gt
        cam_map_ = cam_map[0, cls, :, :]
        cam_map_ = norm_atten_map(cam_map_)  # normalize cam map
        cam_map_cls = cv2.resize(cam_map_, dsize=(w, h))
        maxk_maps.append(cam_map_cls.copy())

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
            # maxk_boxes.append((cls, int(box[0] / scale), int(box[1] / scale), int(box[2] / scale), int(box[3] / scale)))
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]

    return result, maxk_maps


def get_topk_boxes_hier(logits, cam_map, im_file, gt_label, topk=(1,), threshold=0.2, mode='union'):
    """
    Get the top-k pred bbox
    :param logits: pred_cls_res with shape 200, the value is probability.
    :param cam_map: pred cls map with shape (n,c,w,h)
    :param im_file: input img
    :param gt_label: cls label
    :param topk: (1,5)
    :param threshold: threshold for seg fg
    :param mode: default = 'union'
    :return: topk_idx_bbox, topk_cam, gt_known_box, gt_known_cam
    """
    logits = logits.data.cpu().numpy()  # 200
    cam_map = cam_map.data.cpu().numpy()  # n,c,w,h
    maxk = max(topk)  # 5
    species_cls = np.argsort(logits)[::-1][:maxk]  # 按值从小到大获取idx，逆序，取top5; logit[species_cls[0]] 即为top-1得分
    # species_cls 存的是idx
    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    maxk_boxes = []
    maxk_maps = []
    for i in range(maxk):
        cam_map_ = cam_map[0, species_cls[i], :, :]  # (14,14) top-1 cam map
        cam_map_ = norm_atten_map(cam_map_)  # normalize cam map  # (14,14)
        cam_map_cls = cv2.resize(cam_map_, dsize=(w, h))  # (14,14) => (224,224)
        maxk_maps.append(cam_map_cls.copy())
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
            max_box = (species_cls[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            maxk_boxes.append((species_cls[i],) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]  # [[(idx_1,box_1)],[(idx_1,box_1),(idx_2,box_2),...,(idx_5,box_5)]]
    # gt_known
    gt_known_boxes = []
    gt_known_maps = []
    cam_map_ = cam_map[0, int(gt_label[0]), :, :]
    cam_map_ = norm_atten_map(cam_map_)  # normalize cam map
    cam_map_gt_known = cv2.resize(cam_map_, dsize=(w, h))
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
        max_box = (int(gt_label[0]),) + max_box
        gt_known_boxes.append(max_box)
    elif mode == 'union':
        box = extract_bbox_from_map(fg_map)
        gt_known_boxes.append((int(gt_label[0]),) + box)  # (label, xmin,ymin,xmax,ymax)
    else:
        raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    return result, maxk_maps, gt_known_boxes, gt_known_maps


def get_box_sos(pred_scm, im_file, threshold, gt_labels):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    pred_scm = pred_scm.squeeze().data.cpu().numpy()
    sc_map = cv2.resize(pred_scm, dsize=(w, h))  # (14,14) => ori size
    sc_map_cls = np.maximum(0, sc_map)
    maxk_maps = [sc_map_cls.copy()]
    fg_map = sc_map_cls >= threshold
    box = extract_bbox_from_map(fg_map)
    maxk_boxes = [(int(gt_labels[0]),) + box]  # [(label, xmin, ymin, xmax, ymax)]
    result = [maxk_boxes[:k] for k in (1,)]  # [[(label, xmin, ymin, xmax, ymax)]]
    return result, maxk_maps


def get_topk_boxes_mc_sos(logits, mc_sos, im_file, topk=(1, 5), gt_labels=None, threshold=0.2, mode='union'):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    maxk = max(topk)
    _logits = logits.data.cpu().numpy()  # 200
    species_cls = np.argsort(_logits)[::-1][:5]
    maxk_boxes = []
    maxk_maps = []
    for i in range(maxk):
        top_i_sos = mc_sos[species_cls[i]]  # (c,w,h) => (w,h)
        sos_map_top_i = torch.sigmoid(top_i_sos)
        sos_map_top_i = sos_map_top_i.squeeze().data.cpu().numpy()
        sos_map_top_i = cv2.resize(sos_map_top_i, dsize=(w, h))
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
                max_box = (species_cls[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            if gt_labels is not None:
                maxk_boxes.append((int(gt_labels[0]),) + box)
            else:
                maxk_boxes.append((species_cls[i],) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]
    return result, maxk_maps


def get_topk_boxes_scg_v2(logits, top_cams, sc_maps, im_file, topk=(1,), gt_labels=None, threshold=0.2,
                          mode='union', sc_maps_fo=None):
    logits = logits.data.cpu().numpy()
    maxk = max(topk)
    species_cls = np.argsort(logits)[::-1][:maxk]  # top_k's idx
    if isinstance(sc_maps, tuple) or isinstance(sc_maps, list):
        pass
    else:
        sc_maps = [sc_maps]
    if sc_maps_fo is not None:
        if isinstance(sc_maps_fo, tuple) or isinstance(sc_maps_fo, list):
            pass
        else:
            sc_maps_fo = [sc_maps_fo]
    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    maxk_boxes = []
    maxk_maps = []
    for i in range(maxk):
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
            sc_map_cls_i = (sc_map_cls_i - np.min(sc_map_cls_i)) / (np.max(sc_map_cls_i) - np.min(sc_map_cls_i) + 1e-10)
            sc_map_cls_i = cv2.resize(sc_map_cls_i, dsize=(w, h))  # (14,14) => ori image size
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
                max_box = (int(gt_labels[0]),) + max_box
            else:
                max_box = (species_cls[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            if gt_labels is not None:
                maxk_boxes.append((int(gt_labels[0]),) + box)
            else:
                maxk_boxes.append((species_cls[i],) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]
    return result, maxk_maps
    pass


def get_topk_boxes_hier_scg(logits, top_cams, sc_maps, im_file, topk=(1,), gt_labels=None, threshold=0.2,
                            mode='union', fg_th=0.1, bg_th=0.05, sc_maps_fo=None):
    logits = logits.data.cpu().numpy()  # 200
    maxk = max(topk)  # 5
    species_cls = np.argsort(logits)[::-1][:maxk]  # top_k's idx
    if isinstance(sc_maps, tuple) or isinstance(sc_maps, list):
        pass
    else:
        sc_maps = [sc_maps]
    if sc_maps_fo is not None:
        if isinstance(sc_maps_fo, tuple) or isinstance(sc_maps_fo, list):
            pass
        else:
            sc_maps_fo = [sc_maps_fo]
    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    maxk_boxes = []
    maxk_maps = []
    for i in range(maxk):
        sc_map_cls = 0
        for j, sc_map in enumerate(sc_maps):
            # shape of sc_map:(1,196,196)
            cam_map_cls = top_cams[i]  # The size of CAM is the same as ori image.
            sc_map = sc_map.squeeze().data.cpu().numpy()   # (196,196)
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
            sc_map_cls_i = cv2.resize(sc_map_cls_i, dsize=(w, h))  # (14,14) => ori image size
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
                max_box = (int(gt_labels[0]),) + max_box
            else:
                max_box = (species_cls[i],) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            if gt_labels is not None:
                maxk_boxes.append((int(gt_labels[0]),) + box)
            else:
                maxk_boxes.append((species_cls[i],) + box)
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
