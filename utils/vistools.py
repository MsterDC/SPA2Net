import numpy as np
import cv2
import os
import torch


def debug_vis_loc(args, idx, show_idxs, img_path, top_maps, top5_boxes, label, gt_boxes, detail, suffix='cam'):
    if not args.debug:
        return
    cls_wrong = detail.get('cls_wrong')
    multi_instances = detail.get('multi_instances')
    region_part = detail.get('region_part')
    region_more = detail.get('region_more')
    region_wrong = detail.get('region_wrong')

    base_num = (args.debug_num // args.batch_size) if (args.debug_num % args.batch_size) == 0 \
        else  (args.debug_num // args.batch_size) + 1
    for vis_n in range(base_num):
        if idx == vis_n:
            top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_scg'.format(cls_wrong,
                                                                                            multi_instances,
                                                                                            region_part,
                                                                                            region_more,
                                                                                            region_wrong)
            debug_dir = os.path.join(args.debug_dir, top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
            for show_id in show_idxs:
                save_im_heatmap_box(img_path[show_id], top_maps[show_id], top5_boxes[show_id], debug_dir,
                                    gt_label=label[show_id], gt_box=gt_boxes[show_id],
                                    epoch=args.current_epoch, threshold=args.threshold[0], suffix=suffix)
                pass
        pass


def debug_vis_sc(args, idx, show_idxs, img_path, sc_maps_fo_fuse, sc_maps_so_fuse, sc_maps_fuse, label, detail, suffix=None):
    if not args.debug:
        return
    cls_wrong = detail.get('cls_wrong')
    multi_instances = detail.get('multi_instances')
    region_part = detail.get('region_part')
    region_more = detail.get('region_more')
    region_wrong = detail.get('region_wrong')

    base_num = (args.debug_num // args.batch_size) if (args.debug_num % args.batch_size) == 0 \
        else (args.debug_num // args.batch_size) + 1
    for vis_n in range(base_num):
        if idx == vis_n:
            top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_scg'.format(cls_wrong,
                                                                                            multi_instances,
                                                                                            region_part,
                                                                                            region_more,
                                                                                            region_wrong)
            debug_dir = os.path.join(args.debug_dir, top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
            for show_id in show_idxs:
                save_im_sim(img_path[show_id], sc_maps_fo_fuse[show_id], debug_dir, gt_label=label[show_id], epoch=args.current_epoch, suffix=suffix[0])
                save_im_sim(img_path[show_id], sc_maps_so_fuse[show_id], debug_dir, gt_label=label[show_id], epoch=args.current_epoch, suffix=suffix[1])
                save_im_sim(img_path[show_id], sc_maps_fuse[show_id], debug_dir, gt_label=label[show_id], epoch=args.current_epoch, suffix=suffix[-1])
        pass


def save_im_heatmap_box(im_file, top_maps, topk_boxes, save_dir, gt_label=None, gt_box=None,
                        epoch=100, threshold=-1, suffix='cam'):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)
    draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)
    cam_to_save = [draw_hm.copy()]
    draw_im[:h, :, :] = im

    if gt_box is not None:
        gt_box = gt_box.split()
        box_cnt = len(gt_box) // 4
        gt_box = np.asarray(gt_box, int)
        loc_flag = False
        for i in range(box_cnt):
            gt_bbox = gt_box[i * 4:(i + 1) * 4]
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = gt_bbox
            ori_left_top_x = int(left_top_x * w / 224)  # [Attention] 224 is the cropped size
            ori_left_top_y = int(left_top_y * h / 224)
            ori_right_bottom_x = int(right_bottom_x * w / 224)
            ori_right_bottom_y = int(right_bottom_y * h / 224)
            cv2.rectangle(draw_im, (ori_left_top_x, ori_left_top_y),
                          (ori_right_bottom_x, ori_right_bottom_y),
                          color=(0, 0, 255), thickness=2)

    cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    im_to_save = [draw_im.copy()]
    for cls_box, cam_map_cls in zip(topk_boxes, top_maps):
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        cam_map_cls = cv2.resize(cam_map_cls, dsize=(w, h))
        ratio_color = 255
        gray_map = np.uint8(ratio_color * cam_map_cls)
        heatmap = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)

        draw_im[:h, :, :] = heatmap * 0.7 + draw_im[:h, :, :] * 0.3
        draw_hm[:h, :, :] = heatmap

        if gt_box is not None:
            box_cnt = len(gt_box) // 4
            gt_box = np.asarray(gt_box, int)
            loc_flag = False
            for i in range(box_cnt):
                gt_bbox = gt_box[i * 4:(i + 1) * 4]
                loc_flag = loc_flag or (cal_iou(cls_box[1:], gt_bbox) > 0.5)
            loc_str = 'LOC_TRUE' if loc_flag else 'LOC_FALSE'
        if gt_label is not None:
            cls_str = 'CLS_TRUE' if int(cls_box[0]) == int(gt_label) else 'CLS_FALSE'
        else:
            cls_str = 'classified as {}'.format(cls_box[0])
        cv2.putText(draw_im, cls_str + '|{}'.format(loc_str), (0, h + 12), color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        im_to_save.append(draw_im.copy())
        cam_to_save.append(draw_hm.copy())
    im_to_save = np.concatenate(im_to_save, axis=1)
    cam_to_save = np.concatenate(cam_to_save, axis=1)
    im_to_save = np.concatenate((im_to_save, cam_to_save), axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save_name = str(gt_label[0]) + '_' + str(epoch) + '_' + im_file.split('/')[-1]
    save_name = str(gt_label) + '_' + str(epoch) + '_' + im_file.split('/')[-1]
    save_name = save_name.replace('.', 'th-{}_{}.'.format(threshold, suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)


def save_sim_heatmap_box(im_file, top_maps, save_dir, gt_label=None, sim_map=None,
                         epoch=100, threshold=-1, suffix='', fg_th=0.1, bg_th=0.05):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    final_to_save = []
    if isinstance(sim_map, tuple) or isinstance(sim_map, list):
        pass
    else:
        sim_map = [sim_map]
    for sim_map_i in sim_map:
        if sim_map_i is None:
            continue
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5)
        sim_map_i = sim_map_i.squeeze().data.cpu().numpy()
        cam_to_save = [draw_hm.copy()]
        cam_pos_to_save = [draw_hm.copy()]
        cam_neg_to_save = [draw_hm.copy()]
        im_to_save = [draw_im.copy()]
        wh_sim = sim_map_i.shape[0]
        h_sim, w_sim = int(np.sqrt(wh_sim)), int(np.sqrt(wh_sim))
        for cam_map_cls in top_maps:
            draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
            draw_im[:h, :, :] = im
            cam_map_cls = cv2.resize(cam_map_cls, dsize=(w_sim, h_sim))
            cam_map_cls_vector = cam_map_cls.reshape(-1)
            # positive
            cam_map_cls_id = np.arange(wh_sim).astype(np.int)
            cam_map_cls_th_ind_pos = cam_map_cls_id[cam_map_cls_vector > fg_th]
            sim_map_sel_pos = sim_map_i[:, cam_map_cls_th_ind_pos]
            sim_map_sel_pos = (sim_map_sel_pos - np.min(sim_map_sel_pos, axis=0, keepdims=True)) / (
                    np.max(sim_map_sel_pos, axis=0, keepdims=True) - np.min(sim_map_sel_pos, axis=0,
                                                                            keepdims=True) + 1e-10)
            cam_map_cls_val_pos = cam_map_cls_vector[cam_map_cls_th_ind_pos].reshape(1, -1)
            # sim_map_sel_pos = np.sum(sim_map_sel_pos * cam_map_cls_val_pos, axis=1).reshape(h_sim, w_sim)
            sim_map_sel_pos = np.sum(sim_map_sel_pos, axis=1).reshape(h_sim, w_sim)
            sim_map_sel_pos = (sim_map_sel_pos - np.min(sim_map_sel_pos)) / (
                    np.max(sim_map_sel_pos) - np.min(sim_map_sel_pos) + 1e-10)

            # negtive
            cam_map_cls_th_ind_neg = cam_map_cls_id[cam_map_cls_vector < bg_th]
            sim_map_sel_neg = sim_map_i[:, cam_map_cls_th_ind_neg]
            sim_map_sel_neg = (sim_map_sel_neg - np.min(sim_map_sel_neg, axis=0, keepdims=True)) / (
                    np.max(sim_map_sel_neg, axis=0, keepdims=True) - np.min(sim_map_sel_neg, axis=0,
                                                                            keepdims=True) + 1e-10)
            cam_map_cls_val_neg = cam_map_cls_vector[cam_map_cls_th_ind_neg].reshape(1, -1)
            # sim_map_sel_neg = np.sum(sim_map_sel_neg * (1-cam_map_cls_val_neg), axis=1).reshape(h_sim, w_sim)
            sim_map_sel_neg = np.sum(sim_map_sel_neg, axis=1).reshape(h_sim, w_sim)
            sim_map_sel_neg = (sim_map_sel_neg - np.min(sim_map_sel_neg)) / (
                    np.max(sim_map_sel_neg) - np.min(sim_map_sel_neg) + 1e-10)

            #
            sim_map_sel = sim_map_sel_pos - sim_map_sel_neg
            # sim_map_sel = sim_map_sel_pos
            sim_map_sel = sim_map_sel * (sim_map_sel > 0)
            sim_map_sel = (sim_map_sel - np.min(sim_map_sel)) / (np.max(sim_map_sel) - np.min(sim_map_sel) + 1e-10)
            sim_map_sel = cv2.resize(sim_map_sel, dsize=(w, h))
            sim_map_sel_pos = cv2.resize(sim_map_sel_pos, dsize=(w, h))
            sim_map_sel_neg = cv2.resize(sim_map_sel_neg, dsize=(w, h))
            heatmap = cv2.applyColorMap(np.uint8(255 * sim_map_sel), cv2.COLORMAP_JET)
            heatmap_pos = cv2.applyColorMap(np.uint8(255 * sim_map_sel_pos), cv2.COLORMAP_JET)
            heatmap_neg = cv2.applyColorMap(np.uint8(255 * sim_map_sel_neg), cv2.COLORMAP_JET)
            draw_im[:h, :, :] = heatmap * 0.7 + draw_im[:h, :, :] * 0.3
            draw_hm[:h, :, :] = heatmap

            im_to_save.append(draw_im.copy())
            cam_to_save.append(draw_hm.copy())
            draw_hm[:h, :, :] = heatmap_pos
            cam_pos_to_save.append(draw_hm.copy())
            draw_hm[:h, :, :] = heatmap_neg
            cam_neg_to_save.append(draw_hm.copy())
        im_to_save = np.concatenate(im_to_save, axis=1)
        cam_to_save = np.concatenate(cam_to_save, axis=1)
        cam_pos_to_save = np.concatenate(cam_pos_to_save, axis=1)
        cam_neg_to_save = np.concatenate(cam_neg_to_save, axis=1)
        final_to_save.append(im_to_save)
        final_to_save.append(cam_to_save)
        final_to_save.append(cam_pos_to_save)
        final_to_save.append(cam_neg_to_save)
    final_to_save = np.concatenate(final_to_save, axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(gt_label[0]) + '_' + str(epoch) + '_' + im_file.split('/')[-1]
    save_name = save_name.replace('.', 'th-{}_{}.'.format(threshold, suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), final_to_save)


def save_im_sim(im_file, aff_maps, save_dir, suffix='', gt_label=None, epoch=100):
    if isinstance(aff_maps, tuple) or isinstance(aff_maps, list):
        pass
    else:
        aff_maps = [aff_maps]
    break_flag = True
    for aff_i in aff_maps:
        if aff_i is not None:
            break_flag = False
            break
    if break_flag:
        return

    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    draw_aff = []
    draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
    for i in range(len(aff_maps)):
        if aff_maps[i] is not None:
            draw_aff.append(255 * np.ones((h + 15, w, 3), np.uint8))
    draw_aff = np.concatenate(draw_aff, axis=0)
    aff_to_save = [draw_aff.copy()]
    draw_im[:h, :, :] = im
    cv2.putText(draw_im, 'original image', (0, h + 12), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]
    points = [[h // 4, w // 4], [h // 4 * 3, w // 4 * 3], [h // 2, w // 2], [h // 4 * 3, w // 4], [h // 4, w // 4 * 3]]
    for point in points:
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        draw_aff_j = []
        for i, aff_i in enumerate(aff_maps):
            if aff_i is None:
                continue
            aff_i = aff_i.squeeze().data.cpu().numpy()
            draw_aff_ij = 255 * np.ones((h + 15, w, 3), np.uint8)
            h_w_aff = aff_i.shape[0]
            h_aff, w_aff = int(np.sqrt(h_w_aff)), int(np.sqrt(h_w_aff))
            h_aff_i, w_aff_i = int(point[0] * h_aff / h), int(point[1] * w_aff / w)
            aff_map_i = aff_i[:, h_aff_i * w_aff + w_aff_i].reshape(h_aff, w_aff)
            aff_map_i = (aff_map_i - np.min(aff_map_i)) / (np.max(aff_map_i) - np.min(aff_map_i) + 1e-10)
            aff_map_i = cv2.resize(aff_map_i, dsize=(w, h))
            aff_map_i = cv2.applyColorMap(np.uint8(255 * aff_map_i), cv2.COLORMAP_JET)
            ptStart_h = (point[1] - 5, point[0])
            ptEnd_h = (point[1] + 5, point[0])
            point_color = (0, 255, 0)  # BGR
            thickness = 2
            lineType = 4
            cv2.line(draw_im, ptStart_h, ptEnd_h, point_color, thickness, lineType)

            ptStart_v = (point[1], point[0] - 5)
            ptEnd_v = (point[1], point[0] + 5)
            cv2.line(draw_im, ptStart_v, ptEnd_v, point_color, thickness, lineType)

            draw_aff_ij[:h, :, :] = aff_map_i
            cv2.putText(draw_aff_ij, '{} layer'.format(i + 2), (0, h + 12), color=(0, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            draw_aff_j.append(draw_aff_ij)
        draw_aff_j = np.concatenate(draw_aff_j, axis=0)
        im_to_save.append(draw_im.copy())
        aff_to_save.append(draw_aff_j.copy())

    im_to_save = np.concatenate(im_to_save, axis=1)
    aff_to_save = np.concatenate(aff_to_save, axis=1)
    im_to_save = np.concatenate((im_to_save, aff_to_save), axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(gt_label) + '_' + str(epoch) + '_' + im_file.split('/')[-1]
    save_name = save_name.replace('.', '_sim_{}.'.format(suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)


def cal_iou(box1, box2):
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    iou_val = i_area / (box1_area + box2_area - i_area)

    return iou_val


def vis_feature(feat, img_path, vis_path, col=4, row=4, layer='feat3'):
    ## normalize feature
    feat = feat[0, ...]
    c, fh, fw = feat.size()
    feat = feat.view(c, -1)
    min_val, _ = torch.min(feat, dim=-1, keepdim=True)
    max_val, _ = torch.max(feat, dim=-1, keepdim=True)
    norm_feat = (feat - min_val) / (max_val - min_val + 1e-10)
    norm_feat = norm_feat.view(c, fh, fw).contiguous().permute(1, 2, 0)
    norm_feat = norm_feat.data.cpu().numpy()

    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    resized_feat = cv2.resize(norm_feat, (w, h))

    # draw images
    feat_ind = 0
    fig_id = 0

    while feat_ind < 20:
        im_to_save = []
        for i in range(row):
            draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
            draw_im[:h, :w, :] = im
            cv2.putText(draw_im, 'original image', (0, h + 12), color=(0, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5)
            im_to_save_row = [draw_im.copy()]
            for j in range(col):
                draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
                draw_im[:h, :w, :] = im

                heatmap = cv2.applyColorMap(np.uint8(255 * resized_feat[:, :, feat_ind]), cv2.COLORMAP_JET)
                draw_im[:h, :w, :] = heatmap * 1. + draw_im[:h, :w, :] * 0.0

                im_to_save_row.append(draw_im.copy())
                feat_ind += 1
            im_to_save_row = np.concatenate(im_to_save_row, axis=1)
            im_to_save.append(im_to_save_row)
        im_to_save = np.concatenate(im_to_save, axis=0)
        vis_path = os.path.join(vis_path, 'vis_feat')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        save_name = 'vgg_' + img_path.split('/')[-1]
        save_name = save_name.replace('.', '_{}_{}.'.format(layer, fig_id))
        cv2.imwrite(os.path.join(vis_path, save_name), im_to_save)
        fig_id += 1


def vis_var(feat, cls_logits, img_path, vis_path, net='vgg_fpn_l3'):
    cls_logits = cls_logits.squeeze()

    norm_var_no_white = norm_tensor(feat)
    norm_var_no_white = 1 - norm_var_no_white
    norm_var_no_white[norm_var_no_white < 0.05] = 0
    norm_var_no_white = norm_atten_map(norm_var_no_white)
    # norm_var_no_white = (norm_var_no_white < 0.4).astype(norm_var_no_white.dtype)
    norm_cls_no_white = norm_tensor(cls_logits)
    norm_cls_no_white[norm_cls_no_white < 0.2] = 0
    norm_cls_no_white = norm_atten_map(norm_cls_no_white)

    # norm_cls_no_white = (norm_cls_no_white>0.7).astype(norm_cls_no_white.dtype)

    white_feat = whitening_tensor(feat)
    white_cls_logits = whitening_tensor(cls_logits)
    norm_var = norm_tensor(white_feat)
    norm_var[norm_var < 0.8] = 0
    norm_var = norm_atten_map(norm_var)
    # norm_var = (norm_var<0.4).astype(norm_var.dtype)
    norm_cls = norm_tensor(white_cls_logits)
    norm_cls = 1 - norm_cls
    norm_cls[norm_cls < 0.8] = 0
    norm_cls = norm_atten_map(norm_cls)
    # norm_cls = (norm_cls>0.2).astype(norm_cls.dtype)

    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    resized_var_no_white = cv2.resize(norm_var_no_white, (w, h))
    resized_cls_no_white = cv2.resize(norm_cls_no_white, (w, h))
    resized_var = cv2.resize(norm_var, (w, h))
    resized_cls = cv2.resize(norm_cls, (w, h))

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    cv2.putText(draw_im, 'original image', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var_no_white), cv2.COLORMAP_BONE)
    draw_im[:h, :w, :] = heatmap * 1.0 + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'var_nw', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cls_no_white), cv2.COLORMAP_BONE)
    draw_im[:h, :w, :] = heatmap * 1. + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'cls_nw', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    resized_var_cls_no_white = (resized_var_no_white + resized_cls_no_white) * 0.5
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var_cls_no_white), cv2.COLORMAP_JET)
    draw_im[:h, :w, :] = heatmap * 0.5 + draw_im[:h, :w, :] * 0.5
    cv2.putText(draw_im, 'var_cls_nw', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var), cv2.COLORMAP_JET)
    draw_im[:h, :w, :] = heatmap * 1. + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'var', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cls), cv2.COLORMAP_BONE)
    draw_im[:h, :w, :] = heatmap * 1 + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'cls', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    resized_var_cls = (resized_var + resized_cls) * 0.5
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var_cls), cv2.COLORMAP_JET)
    draw_im[:h, :w, :] = heatmap * 0.5 + draw_im[:h, :w, :] * 0.5
    cv2.putText(draw_im, 'var_cls', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    im_to_save = np.concatenate(im_to_save, axis=1)

    vis_path = os.path.join(vis_path, 'vis_var/{}'.format(net))
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    save_name = 'vgg_' + img_path.split('/')[-1]
    cv2.imwrite(os.path.join(vis_path, save_name), im_to_save)


def norm_tensor(feat):
    min_val = torch.min(feat)
    max_val = torch.max(feat)
    norm_feat = (feat - min_val) / (max_val - min_val + 1e-20)
    norm_feat = norm_feat.data.cpu().numpy()
    return norm_feat


def whitening_tensor(feat):
    mean = torch.mean(feat)
    var = torch.std(feat)
    norm_feat = (feat - mean) / (var + 1e-15)
    return norm_feat


def norm_atten_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    atten_norm = (attention_map - min_val) / (max_val - min_val + 1e-10)
    return atten_norm


def norm_for_batch_map(scm):
    b, w, h = scm.shape
    attention_map = scm.reshape(b, -1)  # (n，w*h)
    min_val = np.min(attention_map, axis=1)  # (n,)
    min_val = np.expand_dims(np.expand_dims(min_val, axis=1), axis=2)  # (n,1,1)
    max_val = np.max(attention_map, axis=1)  # (n,)
    max_val = np.expand_dims(np.expand_dims(max_val, axis=1), axis=2)  # (n,1,1)
    atten_norm = (scm - min_val) / (max_val - min_val + 1e-10)  # (n,w,h)
    return atten_norm

class NormalizationFamily:

    def __call__(self, norm_fun, *args):
        norm_fun = 'self.' + norm_fun
        return eval(norm_fun)(*args)

    @staticmethod
    def norm_min_max(*args):
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        min_val = np.min(fm)
        max_val = np.max(fm)
        normed_fm = (fm - min_val) / (max_val - min_val + 1e-10)
        if isinstance(args[0], torch.Tensor):
            normed_fm = torch.from_numpy(normed_fm)
        return normed_fm

    @staticmethod
    def norm_min_max_batch(*args):
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        b, w, h = fm.shape
        flatten_map = fm.reshape(b, -1)  # (n，w*h)
        min_val = np.min(flatten_map, axis=1)  # (n,)
        min_val = np.expand_dims(np.expand_dims(min_val, axis=1), axis=2)  # (n,1,1)
        max_val = np.max(flatten_map, axis=1)  # (n,)
        max_val = np.expand_dims(np.expand_dims(max_val, axis=1), axis=2)  # (n,1,1)
        normed_map = (fm - min_val) / (max_val - min_val + 1e-10)  # (n,w,h)
        if isinstance(args[0], torch.Tensor):
            normed_map = torch.from_numpy(normed_map)
        return normed_map

    @staticmethod
    def norm_max(*args):
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        max_val = np.max(fm)
        norm_map = fm / max_val
        if isinstance(args[0], torch.Tensor):
            norm_map = torch.from_numpy(norm_map)
        return norm_map

    @staticmethod
    def norm_max_batch(*args):
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        b, w, h = fm.shape
        flatten_map = fm.reshape(b, -1)
        max_val = np.max(flatten_map, axis=1)
        max_val = np.expand_dims(np.expand_dims(max_val, axis=1), axis=2)
        normed_map = fm / max_val
        if isinstance(args[0], torch.Tensor):
            normed_map = torch.from_numpy(normed_map)
        return normed_map

    @staticmethod
    def norm_pas(*args):
        """ @ author: Kevin
        Imp of 'Rethinking class activation mapping for weakly supervised object localization', ECCV 2020
        """
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        min_val = np.min(fm)
        d_F_min = fm - min_val
        p_fm = np.percentile(d_F_min, q=args[1])
        normed_fm = d_F_min / p_fm
        if isinstance(args[0], torch.Tensor):
            normed_fm = torch.from_numpy(normed_fm)
        return normed_fm

    @staticmethod
    def norm_pas_batch(*args):
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        b, w, h = fm.shape
        flatten_map = fm.reshape(b, -1)
        min_val = np.min(flatten_map, axis=1)  # (n,)
        min_val = np.expand_dims(np.expand_dims(min_val, axis=1), axis=2)  # (n,1,1)
        d_F_min = fm - min_val
        p_df = d_F_min.reshape(b, -1)  # (n,wh)
        p_fm = np.expand_dims(np.percentile(p_df, q=args[1], axis=1), axis=1)  # (n,1)
        normed_map = p_df / p_fm  # (n, wh)
        normed_map = normed_map.reshape(b, w, h)
        if isinstance(args[0], torch.Tensor):
            normed_map = torch.from_numpy(normed_map)
        return normed_map

    @staticmethod
    def norm_ivr(*args):
        """ @ author: Kevin
        Imp of 'Normalization Matters in Weakly Supervised Object Localization', ICCV 2021
        """
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        p_fm = np.percentile(fm, q=100-args[1])
        d_F_p = fm - p_fm
        normed_fm = d_F_p / np.max(d_F_p)
        if isinstance(args[0], torch.Tensor):
            normed_fm = torch.from_numpy(normed_fm)
        return normed_fm

    @staticmethod
    def norm_ivr_batch(*args):
        fm = None
        if isinstance(args[0], torch.Tensor):
            fm = args[0].data.cpu().numpy()
        if isinstance(args[0], np.ndarray):
            fm = args[0]
        b, w, h = fm.shape
        flatten_map = fm.reshape(b, -1)
        p_fm = np.expand_dims(np.expand_dims(np.percentile(flatten_map, q=100-args[1], axis=1), axis=1), axis=2)
        d_F_p = fm - p_fm  # (n,w,h)
        max_val = np.expand_dims(np.expand_dims(np.max(d_F_p.reshape(b, -1), axis=1), axis=1), axis=2)
        normed_fm = d_F_p / max_val
        if isinstance(args[0], torch.Tensor):
            normed_fm = torch.from_numpy(normed_fm)
        return normed_fm