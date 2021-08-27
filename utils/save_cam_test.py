from .vistools import norm_atten_map
import cv2
import numpy as np
import os
from torchvision import transforms
import torch


def save_cam(args, tras_img, save_cam, cls_logits, img_path, gt_label, epoch, threshold=0.3, suffix='cam'):
    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    # 将image转换为cv2
    tras_img = transforms.ToPILImage()(tras_img.data.cpu()).convert('RGB')
    trans_img = cv2.cvtColor(np.asarray(tras_img), cv2.COLOR_RGB2BGR)
    trans_img = cv2.resize(trans_img, dsize=(w, h))
    logits = cls_logits.data.cpu().numpy()
    idx_top5 = np.argsort(logits)[::-1][:5]
    cam_map = save_cam.data.cpu().numpy()
    maxk_maps = []
    for i in range(5):
        cam_map_ = cam_map[idx_top5[i], :, :]
        cam_map_ = norm_atten_map(cam_map_)
        cam_map_cls = cv2.resize(cam_map_, dsize=(w, h))
        maxk_maps.append(cam_map_cls.copy())

    draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)
    draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)
    draw_ts = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)

    draw_im[:h, :, :] = im
    cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]
    draw_ts[:h, :, :] = trans_img
    cv2.putText(draw_ts, 'transformed image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    cam_to_save = [draw_ts.copy()]
    for pred_idx, m in zip(idx_top5, maxk_maps):
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        cam_map_cls = cv2.resize(m, dsize=(w, h))
        ratio_color = 255
        gray_map = np.uint8(ratio_color * cam_map_cls)
        heatmap = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
        draw_im[:h, :, :] = heatmap * 0.7 + draw_im[:h, :, :] * 0.3
        draw_hm[:h, :, :] = heatmap

        if gt_label is not None:
            cls_str = 'CLS_TRUE' if int(pred_idx) == int(gt_label) else 'CLS_FALSE'
        else:
            cls_str = 'classified as {}'.format(pred_idx)
        cv2.putText(draw_im, cls_str, (0, h + 12), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        im_to_save.append(draw_im.copy())
        cam_to_save.append(draw_hm.copy())

    im_to_save = np.concatenate(im_to_save, axis=1)
    cam_to_save = np.concatenate(cam_to_save, axis=1)
    im_to_save = np.concatenate((im_to_save, cam_to_save), axis=0)
    end_suffix = suffix + '_per_iters'
    save_dir = os.path.join(args.log_dir, end_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(epoch) + '_' + img_path.split('/')[-1]
    save_name = save_name.replace('.', 'th-{}_{}.'.format(threshold, suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)
    pass


def save_sos(args, ts_img, scm, img_path, epoch, threshold=0.3, suffix=''):

    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    # 将image转换为cv2
    tras_img = transforms.ToPILImage()(ts_img.data.cpu()).convert('RGB')
    trans_img = cv2.cvtColor(np.asarray(tras_img), cv2.COLOR_RGB2BGR)
    trans_img = cv2.resize(trans_img, dsize=(w, h))

    draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)
    draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)
    draw_ts = 255 * np.ones((h + 15, w, 3), np.uint8)  # (h+15, w, 3)

    draw_im[:h, :, :] = im
    cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]

    draw_ts[:h, :, :] = trans_img
    cv2.putText(draw_ts, 'transformed image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    cam_to_save = [draw_ts.copy()]
    for m in scm:
        # if gt_label is not None:
        #     m = m[gt_label, :, :]
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        m = m.data.cpu().numpy()
        cam_map_cls = cv2.resize(m, dsize=(w, h))
        ratio_color = 255
        gray_map = np.uint8(ratio_color * cam_map_cls)
        heatmap = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
        draw_im[:h, :, :] = heatmap * 0.7 + draw_im[:h, :, :] * 0.3
        draw_hm[:h, :, :] = heatmap
        im_to_save.append(draw_im.copy())
        cam_to_save.append(draw_hm.copy())
    im_to_save = np.concatenate(im_to_save, axis=1)
    cam_to_save = np.concatenate(cam_to_save, axis=1)
    im_to_save = np.concatenate((im_to_save, cam_to_save), axis=0)
    end_suffix = suffix + '_per_iters'
    save_dir = os.path.join(args.log_dir, end_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(epoch) + '_' + img_path.split('/')[-1]
    save_name = save_name.replace('.', 'th-{}_{}.'.format(threshold, suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)
    pass


def save_scm(args, tras_img, aff_maps, img_path, epoch, suffix='scm45'):
    sc_maps = []
    if args.scg_com:
        for sc_map_fo_i, sc_map_so_i in zip(aff_maps[0], aff_maps[1]):
            if (sc_map_fo_i is not None) and (sc_map_so_i is not None):
                sc_map_i = torch.max(sc_map_fo_i, args.scg_so_weight * sc_map_so_i)
                sc_map_i = sc_map_i / (torch.sum(sc_map_i, dim=1, keepdim=True) + 1e-10)
                sc_maps.append(sc_map_i)
    aff_maps = sc_maps[-2] + sc_maps[-1]

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

    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    # 将image转换为cv2
    tras_img = transforms.ToPILImage()(tras_img.data.cpu()).convert('RGB')
    trans_img = cv2.cvtColor(np.asarray(tras_img), cv2.COLOR_RGB2BGR)
    trans_img = cv2.resize(trans_img, dsize=(w, h))

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
        draw_im[:h, :, :] = trans_img
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

    end_suffix = suffix + '_per_iters'
    save_dir = os.path.join(args.log_dir, end_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(epoch) + '_' + img_path.split('/')[-1]
    save_name = save_name.replace('.', '_{}.'.format(suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)