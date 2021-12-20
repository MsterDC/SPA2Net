import numpy as np
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from utils.vistools import norm_for_batch_map
from .sa import ScaledDotProductAttention

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'model'
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, cnvs=(10, 17, 24), args=None):
        super(VGG, self).__init__()
        self.conv1_2 = nn.Sequential(*features[:cnvs[0]])
        self.conv3 = nn.Sequential(*features[cnvs[0]:cnvs[1]])
        self.conv4 = nn.Sequential(*features[cnvs[1]:cnvs[2]])
        self.conv5 = nn.Sequential(*features[cnvs[2]:-1])
        self.num_classes = num_classes
        # self.normalize = NormalizationFamily()
        self.args = args

        self.cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0))

        if 'sa' in self.args.mode:
            self.sa = ScaledDotProductAttention(d_model=int(args.sa_neu_num), d_k=int(args.sa_neu_num),
                                                d_v=int(args.sa_neu_num), h=int(args.sa_head), weight=args.sa_edge_weight)

        if 'sos' in self.args.mode:
            self.sos = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, 1, kernel_size=1, padding=0))

        self._initialize_weights()

        # loss function
        self.ce_loss = F.cross_entropy
        self.mse_loss = F.mse_loss
        self.bce_loss = F.binary_cross_entropy_with_logits  # with sigmoid function

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_masked_pseudo_gt(self, gt_scm, fg_th, bg_th, method='TC'):
        # for BC: convert scm to [0,1] binary mask.
        if method == 'BC':
            mask_hm_bg = torch.zeros_like(gt_scm)
            mask_hm_fg = torch.ones_like(gt_scm)
            gt_scm = torch.where(gt_scm >= fg_th, mask_hm_fg, mask_hm_bg)
        # for TC: convert scm to [0,value,1] mixed mask.
        elif method == 'TC':
            mask_hm_zero = torch.zeros_like(gt_scm)
            mask_hm_one = torch.ones_like(gt_scm)
            gt_scm = torch.where(gt_scm >= fg_th, mask_hm_one, gt_scm)
            gt_scm = torch.where(gt_scm <= bg_th, mask_hm_zero, gt_scm)
        return gt_scm

    def get_scm(self, logits, gt_label, sc_maps_fo, sc_maps_so):
        # get cam
        loc_map = F.relu(logits)
        cam_map = loc_map.data.cpu().numpy()
        # gt_label: (n, )
        cam_map_ = cam_map[torch.arange(cam_map.shape[0]), gt_label.data.cpu().numpy().astype(int), :, :]  # (bs, w, h)
        cam_map_cls = norm_for_batch_map(cam_map_)  # (64,14,14)

        # using fo/so and diff stage feature to get fused scm.
        sc_maps = []
        if self.args.scg_com:
            for sc_map_fo_i, sc_map_so_i in zip(sc_maps_fo, sc_maps_so):
                if (sc_map_fo_i is not None) and (sc_map_so_i is not None):
                    sc_map_so_i = sc_map_so_i.to(self.args.device)
                    sc_map_i = torch.max(sc_map_fo_i, self.args.scg_so_weight * sc_map_so_i)
                    sc_map_i = sc_map_i / (torch.sum(sc_map_i, dim=1, keepdim=True) + 1e-10)
                    sc_maps.append(sc_map_i)

        if self.args.scg_blocks == '4,5':
            sc_com = sc_maps[-2] + sc_maps[-1]
        elif self.args.scg_blocks == '4':
            sc_com = sc_maps[-2]
        elif self.args.scg_blocks == '5':
            sc_com = sc_maps[-1]
        else:
            raise Exception("[Error] HSC must be calculated by 4 or 5 stage feature of backbone.")

        # weighted sum for scm and cam
        sc_map = sc_com.squeeze().data.cpu().numpy()  # (64,196,196)
        wh_sc, bz = sc_map.shape[1], sc_map.shape[0]
        h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))  # 14,14
        cam_map_seg = cam_map_cls.reshape(bz, 1, -1)  # (64,1,196)
        cam_sc_dot = torch.bmm(torch.from_numpy(cam_map_seg), torch.from_numpy(sc_map))  # (64,1,196)
        cam_sc_map = cam_sc_dot.reshape(bz, w_sc, h_sc)  # (64,14,14)
        sc_map_cls_i = torch.where(cam_sc_map >= 0, cam_sc_map, torch.zeros_like(cam_sc_map))

        sc_map_cls_i = (sc_map_cls_i - torch.min(sc_map_cls_i)) / (
                torch.max(sc_map_cls_i) - torch.min(sc_map_cls_i) + 1e-10)

        gt_scm = torch.where(sc_map_cls_i > 0, sc_map_cls_i, torch.zeros_like(sc_map_cls_i))
        # segment fg/bg for scm or not.
        gt_scm = self.get_masked_pseudo_gt(gt_scm, self.args.sos_fg_th, self.args.sos_bg_th,
                                           method=self.args.sos_seg_method) \
            if self.args.sos_gt_seg == 'True' else gt_scm
        gt_scm = gt_scm.detach()
        return gt_scm

    def hsc(self, f_phi, fo_th=0.2, so_th=1, order=2):
        n, c_nl, h, w = f_phi.size()
        f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
        f_phi_normed = f_phi / (torch.norm(f_phi, dim=2, keepdim=True) + 1e-10)

        # first order
        non_local_cos = F.relu(torch.matmul(f_phi_normed, f_phi_normed.transpose(1, 2)))
        non_local_cos[non_local_cos < fo_th] = 0
        non_local_cos_fo = non_local_cos.clone()
        non_local_cos_fo = non_local_cos_fo / (torch.sum(non_local_cos_fo, dim=1, keepdim=True) + 1e-5)

        # high order
        base_th = 1. / (h * w)
        non_local_cos[:, torch.arange(h * w), torch.arange(w * h)] = 0  # 对角线清零
        non_local_cos = non_local_cos / (torch.sum(non_local_cos, dim=1, keepdim=True) + 1e-5)
        non_local_cos_ho = non_local_cos.clone()
        so_th = base_th * so_th
        for _ in range(order - 1):
            non_local_cos_ho = torch.matmul(non_local_cos_ho, non_local_cos)
            non_local_cos_ho = non_local_cos_ho / (torch.sum(non_local_cos_ho, dim=1, keepdim=True) + 1e-10)
        non_local_cos_ho[non_local_cos_ho < so_th] = 0
        return non_local_cos_fo, non_local_cos_ho

    def normalize_feat(self, feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)
        return norm_feat

    def get_sparse_loss(self, act_map):
        n, h, w = act_map.shape
        post_map = torch.where(act_map >= 0, act_map, torch.zeros_like(act_map))
        spa_loss = torch.mean(torch.sum(torch.sum(post_map, dim=1), dim=1) / (h*w) * 1.0)
        return spa_loss * self.args.spa_loss_weight

    def get_sos_loss(self, pre_hm, gt_hm):
        if self.args.sos_gt_seg == 'False' or self.args.sos_loss_method == 'MSE':
            return self.mse_loss(pre_hm, gt_hm)
        if self.args.sos_seg_method == 'TC':
            if self.args.sos_loss_method == 'MSE':
                return self.mse_loss(pre_hm, gt_hm)
            elif self.args.sos_loss_method == 'BCE':
                return self.bce_loss(pre_hm, gt_hm)
        elif self.args.sos_seg_method == 'BC':
            if self.args.sos_loss_method == 'MSE':
                return self.mse_loss(pre_hm, gt_hm)
            elif self.args.sos_loss_method == 'BCE':
                return self.bce_loss(pre_hm, gt_hm)
        raise Exception("[Error] Invalid SOS segmentation or wrong sos loss type.")

    def get_cls_loss(self, logits, label):
        return self.ce_loss(logits, label.long())

    def get_ra_loss(self, logits, label, th_bg=0.3, bg_fg_gap=0.0):
        n, _, _, _ = logits.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)

        norm_var_logits = self.normalize_feat(var_logits)  # (n, w, h)

        bg_mask = (norm_var_logits < th_bg).float()
        fg_mask = (norm_var_logits > (th_bg + bg_fg_gap)).float()
        cls_map = logits[torch.arange(n), label.long(), ...]
        cls_map = torch.sigmoid(cls_map)
        ra_loss = torch.mean(cls_map * bg_mask + (1 - cls_map) * fg_mask)
        return ra_loss

    def get_loss(self, loss_params):
        loss = 0
        epoch, logits, label, pred_sos, gt_sos = loss_params.get('current_epoch'), loss_params.get('cls_logits'), \
                                                            loss_params.get('cls_label'), \
                                                            loss_params.get('pred_sos'), loss_params.get('gt_sos')
        cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
        # get cls loss
        cls_loss = self.get_cls_loss(cls_logits, label)
        loss = loss + cls_loss
        if 'sos' in self.args.mode and epoch >= self.args.sos_start:
            sos_loss = self.get_sos_loss(pred_sos, gt_sos)
            loss += self.args.sos_loss_weight * sos_loss
        else:
            sos_loss = torch.zeros_like(loss)
        if self.args.ram and epoch >= self.args.ram_start:
            ra_loss = self.get_ra_loss(logits, label, self.args.ram_th_bg, self.args.ram_bg_fg_gap)
            loss += self.args.ra_loss_weight * ra_loss
        else:
            ra_loss = torch.zeros_like(loss)
        return loss, cls_loss, ra_loss, sos_loss

    def cal_sc(self, feat):
        F1_2, F3, F4, F5 = feat
        sc_fo_2, sc_so_2, sc_fo_3, sc_so_3, sc_fo_4, sc_so_4, sc_fo_5, sc_so_5 = [None] * 8
        fo_th, so_th, order, stage = self.args.scg_fosc_th, self.args.scg_sosc_th, self.args.scg_order, self.args.scg_blocks
        if '2' in stage:
            fo_2, so_2 = self.hsc(F1_2, fo_th, so_th, order)
            sc_fo_2 = fo_2.clone().detach()
            sc_so_2 = so_2.clone().detach()
        if '3' in stage:
            fo_3, so_3 = self.hsc(F3, fo_th, so_th, order)
            sc_fo_3 = fo_3.clone().detach()
            sc_so_3 = so_3.clone().detach()
        if '4' in stage:
            fo_4, so_4 = self.hsc(F4, fo_th, so_th, order)
            sc_fo_4 = fo_4.clone().detach()
            sc_so_4 = so_4.clone().detach()
        if '5' in stage:
            fo_5, so_5 = self.hsc(F5, fo_th, so_th, order)
            sc_fo_5 = fo_5.clone().detach()
            sc_so_5 = so_5.clone().detach()
        return (sc_fo_2, sc_fo_3, sc_fo_4, sc_fo_5), (sc_so_2, sc_so_3, sc_so_4, sc_so_5)

    def cal_edge(self, feat_45):
        s_fo_th = self.args.scg_fosc_th
        s_so_th = self.args.scg_sosc_th
        s_order = self.args.scg_order
        ff4, ff5 = feat_45
        _mixed_edges = 0
        e_codes = []
        if '4' in self.args.sa_edge_stage:
            edge_fo4, edge_so4 = self.hsc(ff4, fo_th=s_fo_th, so_th=s_so_th, order=s_order)
            mixed_edge_4 = torch.max(edge_fo4, edge_so4)
            e_codes.append(mixed_edge_4)
        if '5' in self.args.sa_edge_stage:
            edge_fo5, edge_so5 = self.hsc(ff5, fo_th=s_fo_th, so_th=s_so_th, order=s_order)
            mixed_edge_5 = torch.max(edge_fo5, edge_so5)
            e_codes.append(mixed_edge_5)
        for c in e_codes:
            _mixed_edges += c
        _mixed_edges = _mixed_edges.detach()
        return _mixed_edges

    def _forward_spa(self, train_flag, feat):
        f12, f3, f4, f5 = feat
        sc_fo, sc_so = None, None
        cls_map = self.cls(f5)
        if not train_flag:
            sc_fo, sc_so = self.cal_sc(feat)
        return cls_map, sc_fo, sc_so  # 训练时sc_fo和sc_so=None

    def _forward_spa_sa(self, train_flag, current_epoch, feat):
        f12, f3, f4, f5 = feat
        batch, channel, _, _ = f5.shape
        sc_fo, sc_so = None, None
        if train_flag:
            if current_epoch >= self.args.sa_start:
                sa_in = f5.view(batch, channel, -1).permute(0, 2, 1)
                ho_self_corr = None
                if self.args.sa_use_edge == 'True':
                    ho_self_corr = self.cal_edge((f4, f5))
                cls_in = self.sa(sa_in, sa_in, sa_in, ho_self_corr)
            else:
                cls_in = f5
        else:
            sa_in = f5.view(batch, channel, -1).permute(0, 2, 1)
            sc_fo, sc_so = self.cal_sc(feat)
            edge_code = None
            if self.args.sa_use_edge == 'True':
                edge_code = self.cal_edge((f4, f5))
            cls_in = self.sa(sa_in, sa_in, sa_in, edge_code)
        cls_map = self.cls(cls_in)
        return cls_map, sc_fo, sc_so

    def _forward_sos_sa_v1(self, train_flag, current_epoch, feat):
        """
        sa module at cls branch.
        """
        pass

    def _forward_sos_sa_v2(self, train_flag, current_epoch, feat):
        """
        sa module at sos branch.
        """
        f12, f3, f4, f5 = feat
        cls_map = self.cls(f5)
        batch, channel, _, _ = f5.shape
        sc_fo, sc_so = self.cal_sc(feat)
        sos_map = None
        if train_flag:  # train
            try:
                assert self.args.sos_start <= self.args.sa_start
            except:
                raise Exception("[Error] sos start must before sa start! ")
            if self.args.sos_start <= current_epoch:
                if self.args.sa_start <= current_epoch:
                    edge_code = None
                    if self.args.sa_use_edge == 'True':
                        edge_code = self.cal_edge((f4, f5))
                    sa_in = f5.view(batch, channel, -1).permute(0, 2, 1)
                    sos_in = self.sa(sa_in, sa_in, sa_in, edge_code)
                else:
                    sos_in = f5
                sos_map = self.sos(sos_in)
                sos_map = sos_map.squeeze()
        else:  # test
            edge_code = None
            if self.args.sa_use_edge == 'True':
                edge_code = self.cal_edge((f4, f5))
            sa_in = f5.view(batch, channel, -1).permute(0, 2, 1)
            sos_in = self.sa(sa_in, sa_in, sa_in, edge_code)
            sos_map = self.sos(sos_in)
            sos_map = sos_map.squeeze()
        return cls_map, sos_map, sc_fo, sc_so

    def _forward_sos_sa_v3(self, train_flag, current_epoch, feat):
        """
        sa module at main branch.
        """
        sc_fo, sc_so = self.cal_sc(feat)
        f12, f3, f4, f5 = feat
        sos_map = None
        batch, channel, _, _ = f5.shape
        if train_flag:  # train
            if self.args.sa_start <= current_epoch:
                edge_code = None
                if self.args.sa_use_edge == 'True':
                    edge_code = self.cal_edge((f4, f5))
                sa_in = f5.view(batch, channel, -1).permute(0, 2, 1)
                sa_out = self.sa(sa_in, sa_in, sa_in, edge_code)
                cls_map = self.cls(sa_out)
                if self.args.sos_start <= current_epoch:
                    sos_map = self.sos(sa_out)
                    sos_map = sos_map.squeeze()
            else:
                cls_map = self.cls(f5)
                if self.args.sos_start <= current_epoch:
                    sos_map = self.sos(f5)
                    sos_map = sos_map.squeeze()
        else:  # test
            edge_code = None
            if self.args.sa_use_edge == 'True':
                edge_code = self.cal_edge((f4, f5))
            sa_in = f5.view(batch, channel, -1).permute(0, 2, 1)
            sa_out = self.sa(sa_in, sa_in, sa_in, edge_code)
            cls_map = self.cls(sa_out)
            sos_map = self.sos(sa_out)
            sos_map = sos_map.squeeze()
        return cls_map, sos_map, sc_fo, sc_so

    def _forward_sos(self, train_flag, current_epoch, feat):
        f12, f3, f4, f5 = feat
        cls_map = self.cls(f5)
        sc_fo, sc_so, sos_map = None, None, None
        if train_flag:  # train
            if self.args.sos_start <= current_epoch:
                sc_fo, sc_so = self.cal_sc(feat)
                sos_map = self.sos(f5)
                sos_map = sos_map.squeeze()  # squeeze cls_channel
        else:  # test
            sc_fo, sc_so = self.cal_sc(feat)
            sos_map = self.sos(f5)
            sos_map = sos_map.squeeze()  # squeeze batch_channel
        return cls_map, sos_map, sc_fo, sc_so

    # def get_masked_obj(self, sos_map, img):
    #     b_s, _, h, w = np.shape(img)
    #     sos_map = sos_map.unsqueeze(1)  # (bs,14,14) => (bs,1,14,14)
    #     sc_map = F.interpolate(sos_map, scale_factor=h / sos_map.shape[2], mode='bilinear', align_corners=True)
    #     sc_map = sc_map.squeeze()  # (bs,224,224)
    #     sc_map = sc_map.data.cpu().numpy()
    #     sc_map_cls = np.maximum(0, sc_map)
    #     gray_map = self.args.rcst_ratio * sc_map_cls
    #     gray_map = np.minimum(255, gray_map)
    #     gray_map = np.uint8(gray_map)
    #     gray_map[gray_map < 128] = 0
    #     gray_map[gray_map >= 128] = 255
    #     gray_map = gray_map.astype(np.float32) / 255  # convert to float32
    #     gray_map = torch.from_numpy(gray_map)
    #     gray_map = gray_map.cuda()
    #     gray_map = gray_map.view(b_s, 1, h, w)
    #     gt_masked = gray_map * img  # (bs,c,w,h)
    #     # test for mask
    #     # for i in range(gt_masked.shape[0]):
    #     #     import torchvision
    #     #     unloader = torchvision.transforms.ToPILImage()
    #     #     dd = img[i].cpu().clone()
    #     #     dd_d = unloader(dd)
    #     #     dd_d.save('bak.jpg')
    #     #     image = gt_masked[i].cpu().clone()
    #     #     img_masked = unloader(image)
    #     #     img_masked.save('masked.jpg')
    #     #     import pdb
    #     #     pdb.set_trace()
    #     gt_masked.detach()
    #     return gt_masked

    def forward(self, x, train_flag=True, cur_epoch=500):
        ft_1_2 = self.conv1_2(x)
        ft_3 = self.conv3(ft_1_2)
        ft_4 = self.conv4(ft_3)
        ft_5 = self.conv5(ft_4)
        ft_1_5 = (ft_1_2, ft_3, ft_4, ft_5)
        if self.args.mode == 'spa':
            return self._forward_spa(train_flag, ft_1_5)
        if self.args.mode == 'spa+sa':
            return self._forward_spa_sa(train_flag, cur_epoch, ft_1_5)
        if self.args.mode == 'sos':
            return self._forward_sos(train_flag, cur_epoch, ft_1_5)
        if self.args.mode == 'sos+sa_v3':
            return self._forward_sos_sa_v3(train_flag, cur_epoch, ft_1_5)
        raise Exception("[Error] Invalid training mode: ", self.args.mode)


def make_layers(cfg, dilation=None, batch_norm=False, instance_norm=False, inl=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'L':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif instance_norm and 256 > v > 64:
                layers += [conv2d, nn.InstanceNorm2d(v, affine=inl), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    # 'D_deeplab': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}

dilation = {
    'D_deeplab': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 2, 2, 2, 'N'],
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}

cnvs = {'O': (10, 7, 7), 'OI': (12, 7, 7)}


def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """

    layers = make_layers(cfg['O'], dilation=dilation['D1'])
    cnv = np.cumsum(cnvs['O'])
    model = VGG(layers, cnvs=cnv, **kwargs)
    if pretrained:
        pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.weight'.format(i + 10), 'conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 10), 'conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 17), 'conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 17), 'conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 24), 'conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 24), 'conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)

        model_dict = model.state_dict()
        pretrained_file = os.path.join(kwargs['args'].pretrained_model_dir, kwargs['args'].pretrained_model)
        if os.path.isfile(pretrained_file):
            pretrained_dict = torch.load(pretrained_file)
            print('load pretrained model from {}'.format(pretrained_file))
        else:
            pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
            print('load pretrained model from {}'.format(model_urls['vgg16']))
        # 0. replace the key
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}
        # *. show the loading information
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from vgg16'.format(k))
        print(' ')
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for SPA-Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

def load_finetune(pretrained=True, **kwargs):
    # construct model
    layers = make_layers(cfg['O'], dilation=dilation['D1'])
    cnv = np.cumsum(cnvs['O'])
    model = VGG(layers, cnvs=cnv, **kwargs)
    if pretrained:
        pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.weight'.format(i + 10), 'conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 10), 'conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 17), 'conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 17), 'conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 24), 'conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 24), 'conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)

        model_dict = model.state_dict()
        pretrained_file = os.path.join(kwargs['args'].pretrained_model_dir, kwargs['args'].pretrained_model)
        if os.path.isfile(pretrained_file):
            try:
                pretrained_dict = torch.load(pretrained_file)
                if 'state_dict' in pretrained_dict.keys():
                    pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
                else:
                    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
                print('load fine-tuned model from {}'.format(pretrained_file))
            except KeyError:
                print("Loading fine-tuned model failed.")
        else:
            raise Exception('[Error] Fine-tuned model does not exist, please check.')
        # 0. replace the key
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}
        # *. show the loading information
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from fine-tuned model.'.format(k))
        print(' ')
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for SPA-Net.'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        return model
    else:
        raise Exception('[Error] Load fine-tune model is only used in training mode.')


def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

if __name__ == '__main__':
    model(True)
