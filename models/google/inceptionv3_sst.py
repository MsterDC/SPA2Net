import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from utils.vistools import norm_for_batch_map
from .sa import ScaledDotProductAttention

__all__ = ['Inception3', 'model']

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def model(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = Inception3(**kwargs)
        model_dict = model.state_dict()
        pretrained_file = os.path.join(kwargs['args'].pretrained_model_dir, kwargs['args'].pretrained_model)
        if os.path.isfile(pretrained_file):
            pretrained_dict = torch.load(pretrained_file)
            print('load pretrained model from {}'.format(pretrained_file))
        else:
            pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
            print('load pretrained model from: {}'.format(model_urls['inception_v3_google']))
        for k in pretrained_dict.keys():
            # print('Pretrained Key {}:'.format(k))
            if k not in model_dict:
                print('Key {} is removed from inception v3'.format(k))
        for k in model_dict.keys():
            # print('SPA-Net Key {}:'.format(k))
            if k not in pretrained_dict:
                print('Key {} is new added for SPA-Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, args=None):
        super(Inception3, self).__init__()
        self.args = args
        self.num_classes = num_classes

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3,stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3,stride=1, padding=0)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.cls = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0))

        if 'sa' in self.args.mode:
            self.sa = ScaledDotProductAttention(d_model=int(args.sa_neu_num), d_k=int(args.sa_neu_num),
                                                d_v=int(args.sa_neu_num), h=int(args.sa_head), weight=args.sa_edge_weight)

        if 'sos' in self.args.mode:
            self.sos = nn.Sequential(
                nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
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
        spa_loss = torch.mean(torch.sum(torch.sum(post_map, dim=1), dim=1) / (h * w) * 1.0)
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
        return cls_map, sc_fo, sc_so

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

    def _forward_sos_sa_v3(self, train_flag, current_epoch, feat):
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

    def forward(self, x, train_flag=True, cur_epoch=500):

        x = self.Conv2d_1a_3x3(x)  # 224 x 224 x 3
        x = self.Conv2d_2a_3x3(x)  # 112 x 112 x 32
        x = self.Conv2d_2b_3x3(x)  # 112 x 112 x 32
        feat1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)  # 112 x 112 x 32
        x = self.Conv2d_3b_1x1(feat1)  # 56 x 56 x 64
        x = self.Conv2d_4a_3x3(x)  # 56 x 56 x 64
        feat2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)  # 56 x 56 x 64
        x = self.Mixed_5b(feat2)  # 28 x 28 x 192
        x = self.Mixed_5c(x)  # 28 x 28 x 192
        feat3 = self.Mixed_5d(x)  # 28 x 28 x 192
        x = self.Mixed_6a(feat3)  # 28 x 28 x 192
        x = self.Mixed_6b(x)  # 28 x 28 x 768
        x = self.Mixed_6c(x)  # 28 x 28 x 768
        x = self.Mixed_6d(x)  # 28 x 28 x 768
        feat4 = self.Mixed_6e(x)  # 28 x 28 x 768

        ft_1_5 = (feat1, feat2, feat3, feat4)
        if self.args.mode == 'spa':
            return self._forward_spa(train_flag, ft_1_5)
        if self.args.mode == 'spa+sa':
            return self._forward_spa_sa(train_flag, cur_epoch, ft_1_5)
        if self.args.mode == 'sos':
            return self._forward_sos(train_flag, cur_epoch, ft_1_5)
        if self.args.mode == 'sos+sa_v3':
            return self._forward_sos_sa_v3(train_flag, cur_epoch, ft_1_5)
        raise Exception("[Error] Invalid training mode: ", self.args.mode)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        self.stride = stride
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x

