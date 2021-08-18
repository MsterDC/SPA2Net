import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
import os
from utils.vistools import norm_for_batch_map

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
        self.conv5_add = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True))
        self.fmp = features[-1]  # final max pooling
        self.num_classes = num_classes
        self.args = args

        self.cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0)
        )

        self.sos = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, 1, kernel_size=1, padding=0)
        )

        self._initialize_weights()

        # loss function
        self.loss_cross_entropy = F.cross_entropy
        self.loss_bce = F.binary_cross_entropy_with_logits
        self.nll_loss = F.nll_loss
        self.mse_loss = torch.nn.MSELoss(reduce=True, size_average=True)

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

    def forward(self, x, scg_flag=False):
        # use hsc
        if self.args.scg:
            x = self.conv1_2(x)
            sc_fo_2 = None
            sc_so_2 = None
            if scg_flag and '2' in self.args.scg_blocks:
                sc_2, sc_2_so = self.hsc(x, fo_th=self.args.scg_fosc_th,
                                         so_th=self.args.scg_sosc_th,
                                         order=self.args.scg_order)
                sc_fo_2 = sc_2.detach()
                sc_so_2 = sc_2_so.detach()

            feat_3 = self.conv3(x)
            sc_fo_3 = None
            sc_so_3 = None
            if scg_flag and '3' in self.args.scg_blocks:
                sc_3, sc_3_so = self.hsc(feat_3, fo_th=self.args.scg_fosc_th,
                                         so_th=self.args.scg_sosc_th,
                                         order=self.args.scg_order)
                sc_fo_3 = sc_3.detach()
                sc_so_3 = sc_3_so.detach()

            feat_4 = self.conv4(feat_3)
            sc_fo_4 = None
            sc_so_4 = None
            if scg_flag and '4' in self.args.scg_blocks:
                sc_4, sc_4_so = self.hsc(feat_4, fo_th=self.args.scg_fosc_th,
                                         so_th=self.args.scg_sosc_th,
                                         order=self.args.scg_order)
                sc_fo_4 = sc_4.detach()
                sc_so_4 = sc_4_so.detach()

            feat_5 = self.conv5(feat_4)
            sc_fo_5 = None
            sc_so_5 = None
            if scg_flag and '5' in self.args.scg_blocks:
                sc_5, sc_5_so = self.hsc(feat_5, fo_th=self.args.scg_fosc_th,
                                         so_th=self.args.scg_sosc_th,
                                         order=self.args.scg_order)
                sc_fo_5 = sc_5.detach()
                sc_so_5 = sc_5_so.detach()

            # For classification
            cls_map = self.cls(feat_5)

            # For second order map supervised
            sos_map = self.sos(feat_5)

            # [warning] output has changed
            return cls_map, sos_map, (sc_fo_2, sc_fo_3, sc_fo_4, sc_fo_5), (sc_so_2, sc_so_3, sc_so_4, sc_so_5)

    def hsc(self, f_phi, fo_th=0.2, so_th=1, order=2):
        """
        Calculate affinity matrix and update feature.
        :param feat:
        :param f_phi:
        :param fo_th:
        :param so_weight:
        :return:
        """
        n, c_nl, h, w = f_phi.size()
        # test for block 3
        if h != 14 or w != 14:
            h, w = 14, 14
            f_phi = F.interpolate(f_phi, size=(h, w), mode='bilinear', align_corners=True)
        f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
        f_phi_normed = f_phi / (torch.norm(f_phi, dim=2, keepdim=True) + 1e-10)

        # first order
        non_local_cos = F.relu(torch.matmul(f_phi_normed, f_phi_normed.transpose(1, 2)))
        non_local_cos[non_local_cos < fo_th] = 0
        non_local_cos_fo = non_local_cos.clone()
        non_local_cos_fo = non_local_cos_fo / (torch.sum(non_local_cos_fo, dim=1, keepdim=True) + 1e-5)

        # high order
        base_th = 1. / (h * w)
        non_local_cos[:, torch.arange(h * w), torch.arange(w * h)] = 0
        non_local_cos = non_local_cos / (torch.sum(non_local_cos, dim=1, keepdim=True) + 1e-5)
        non_local_cos_ho = non_local_cos.clone()
        so_th = base_th * so_th
        for _ in range(order - 1):
            non_local_cos_ho = torch.matmul(non_local_cos_ho, non_local_cos)
            non_local_cos_ho = non_local_cos_ho / (torch.sum(non_local_cos_ho, dim=1, keepdim=True) + 1e-10)
        non_local_cos_ho[non_local_cos_ho < so_th] = 0
        return non_local_cos_fo, non_local_cos_ho

    def get_heatmap(self, logits, gt_label, sc_maps_fo, sc_maps_so, fg_th, bg_th):
        """
        Get the SOS supervised signal.
        :param logits: cls logits
        :param gt_label:
        :param sc_maps_fo:
        :param sc_maps_so:
        :param fg_th:
        :param bg_th:
        :return: GT-SCM
        """
        logit_grad = logits.detach()
        loc_map = F.relu(logit_grad)
        cam_map = loc_map.data.cpu().numpy()
        # gt_known
        cam_map_ = cam_map[:, int(gt_label[0]), :, :]  # (bs,14,14)
        cam_map_cls = norm_for_batch_map(cam_map_)  # (bs,14,14)
        sc_maps = []
        if self.args.scg_com:
            for sc_map_fo_i, sc_map_so_i in zip(sc_maps_fo, sc_maps_so):
                if (sc_map_fo_i is not None) and (sc_map_so_i is not None):
                    sc_map_so_i = sc_map_so_i.to(self.args.device)
                    sc_map_i = torch.max(sc_map_fo_i, self.args.scg_so_weight * sc_map_so_i)  # sc_fo 与 sc_so 融合
                    sc_map_i = sc_map_i / (torch.sum(sc_map_i, dim=1, keepdim=True) + 1e-10)
                    sc_maps.append(sc_map_i)
        sc_com = sc_maps[-2] + sc_maps[-1]  # Conv_layer4 + Conv_layer5
        sc_map = sc_com.squeeze().data.cpu().numpy()  # (bs,196,196)
        wh_sc, bz = sc_map.shape[1], sc_map.shape[0]
        h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))  # 14,14
        cam_map_cls_vector = cam_map_cls.reshape(bz, 1, -1)  # (bs,1,196)
        cam_sc_mul = torch.bmm(torch.from_numpy(cam_map_cls_vector), torch.from_numpy(sc_map))  # 矩阵乘法:得(bs,1,196)
        cam_sc_map = cam_sc_mul.reshape(bz, w_sc, h_sc)  # (bs,14,14)
        sc_map_cls_i = torch.where(cam_sc_map >= 0, cam_sc_map, torch.zeros_like(cam_sc_map))  # 负值取0
        sc_map_cls_i = (sc_map_cls_i - torch.min(sc_map_cls_i)) / \
                       (torch.max(sc_map_cls_i) - torch.min(sc_map_cls_i) + 1e-10)  # 归一化
        gt_scm = torch.where(sc_map_cls_i > 0, sc_map_cls_i, torch.zeros_like(sc_map_cls_i))
        if fg_th and bg_th:
            mask_zero = torch.zeros_like(gt_scm)
            mask_one = torch.ones_like(gt_scm)
            gt_scm = torch.where(gt_scm > fg_th, mask_one, gt_scm)
            gt_scm = torch.where(gt_scm < bg_th, mask_zero, gt_scm)
        gt = gt_scm.detach()
        return gt

    def get_loss(self, logits, gt_child_label, pre_hm, gt_hm, epoch=0, ram_start=10, sos_start=10):
        cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
        loss = 0
        loss += self.loss_cross_entropy(cls_logits, gt_child_label.long())
        # print("[TEST] cls loss:", loss)
        if self.args.sos and epoch >= sos_start:
            sos_loss = self.mse_loss(pre_hm, gt_hm)
            # print("[TEST] sos_loss:", sos_loss)
            loss += self.args.sos_loss_weight * sos_loss
            # print("[TEST] sos_added_loss:", loss)
        else:
            sos_loss = torch.zeros_like(loss)
        if self.args.ram and epoch >= ram_start:
            ra_loss = self.get_ra_loss(logits, gt_child_label, self.args.ram_th_bg, self.args.ram_bg_fg_gap)
            loss += self.args.ra_loss_weight * ra_loss
        else:
            ra_loss = torch.zeros_like(loss)

        return loss, ra_loss, sos_loss

    def get_ra_loss(self, logits, label, th_bg=0.3, bg_fg_gap=0.0):
        n, _, _, _ = logits.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = self.normalize_feat(var_logits)

        bg_mask = (norm_var_logits < th_bg).float()
        fg_mask = (norm_var_logits > (th_bg + bg_fg_gap)).float()
        cls_map = logits[torch.arange(n), label.long(), ...]
        cls_map = torch.sigmoid(cls_map)

        ra_loss = torch.mean(cls_map * bg_mask + (1 - cls_map) * fg_mask)
        return ra_loss

    def normalize_feat(self, feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)

        return norm_feat

    def get_cls_maps(self):
        return F.relu(self.cls_map)

    def get_loc_maps(self):
        return torch.sigmoid(self.loc_map)


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
            elif instance_norm and v < 256 and v > 64:
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
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    model(True)
