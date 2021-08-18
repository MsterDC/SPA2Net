import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
import os
from utils.vistools import norm_for_batch_map
from .fpn import FPN

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
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.num_classes = num_classes
        self.args = args

        self.cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0)
        )

        self.rcst = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, 3, kernel_size=1, padding=0)
        )

        self.fpn = FPN(
            out_channels=512
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

    def scg_process(self, feat):
        F1, F2, F3, F4 = feat
        fo_2, so_2, fo_3, so_3, fo_4, so_4, fo_5, so_5 = [None]*8
        if '2' in self.args.scg_blocks:
            fo_2, so_2 = self.hsc(F1, fo_th=self.args.scg_fosc_th,
                                  so_th=self.args.scg_sosc_th,
                                  order=self.args.scg_order)
        if '3' in self.args.scg_blocks:
            fo_3, so_3 = self.hsc(F2, fo_th=self.args.scg_fosc_th,
                                  so_th=self.args.scg_sosc_th,
                                  order=self.args.scg_order)
        if '4' in self.args.scg_blocks:
            fo_4, so_4 = self.hsc(F3, fo_th=self.args.scg_fosc_th,
                                  so_th=self.args.scg_sosc_th,
                                  order=self.args.scg_order)
        if '5' in self.args.scg_blocks:
            fo_5, so_5 = self.hsc(F4, fo_th=self.args.scg_fosc_th,
                                  so_th=self.args.scg_sosc_th,
                                  order=self.args.scg_order)
        return (fo_2, fo_3, fo_4, fo_5), (so_2, so_3, so_4, so_5)

    def forward(self, x, c_epoch=0, scg_flag=False):
        C1 = self.conv1_2(x)
        C2 = self.conv3(C1)
        C3 = self.conv4(C2)
        feat_5 = self.conv5(C3)
        cls_map = self.cls(feat_5)

        if scg_flag:
            fo_bb, so_bb, fo_fpn, so_fpn = [None]*4
            if 'bb' in self.args.scg_layers or 'both' in self.args.scg_layers:
                fo_bb, so_bb = self.scg_process((C1, C2, C3, feat_5))
            if 'fpn' in self.args.scg_layers or 'both' in self.args.scg_layers:
                C4 = self.maxpool(feat_5)
                p2, p3, p4, p5 = self.fpn(C2, C3, C4)
                fo_fpn, so_fpn = self.scg_process((p2, p3, p4, p5))
            return cls_map, (fo_bb, so_bb, fo_fpn, so_fpn)
        else:
            rcst_map = None
            if self.args.rcst and self.args.rcst_start <= c_epoch:
                C4 = self.maxpool(feat_5)
                p2, p3, p4, p5 = self.fpn(C2, C3, C4)
                rcst_map = self.rcst(p2)
            return cls_map, rcst_map

    def get_loss(self, logits, gt_child_label, rcst, img, current_epoch=0):
        cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
        loss = 0
        loss += self.loss_cross_entropy(cls_logits, gt_child_label.long())
        print("[TEST] cls loss:", loss)
        # cal reconstruct loss
        if self.args.rcst and current_epoch >= self.args.rcst_start:
            _, _, h, w = img.size()
            rcst = F.interpolate(rcst, size=(h, w), mode='bilinear', align_corners=True)
            rcst_loss = self.mse_loss(rcst, img)
            print("[TEST] rcst loss:", rcst_loss)
            loss += self.args.rcst_loss_weight * rcst_loss
        else:
            rcst_loss = torch.zeros_like(loss)
        if self.args.ram and current_epoch >= self.args.ram_start:
            ra_loss = self.get_ra_loss(logits, gt_child_label, self.args.ram_th_bg, self.args.ram_bg_fg_gap)
            loss += self.args.ra_loss_weight * ra_loss
        else:
            ra_loss = torch.zeros_like(loss)

        return loss, ra_loss, rcst_loss

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
