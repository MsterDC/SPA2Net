import numpy as np

import torch
from torch import nn
from torch.nn import init


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)  # 512 -> 512*h
        self.fc_k = nn.Linear(d_model, h * d_k)  # 512 -> 512*h
        self.fc_v = nn.Linear(d_model, h * d_v)  # 512 -> 512*h
        self.fc_o = nn.Linear(h * d_v, d_model)  # 512*h -> 512
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.attention_map = None
        self.enhanced_map = None

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, self_corr=None, attention_mask=None, attention_weights=None):
        bt_sz, nq, cc = queries.shape  # (bs, h*w, c_n)
        nk = keys.shape[1]  # h*w

        q = self.fc_q(queries).view(bt_sz, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(bt_sz, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(bt_sz, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_qk = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        self.attention_map = att_qk

        if self_corr is not None:
            self_corr = self_corr.view(bt_sz, -1, nq, nk)

            enhanced_weight = att_qk + self_corr
            self.enhanced_map = enhanced_weight
            soft_en_map = torch.softmax(enhanced_weight, -1)
            drop_en_map = self.dropout(soft_en_map)

            # soft_en_map = torch.softmax(att_qk, -1)
            # enhanced_weight = soft_en_map + self.edge_weight * self_corr
            # self.enhanced_map = enhanced_weight
            # drop_en_map = self.dropout(enhanced_weight)

            att_qkv = torch.matmul(drop_en_map, v).permute(0, 2, 1, 3).contiguous().view(bt_sz, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        else:
            soft_att_qk = torch.softmax(att_qk, -1)
            drop_att_qk = self.dropout(soft_att_qk)
            att_qkv = torch.matmul(drop_att_qk, v).permute(0, 2, 1, 3).contiguous().view(bt_sz, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)

        att_qkv = self.fc_o(att_qkv)  # (b_s, nq, d_model)
        att_qkv = att_qkv.transpose(1, 2).contiguous().view(bt_sz, cc, int(np.sqrt(nq)), int(np.sqrt(nq)))

        return att_qkv

    def get_attention(self):
        return self.attention_map

    def get_enhanced(self):
        return self.enhanced_map