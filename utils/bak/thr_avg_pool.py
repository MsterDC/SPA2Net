import torch
import torch.nn as nn


class ThresholdedAvgPool2d(nn.Module):
    def __init__(self, threshold=0.0):
        super(ThresholdedAvgPool2d, self).__init__()
        self.threshold = threshold  # 0.1

    def forward(self, feature_map):
        # threshold feature map
        batch_size, channel, height, width = feature_map.shape
        # find the max value of feature map
        max_vals, _ = torch.max(feature_map.view(batch_size, channel, -1), dim=2)  # (bs, c_l)
        # each channel of feature map has a max value
        thr_vals = (max_vals * self.threshold).view(batch_size, channel, 1, 1).expand_as(feature_map)  # (bs,c_l,h,w)
        # 大于阈值的该位置值=feat.value，小于阈值的=0
        thr_feature_map = torch.where(torch.gt(feature_map, thr_vals), feature_map, torch.zeros_like(feature_map))  # (bs,c_l,h,w)
        batch_cl_thr_sum = torch.sum(thr_feature_map, dim=(2, 3))  # (bs,channel)
        # divided by the number of positives
        num_positives = torch.sum(torch.gt(thr_feature_map, 0.), dim=(2, 3))  # (bs, channel)
        num_positives = torch.where(torch.eq(num_positives, 0), torch.ones_like(num_positives), num_positives)
        thr_pooled_logits = torch.div(batch_cl_thr_sum, num_positives.float())  # (bs, channel)
        # (bs,c,1,1) => (bs,c,h,w)
        # num_positives = torch.where(torch.eq(num_positives, 0),  # 相等的位置为1，不相等的位置为0
        #                             torch.ones_like(num_positives),  # (bs, channel)
        #                             num_positives).view(batch_size, channel, 1, 1).expand_as(feature_map)  # (bs,cl,h,w)
        # avg_feature_map = torch.div(thr_feature_map, num_positives.float())  # (bs,cl,h,w)

        return thr_pooled_logits
