import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """ Version 1.0
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """Replace the original CrossEntropyLoss Function.
        Copy from [https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/smoothing.py]
        :param x: size => [batch_size, channel_num]
        :param target: size => [batch_size]
        :return: batch average loss
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothingCrossEntropy(nn.Module):
    """ Version 2.0
    NLL loss with label smoothing.
    """
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        """Replace the original CrossEntropyLoss Function.
        Copy from DeepHub [https://blog.csdn.net/deephub/article/details/121314083]
        :param preds: size => [batch_size, channel_num]
        :param target: size => [batch_size]
        :return: batch average loss
        """
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


# if __name__ == "__main__":
#     # Either of the two will do
#     loss_1 = LabelSmoothing(smoothing=0.1)
#     loss_2 = LabelSmoothingCrossEntropy()
#
#     logits = torch.rand(64, 1000)
#     random_list = [random.randint(0,999) for i in range(64)]
#     target = torch.tensor(random_list)
#
#     print("final loss:\n", loss_1(logits, target))
#     print("final loss:\n", loss_2(logits, target))