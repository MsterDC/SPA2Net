import torch.optim as optim
import numpy as np
from collections.abc import Iterable


def get_finetune_optimizer(args, model):
    lr = args.lr
    added_layers = ['fc6', 'fc7_1', 'classier_1', 'branchB', 'side3', 'side4', 'side_all'] if args.diff_lr == 'True' else []
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if any([x in name for x in added_layers]):
            print(name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr': lr},
                     {'params': bias_list, 'lr': lr * 2},
                     {'params': last_weight_list, 'lr': lr * 10},
                     {'params': last_bias_list, 'lr': lr * 20}],
                     momentum=0.9, weight_decay=0.0005)

    return opt


def lr_poly(base_lr, iter, max_iter, power=0.9):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)


def get_optimizer(args, model):
    lr = args.lr
    opt = optim.SGD(params=[para for name, para in model.named_parameters() if 'features' not in name], lr=lr,
                    momentum=0.9, weight_decay=0.0001)
    return opt


def get_adam(args, model):
    lr = args.lr
    opt = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0005)
    return opt


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
    pass


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)