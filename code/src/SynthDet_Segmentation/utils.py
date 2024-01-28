from abc import ABCMeta, abstractmethod
import os
import os.path as osp
import random
import time
import torch
import torch.nn as nn


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)
   
    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group

class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass

class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)
        