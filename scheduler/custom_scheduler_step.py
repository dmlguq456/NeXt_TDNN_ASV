#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.optim.lr_scheduler import _LRScheduler


       
class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, plateau_epoch, decay_interval, initial_lr=1e-5, plateau_lr=0.1, decay_rate=0.5, iters_per_epoch = 1365, **kwargs):
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.plateau_iters = plateau_epoch * iters_per_epoch
        self.decay_interval_iters = decay_interval * iters_per_epoch
        self.initial_lr = initial_lr
        self.plateau_lr = plateau_lr
        self.decay_rate = decay_rate
        self.update = "step"
        super().__init__(optimizer)

    def get_lr(self):
        iteration = self.last_epoch
        if iteration < self.warmup_iters:
            return [self.initial_lr + iteration * (self.plateau_lr - self.initial_lr) / self.warmup_iters for _ in self.base_lrs]
        elif iteration < (self.warmup_iters + self.plateau_iters):
            return [self.plateau_lr for _ in self.base_lrs]
        else:
            return [self.plateau_lr * self.decay_rate ** ((iteration - self.warmup_iters - self.plateau_iters) // self.decay_interval_iters) for _ in self.base_lrs]



def Scheduler(optimizer, warmup_epochs, plateau_epoch, decay_interval, initial_lr=1e-5, plateau_lr=0.1, decay_rate=0.5, iters_per_epoch = 1365, **kwargs):
    sche_fn = CustomScheduler(optimizer, warmup_epochs, plateau_epoch, decay_interval, initial_lr, plateau_lr, decay_rate, iters_per_epoch, **kwargs)
    print('Initialised custom LR scheduler - Step')
    return sche_fn
