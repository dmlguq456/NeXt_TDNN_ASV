#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=3, plateau_epoch=10, decay_interval=4, initial_lr=1e-5, plateau_lr=0.1, decay_rate=0.5, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.plateau_epochs = plateau_epoch
        self.decay_interval = decay_interval
        self.initial_lr = initial_lr
        self.plateau_lr = plateau_lr
        self.decay_rate = decay_rate
        self.update = "epoch"
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            return [self.initial_lr + epoch * (self.plateau_lr - self.initial_lr) / self.warmup_epochs for _ in self.base_lrs]
        elif epoch < (self.warmup_epochs + self.plateau_epochs):
            return [self.plateau_lr for _ in self.base_lrs]
        else:
            return [self.plateau_lr * self.decay_rate ** ((epoch - self.warmup_epochs - self.plateau_epochs) // self.decay_interval) for _ in self.base_lrs]





def Scheduler(optimizer, warmup_epochs=3, plateau_epoch=10, decay_interval=4, initial_lr=1e-5, plateau_lr=0.1, decay_rate=0.5, **kwargs):
    sche_fn = CustomScheduler(optimizer, warmup_epochs=warmup_epochs, plateau_epoch=plateau_epoch, decay_interval=decay_interval, initial_lr=initial_lr, plateau_lr=plateau_lr, decay_rate=decay_rate, **kwargs)
    print('Initialised custom LR scheduler')
    return sche_fn
