#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, T_max, eta_min, last_epoch, verbose, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                T_max = T_max,
                                eta_min = eta_min,
                                last_epoch = last_epoch,
                                verbose = verbose)

	print('Initialised ReduceLROnPlateau scheduler')

	return sche_fn
