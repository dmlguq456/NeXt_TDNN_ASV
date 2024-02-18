#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, base_lr = 1e-8, max_lr = 1e-3, step_size_up = 130000/2, mode = 'triangular2', cycle_momentum = False, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, step_size_up = step_size_up, mode = mode, cycle_momentum = cycle_momentum)

	print('Initialised step LR scheduler')

	return sche_fn