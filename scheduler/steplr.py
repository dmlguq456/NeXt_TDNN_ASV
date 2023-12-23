#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, step_size, gamma, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	print('Initialised step LR scheduler')

	return sche_fn