#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, factor, patience, min_lr, threshold, **kwargs):

	sche_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
							 'min', factor = factor, 
							 patience=patience, 
							 min_lr=min_lr, 
							 threshold=threshold, 
							 verbose=1)

	print('Initialised ReduceLROnPlateau scheduler')

	return sche_fn
