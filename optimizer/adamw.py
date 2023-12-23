#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):

	print('Initialised AdamW optimizer')

	return torch.optim.AdamW(parameters, lr = lr, weight_decay = weight_decay);
