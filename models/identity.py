import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def MainModel(**kwargs):
    return Identity(**kwargs)