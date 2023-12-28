# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import List


from models.utils import LayerNorm, GRN



class TSConvNeXt(nn.Module):
    """ TSConvNeXt Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., kernel_size:List[int] = [7, 65]):
        super().__init__()
        for kernel in kernel_size:
            assert (kernel - 1) % 2 == 0, "kernel size must be odd number"
        self.projection_linear = nn.Conv1d(dim, dim, kernel_size=1) 
        self.num_scale = len(kernel_size)
        self.mscconv = nn.ModuleList()
        for i in range(self.num_scale):
            self.mscconv.append(nn.Conv1d(dim//self.num_scale, dim//self.num_scale, kernel_size=kernel_size[i], padding=((kernel_size[i]-1)//2), groups=dim//self.num_scale))

        self.pwconv_1stage = nn.Linear(dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, T).
        Returns:
            Tensor: Output tensor of shape (N, C, T).            
        """

        # MSC module
        input = x
        # Linear projection
        x = self.projection_linear(x)
        x = x.chunk(self.num_scale, dim=1)                
        
        x_msc = []
        for i in range(self.num_scale):
            x_msc.append(self.mscconv[i](x[i]))
        x = torch.cat(x_msc, dim=1)

        x = self.act(x)
        x = x.permute(0, 2, 1) # (N, C, T) -> (N, T, C) 
        x = self.pwconv_1stage(x)
        x = x.permute(0, 2, 1) # (N, C, T) -> (N, T, C)
        x = x + input

        # FFN module
        input = x
        x = x.permute(0, 2, 1) # (N, C, T) -> (N, T, C) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, T, C) -> (N, C, T) 

        x = input + self.drop_path(x)
        return x
