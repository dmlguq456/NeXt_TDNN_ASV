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




# class NeXtTDNN(nn.Module): # ⚡
#     """ NeXt-TDNN model.
        
#     Args:
#         in_chans (int): Number of input channels. Default: 80
#         num_classes (int): Number of classes for classification head. Default: 192
#         depths (tuple(int)): Number of blocks at each stage. Default: [1, 1, 1]
#         dims (int): Feature dimension at each stage. Default: [256, 256, 256]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#     """
#     def __init__(self, in_chans=80, # in_channels: 3 -> 80
#                  depths=[1, 1, 1], dims=[256, 256, 256],  
#                  drop_path_rate=0., 
#                  kernel_size_list = [7, 15, 33, 65], 
#                  ):
#         super().__init__()
#         self.depths = depths
#         self.stem = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
#         Conv1DLayer = nn.Sequential(
#             nn.Conv1d(in_chans, dims[0], kernel_size=4), 
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.stem.append(Conv1DLayer) 


#         self.stages = nn.ModuleList() # 3 feature resolution stages, each consisting of multiple residual blocks
#         dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
#         cur = 0
#         for i in range(len(self.depths)): 
#             stage = nn.Sequential(
#                 *[TSConvNeXt(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size_list=kernel_size_list) for j in range(depths[i])] # ⚡
#             )
#             self.stages.append(stage)
#             cur += depths[i]

#         self.apply(self._init_weights)


#         # MFA layer
#         self.MFA = nn.Sequential( # ⚡
#             nn.Conv1d(3*dims[-1], int(3*dims[-1]), kernel_size=1),
#             LayerNorm(int(3*dims[-1]), eps=1e-6, data_format="channels_first") # ⚡
#         )


#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv1d, nn.Linear)): # ⚡
#             trunc_normal_(m.weight, std=.02)
#             nn.init.constant_(m.bias, 0)

#     def forward_features(self, x):
#         x = self.stem[0](x) # ⚡ stem_tdnn

#         mfa_in = []
#         for i in range(len(self.depths)): # ⚡ 4 -> 3(len(self.depths))
#             x = self.stages[i](x)
#             mfa_in.append(x)

#         return mfa_in # ⚡

#     def forward(self, x):
#         """
#         Args:
#             x (Tensor): Input tensor of shape (N, C, T).
#         Returns:
#             Tensor: Output tensor of shape (N, C, T).
#         """
#         # x = self.stem[0](x)

#         # x1 = self.stages[0](x)
#         # x2 = self.stages[1](x1)
#         # x3 = self.stages[2](x2)

#         x = self.forward_features(x) # ⚡

#         # MFA layer
#         x = torch.cat(x, dim=1)
#         x = self.MFA(x) # Conv1d + LayerNorm_TDNN

#         return x

# def MainModel(**kwargs):
#     model = NeXtTDNN(**kwargs)
#     return model
