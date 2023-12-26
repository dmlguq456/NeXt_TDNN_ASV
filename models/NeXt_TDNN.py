import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import Union, List

from models.utils import LayerNorm


class NeXtTDNN(nn.Module): #
    """ NeXt-TDNN / NeXt-TDNN-Light model.
        
    Args:
        in_chans (int): Number of input channels. Default: 80
        depths (tuple(int)): Number of blocks at each stage. Default: [1, 1, 1]
        dims (int): Feature dimension at each stage. Default: [256, 256, 256]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """
    def __init__(self, in_chans=80, # in_channels: 3 -> 80
                 depths=[1, 1, 1], dims=[256, 256, 256],  
                 drop_path_rate=0., # 
                 kernel_size: Union[int, List[int]] = 7,
                 block = "TSConvNeXt_light", # TSConvNeXt_light or TSConvNeXt
                 ):
        super().__init__()
        self.depths = depths
        self.stem = nn.ModuleList() 
        Conv1DLayer = nn.Sequential( 
            nn.Conv1d(in_chans, dims[0], kernel_size=4), 
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") # ⚡
        )
        self.stem.append(Conv1DLayer)

 
        block = importlib.import_module(f"models.{block}").__getattribute__(block) # TSConvNeXt_light or TSConvNeXt

        self.stages = nn.ModuleList() # 3 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(self.depths)): # ⚡
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size) for j in range(depths[i])] # ⚡
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

        # MFA layer
        self.MFA = nn.Sequential( # ⚡
            nn.Conv1d(3*dims[-1], int(3*dims[-1]), kernel_size=1),
            LayerNorm(int(3*dims[-1]), eps=1e-6, data_format="channels_first") # ⚡
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)): # ⚡
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        x = self.stem[0](x) # ⚡ stem_tdnn

        mfa_in = []
        for i in range(len(self.depths)): # ⚡ 4 -> 3(len(self.depths))
            x = self.stages[i](x)
            mfa_in.append(x)

        return mfa_in # ⚡

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, T).
        Returns:
            Tensor: Output tensor of shape (N, C, T).
        """
        x = self.forward_features(x) # ⚡

        # MFA layer
        x = torch.cat(x, dim=1)
        x = self.MFA(x) # Conv1d + LayerNorm_TDNN

        return x

def MainModel(**kwargs):
    model = NeXtTDNN(**kwargs)
    return model