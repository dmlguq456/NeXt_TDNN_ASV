import torch
import torch.nn as nn
import torchaudio


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=8, time_mask_param=8, **kwargs):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param = self.time_mask_param)

    def forward(self, x):
        """
        Args:
            x (torch tensor): (batch, freq, time)
        Return:
            x (torch tensor): (batch, freq, time)
        """
        assert x.dim() == 3, "Input must be [batch, freq, time]"

        if self.training is False:
            return x
        
        else:            
            x = self.fm(x)
            x = self.tm(x)

        return x
    
def spec_aug(freq_mask_param=8, time_mask_param=8, **kwargs):
    return SpecAugment(freq_mask_param=freq_mask_param, time_mask_param=time_mask_param, **kwargs)