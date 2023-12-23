
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

"""
Ref: https://github.com/zyzisyz/mfa_conformer/blob/1b9c229948f8dbdbe9370937813ec75d4b06b097/module/feature.py#L26
Ref2: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
Ref3: https://github.com/zyzisyz/mfa_conformer/blob/master/module/feature.py
"""

    
class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, coef=0.97, **kwargs):
        super(Mel_Spectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.pre_emphasis = PreEmphasis(coef)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, \
                                                                    n_fft=self.n_fft,\
                                                                    win_length=self.win_length,\
                                                                    hop_length=self.hop_length, \
                                                                    n_mels=self.n_mels, \
                                                                    f_min = 20, f_max = 7600, \
                                                                    window_fn=torch.hamming_window, )


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, time)
        Returns:
            x (torch.Tensor): (batch, n_mels, time)
        """
        with torch.no_grad():
            x = self.pre_emphasis(x)
            x = self.mel_spectrogram(x) + 1e-6
            x = torch.log(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)

        return x
    

def feature_extractor(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80, coef=0.97, **kwargs):
    return Mel_Spectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, coef=coef)