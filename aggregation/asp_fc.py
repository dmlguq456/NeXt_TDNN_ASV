import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Ref: https://github.com/clovaai/voxceleb_trainer/blob/master/models/ResNetSE34L.py
"""

class ASP_FC(nn.Module):
    def __init__(self, channel_size, embeding_size):
        super(ASP_FC, self).__init__()
        self.channel_size = channel_size
        self.embeding_size = embeding_size

        self.asp_linear = nn.Linear(self.channel_size, self.channel_size)
        self.context_vector = self.new_parameter(self.channel_size, 1)
        self.tahn = nn.Tanh()

        self.fc = nn.Linear(self.channel_size * 2, self.embeding_size)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        """
        Args:
            x: (batch_size, channel_size, T)
        Returns:
            x: (batch_size, embeding_size)
        """
        assert x.dim() == 3, "x.dim() must be 3"

        x = x.permute(0, 2, 1)  # (batch_size, T, channel_size)
        h = self.asp_linear(x) # (batch_size, T, channel_size)
        h = self.tahn(h)        
        w = torch.matmul(h, self.context_vector)  # (batch_size, T, 1)
        w = F.softmax(w, dim=1) # (batch_size, T, 1)

        mu = torch.sum(x * w, dim=1)  # (batch_size, channel_size)
        rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5))

        x = torch.cat((mu, rh), dim=1)

        x = self.fc(x)

        return x


def Aggregation(channel_size, embeding_size):
    return ASP_FC(channel_size, embeding_size)