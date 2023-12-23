import torch
import torch.nn as nn
import torch.nn.functional as F


class SAP_FC(nn.Module):
    def __init__(self, channel_size, embeding_size):
        super(SAP_FC, self).__init__()
        self.channel_size = channel_size
        self.embeding_size = embeding_size

        self.sap_linear = nn.Linear(self.channel_size, self.channel_size)
        self.context_vector = self.new_parameter(self.channel_size, 1)
        self.tahn = nn.Tanh()

        self.fc = nn.Linear(self.channel_size, self.embeding_size)

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
        h = self.sap_linear(x) # (batch_size, T, channel_size)
        h = self.tahn(h)        
        w = torch.matmul(h, self.context_vector)  # (batch_size, T, 1)
        w = F.softmax(w, dim=1) # (batch_size, T, 1)

        x = torch.sum(x * w, dim=1)  # (batch_size, channel_size)

        x = self.fc(x)



        return x


def Aggregation(channel_size, embeding_size):
    return SAP_FC(channel_size, embeding_size)