import torch.nn.functional as F
import torch.nn as nn

# def loss_function(y_hat, target):
#     return F.binary_cross_entropy(y_hat, target)

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, y_hat, target):
        return self.bce_loss(y_hat, target)