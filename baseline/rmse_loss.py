import torch
from torch import nn


class RMSELoss(nn.Module):

    def __init__(self, mode):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mode = mode

    def forward(self, y_hat, y):

        if self.mode == 'train':  # 训练的时候，loss 算在一起，其他时候要分开算
            return torch.sqrt(self.mse(y_hat, y))
        else:
            return torch.sqrt(self.mse(y_hat[:, 0], y[:, 0])), torch.sqrt(self.mse(y_hat[:, 1], y[:, 1]))
