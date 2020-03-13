import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn import metrics
from torch import nn


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


def mae(y, pred):
    return metrics.mean_absolute_error(y, pred)


def mse(y, pred):
    return metrics.mean_squared_error(y, pred)


def rmse(y, pred):
    return np.sqrt(mse(y, pred))


def mape(y, pred):
    # 过小的数据不计算进去
    err = 0.0
    remove_num = 0

    for i in range(len(y)):

        if abs(y[i]) < 0.01:
            remove_num += 1
        else:
            err += abs((y[i] - pred[i]) / y[i])

    return 100 * err / (len(y) - remove_num)


def pcc(y, pred):
    return pearsonr(y, pred)[0]


def all_metric(y, pred):
    return {
        'RMSE': rmse(y, pred),
        'MAE': mae(y, pred),
        'MAPE': mape(y, pred),
        'PCC': pcc(y, pred)
    }
