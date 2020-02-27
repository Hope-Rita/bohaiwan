import numpy as np
from scipy.stats import pearsonr
from sklearn import metrics


def mae(y, pred):
    return metrics.mean_absolute_error(y, pred)


def mse(y, pred):
    return metrics.mean_squared_error(y, pred)


def rmse(y, pred):
    return np.sqrt(mse(y, pred))


def pcc(y, pred):
    return pearsonr(y, pred)[0]


def all_metric(y, pred):
    return {
        'RMSE': rmse(y, pred),
        'MAE': mae(y, pred),
        'PCC': pcc(y, pred)
    }
