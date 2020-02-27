import numpy as np
import pandas as pd
import utils.load_utils as ld
from utils import data_process
from utils.config import get_config


# 加载参数 使用时间序列的长度，预测未来的天数
pred_len, future_days = get_config('config.json', 'data-parameters', inner_keys=['pred-len', 'future-days'])


def gen_data(filename, col_id, add_date=False):
    """
    生成数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param add_date: 是否返回预测那天的日期
    :return: x in shape(m, pred_len), y in shape(m,)
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    dates = data_process.split_date(frame)  # 找出合法的、连续的日期
    predict_dates = []
    x, y = list(), list()

    for ds in dates:
        for d in ds:

            pred_date = d + pd.Timedelta(days=(pred_len + future_days - 1))
            if pred_date > ds[-1]:
                break

            x.append(frame.loc[d: d + pd.Timedelta(days=(pred_len - 1)), col_id])
            y.append(frame.loc[pred_date, col_id])
            predict_dates.append(pred_date)

    if add_date:
        return np.array(x), np.array(y), np.array(predict_dates)
    else:
        return np.array(x), np.array(y)


def load_all(filename):
    """
    加载所有列的数据，并进行划分
    :param filename: 存放数据的 CSV 文件
    :return: 划分好的四个数据集
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_all(filename, cols=frame.columns[1:], load_func=gen_data)
