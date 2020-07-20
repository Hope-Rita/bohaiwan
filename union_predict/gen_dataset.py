import numpy as np
import pandas as pd
import utils.load_utils as ld
from utils.config import Config
from utils import data_process


conf = Config()
# 加载环境数据的路径
weather = conf.get_data_loc('weather')
waterline = conf.get_data_loc('waterline')
# 加载运行配置
pred_len, future_days, env_factor_num = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-days', 'env-factor-num'])
# 环境变量加载配置
add_high_tp, add_low_tp, add_waterline = \
    conf.get_config('env-factor-load', inner_keys=['high-tp', 'low-tp', 'waterline'])
# 检查环境因素个数是否一致
if env_factor_num != add_high_tp + add_low_tp + add_waterline:
    raise ValueError('env-factor 配置出现矛盾')
print(f'\n配置文件：{conf.path}，载入gen_dataset模块, pred: {pred_len}, future: {future_days}, env: {env_factor_num}')


def gen_data(filename, col_id, add_date=False, normalize=True):
    """
    生成一个列的数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param add_date: 是否返回预测那天的日期
    :param normalize: 是否对数据进行归一化
    :return: x in shape(m, pred_len + env_factor_len), y in shape(m,)
    """
    return produce_dataset(filename, col_id, add_date=add_date, normalize=normalize)


def produce_dataset(filename, col_id, add_date, normalize):
    """
    根据不同的输入生成数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param add_date: 是否返回预测那天的日期
    :param normalize: 是否对
    :return: 返回根据要求生成的 x 和 y， 若 add_date 为真， 则加上预测日期的序列
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    weather_frame = pd.read_csv(weather, parse_dates=True, index_col='date')
    waterline_frame = pd.read_csv(waterline, parse_dates=True, index_col='date')

    dates = data_process.split_date(frame)  # 找出合法的、连续的日期
    predict_dates = []
    x, y = list(), list()

    for ds in dates:
        for d in ds:

            pred_date = d + pd.Timedelta(days=(pred_len + future_days - 1))
            if pred_date > ds[-1]:
                break

            # 放入时间序列
            series = frame.loc[d: d + pd.Timedelta(days=(pred_len - 1)), col_id].to_list()
            # 放入外部特征
            if add_high_tp:
                series.append(weather_frame.loc[pred_date, 'high_tp'])
            if add_low_tp:
                series.append(weather_frame.loc[pred_date, 'low_tp'])
            if add_waterline:
                series.append(waterline_frame.loc[pred_date, 'waterline'])

            x.append(series)
            y.append(frame.loc[pred_date, col_id])
            predict_dates.append(pred_date)

    if normalize:
        x = data_process.col_normalization(np.array(x))
    if add_date:
        return x, np.array(y), np.array(predict_dates)
    else:
        return x, np.array(y)


def load_one_col(filename, col, add_date=False, random_pick=False):
    """
    载入给定列的数据
    :param filename: 存放数据的 CSV 文件
    :param col: 指定的列号
    :param add_date: 是否增加日期序列
    :param random_pick: 在划分数据集的时候是否随机选取
    :return: 四个数据集
    """
    return ld.load_one_col(filename, col, load_func=gen_data, add_date=add_date, random_pick=random_pick)


def load_one_col_not_split(filename, col, add_date=False):
    """
    分别导入各列数据，不进行划分训练集和测试集
    :param filename: 存放数据的 CSV 文件
    :param col: 指定的列号
    :param add_date: 是否增加日期序列
    """
    return ld.load_one_col(filename, col, load_func=gen_data, add_date=add_date, split=False)


def future_dataset(filename, col):
    train_x, train_y, train_date \
        = ld.load_one_col(filename, col, load_func=gen_data, add_date=True, split=False, normalize=False)
    data_frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    weather_frame = pd.read_csv(weather, parse_dates=True, index_col='date')
    waterline_frame = pd.read_csv(waterline, parse_dates=True, index_col='date')

    test_x = []
    test_date = []
    for day in pd.date_range(data_frame.index[-future_days], data_frame.index[-1], freq='D'):
        series = data_frame.loc[day - pd.Timedelta(days=(pred_len - 1)): day, col].to_list()
        # 放入外部特征
        pred_date = day + pd.Timedelta(days=future_days)
        if add_high_tp:
            series.append(weather_frame.loc[pred_date, 'high_tp'])
        if add_low_tp:
            series.append(weather_frame.loc[pred_date, 'low_tp'])
        if add_waterline:
            series.append(waterline_frame.loc[pred_date, 'waterline'])

        test_x.append(series)
        test_date.append(pred_date)

    # 对 x 进行列归一化
    tmp = data_process.col_normalization(np.concatenate((train_x, test_x)))
    train_x = tmp[:len(train_x)]
    test_x = tmp[len(train_x):]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.concatenate((train_date, test_date))


def load_cols(filename, random_pick=False):
    """
    分别导入各列数据，并进行划分
    :param filename: 存放数据的 csv 文件路径
    :param random_pick: 是否随机选取数据集
    :return: dict(numpy.ndarray) key 是列名
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_cols(filename, cols=frame.columns, load_func=gen_data, random_pick=random_pick)


def load_all(filename):
    """
    载入所有列，并划分训练集和测试集
    :param filename: 存放数据的 csv 文件路径
    :return: 四个 numpy 数组
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_all(filename, cols=frame.columns, load_func=gen_data)


def get_all_col_name(filename):
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return frame.columns
