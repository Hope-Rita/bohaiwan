import json
import numpy as np
import pandas as pd
from utils import normalization
import utils.load_utils as ld
from utils.config import Config


conf = Config()
col_match_path = conf.get_data_loc('col-match')
pressure = conf.get_data_loc('pressure')
weather = conf.get_data_loc('weather')
waterline = conf.get_data_loc('waterline')
temperature = conf.get_data_loc('temperature')

valid_dates = None


def gen_data(filename, col_id, add_date=False):
    """
    生成对应列号的数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param add_date: 是否返回预测那天的日期
    :return: numpy array
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    pressure_frame = pd.read_csv(pressure, parse_dates=True, index_col='date')
    weather_frame = pd.read_csv(weather, parse_dates=True, index_col='date')
    waterline_frame = pd.read_csv(waterline, parse_dates=True, index_col='date')
    temperature_frame = pd.read_csv(temperature, parse_dates=True, index_col='date')

    # 取出 JSON 字典
    f = open(col_match_path, encoding='utf-8')
    matched_cols = json.load(f)
    f.close()

    # 获取合法的日期
    global valid_dates
    if not valid_dates:
        valid_dates = get_valid_dates(frame, temperature_frame, pressure_frame)

    predict_dates = []
    x, y = list(), list()

    for date in valid_dates:
        y.append(frame.loc[date, col_id])

        tmp = list()
        tmp.append(get_matched_val(temperature_frame, matched_cols, col_id, 'temperature', date))
        tmp.append(get_matched_val(pressure_frame, matched_cols, col_id, 'pressure', date))
        # tmp.append(weather_frame.loc[date, 'high_tp'])
        # tmp.append(weather_frame.loc[date, 'low_tp'])
        # tmp.append(waterline_frame.loc[date, 'waterline'])
        x.append(tmp)
        predict_dates.append(date)

    if add_date:
        return np.array(x), np.array(y), np.array(predict_dates)
    else:
        return np.array(x), np.array(y)


def load_all(filename):
    """
    把所有合法列的数据都载入，并划分训练集和测试集（采用随机选取的方式）
    :param filename: 数据所在的文件
    :return: 划分好的训练集和测试集，numpy 数组
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_all(filename, cols=get_valid_cols(frame.columns[1:]), load_func=gen_data, random_pick=True)


def load_cols(filename):
    """
    分别导入各列数据，并进行划分（随机选取）
    :param filename: 存放数据的 csv 文件路径
    :return: dict(numpy.ndarray) key 是列名
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_cols(filename, cols=get_valid_cols(frame.columns[1:]), load_func=gen_data, random_pick=True)


def load_one_col(filename, col, add_date=False):
    if add_date:
        x, y, dates = gen_data(filename, col, add_date)
    else:
        x, y = gen_data(filename, col)

    x, y, normal_y = feature_normalization(x, y)

    train_size = int(0.7 * len(x))
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:], normal_y


def load_one_col_not_split(filename, col, add_date=False):
    """
    分别导入各列数据，不进行划分训练集和测试集
    :param filename: 存放数据的 CSV 文件
    :param col: 指定的列号
    :param add_date: 是否增加日期序列
    """
    return ld.load_one_col(filename, col, load_func=gen_data, add_date=add_date, split=False)


def feature_normalization(x, y):
    # 要求 x 和 y 都是 numpy 数组

    for i in range(x.shape[1]):
        print(f'对第x[{i}]列进行归一化')
        normal_x = normalization.MinMaxNormal(x[:, i])
        x[:, i] = normal_x.transform(x[:, i])

    print('对y进行归一化')
    normal_y = normalization.MinMaxNormal(y)
    y = normal_y.transform(y)

    return x, y, normal_y


def get_matched_val(frame, json_dict, col, key, date):
    """
    获取数据表 frame 中与列 col 相匹配的列（可能是一个或多个）的值
    :param frame: 查找值的数据表
    :param json_dict: 从 JSON 文件中取得的字典
    :param key: 匹配的表项，即为 JSON 文件中的第二层 key
    :param date: 取值那天的日期
    :return: 如果匹配一个列就返回那个列的值，如果有多个列的话就取平均值
    """
    cols = json_dict[col][key]
    return sum([frame.loc[date, c] for c in cols]) / len(cols)


def get_valid_cols(cols):
    """
    从 JSON 文件中取出温度和水压都能匹配上的列
    :param cols: 所有的列
    :return: 合法的列
    """
    f = open(col_match_path, encoding='utf-8')
    d = json.load(f)
    f.close()

    valid_cols = []
    for col in cols:
        if d[col]['temperature'] and d[col]['pressure']:
            valid_cols.append(col)

    return valid_cols


def get_valid_dates(frame, tf, pf):
    """
    获取 frame 中可用的日期
    :param frame: 待处理的 DataFrame
    :param tf: temperature
    :param pf: pressure
    :return: 合法的日期
    """
    vd = list()
    for day in pd.date_range(frame.index[0], frame.index[-1], freq='D'):
        if day in tf.index and day in pf.index:
            vd.append(day)

    return vd
