import numpy as np
import pandas as pd
import utils.load_utils as ld
from utils.config import get_config
from utils import data_process


config_path = 'config.json'
weather = get_config('../data/data.json', 'weather', 'server')
waterline = get_config('../data/data.json', 'waterline', 'server')
pred_len, future_days, env_factor_num = get_config(config_path,
                                                   'data-parameters',
                                                   inner_keys=['pred-len', 'future-days', 'env-factor-num']
                                                   )
print(f'从{config_path}载入gen_dataset模块, pred: {pred_len}, future: {future_days}, env: {env_factor_num}')


def gen_data(filename, col_id, add_date=False):
    """
    生成一个列的数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param add_date: 是否返回预测那天的日期
    :return: x in shape(m, pred_len + env_factor_len), y in shape(m,)
    """
    return produce_dataset(filename, col_id, add_date=add_date)


def gen_section_data(filename, col_id, add_date=False):
    """
    生成某一 section 的数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param add_date: 是否返回预测那天的日期
    :return: x 是该 section 上所有列的时间序列加上环境因素， y 是 col_id 列在预测那天的数值
             x in shape(m, pred_len * col_num + env_factor_len), y in shape(m,)
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    neighbors = get_section_neighbors(col_id, frame.columns)  # 与 col_id 同 section 的邻居
    return produce_dataset(filename, col_id, section_neighbors=neighbors, add_date=add_date)


def produce_dataset(filename, col_id, section_neighbors=None, add_date=False):
    """
    根据不同的输入生成数据集
    :param filename: 数据来源的文件
    :param col_id: 列号
    :param section_neighbors: 该列所处的 section 上所有的列号
    :param add_date: 是否返回预测那天的日期
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
            series = []
            if section_neighbors is not None:

                # 检查类型和值
                if type(section_neighbors) is not list:
                    raise TypeError('section_neighbor 必须是列表')
                if len(section_neighbors) <= 0:
                    raise ValueError('section_neighbor 为空列表')

                for col in section_neighbors:
                    series.extend(frame.loc[d: d + pd.Timedelta(days=(pred_len - 1)), col].to_list())
            else:
                series.extend(frame.loc[d: d + pd.Timedelta(days=(pred_len - 1)), col_id].to_list())
            x.append(series)

            # 放入外部特征
            x[-1].extend([
                weather_frame.loc[pred_date, 'low_tp'],
                weather_frame.loc[pred_date, 'high_tp'],
                waterline_frame.loc[pred_date, 'waterline']
            ])

            y.append(frame.loc[pred_date, col_id])
            predict_dates.append(pred_date)

    # 对样本集进行归一化，结果集不需要归一化
    if add_date:
        return data_process.col_normalization(np.array(x)), np.array(y), np.array(predict_dates)
    else:
        return data_process.col_normalization(np.array(x)), np.array(y)


def load_one_col(filname, col, add_date=False, random_pick=False):
    """
    载入给定列的数据
    :param filname: 存放数据的 CSV 文件
    :param col: 指定的列号
    :param add_date: 是否增加日期序列
    :param random_pick: 在划分数据集的时候是否随机选取
    :return: 四个数据集
    """
    return ld.load_one_col(filname, col, load_func=gen_data, add_date=add_date)


def load_cols(filename, random_pick=False):
    """
    分别导入各列数据，并进行划分
    :param filename: 存放数据的 csv 文件路径
    :param random_pick: 是否随机选取数据集
    :return: dict(numpy.ndarray) key 是列名
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_cols(filename, cols=frame.columns, load_func=gen_data, random_pick=random_pick)


def load_every_col(filename):
    """
    载入训练集和测试集，训练集是整体的，测试集是分列的
    :param filename: 存放数据的 csv 文件路径
    :return: x_train, y_train 是 numpy 数组, test_data 是 dict((x_test, y_test))
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_every_col(filename, cols=frame.columns, load_func=gen_data, random_pick=True)


def load_all(filename):
    """
    载入所有列，并划分训练集和测试集
    :param filename: 存放数据的 csv 文件路径
    :return: 四个 numpy 数组
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return ld.load_all(filename, cols=frame.columns, load_func=gen_data)


def load_section(filename, section):
    """
    加载一个 section 中所有列的 section 数据集
    section 数据集指的是，针对给定的 col，与它在同一个 section 上所有列上的时间序列所组成的数据集
    :param filename: 存放数据的文件
    :param section: section 名，格式为 SXX
    :return: dict(key=col_name, value=section_data)
    """
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    section_members = get_section_members(section, frame.columns)
    return ld.load_cols(filename, cols=section_members, load_func=gen_section_data)


def load_full_section():
    pass


def get_all_col_name(filename):
    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    return frame.columns


def get_section_neighbors(col, cols):
    """
    得到某个传感器（列）在同一个 Section 的邻居(包括其本身)
    :param col: 目标列
    :param cols: 所有列
    :return: 邻居列表
    """
    section = col.split('_')[1]
    return [c for c in cols if section in c]


def get_section_members(section_name, cols):
    """
    给定一个 section，取出在这个 section 所有成员
    :param section_name: section 的名字，格式为 Sxx
    :param cols: 所有的列
    :return: 该 section 上所有成员的列表
    """
    return [col for col in cols if section_name in col]
