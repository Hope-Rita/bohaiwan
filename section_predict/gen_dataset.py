import numpy as np
import pandas as pd
import utils.load_utils as ld
from utils.config import Config
from utils import data_process
from union_predict.gen_dataset import gen_data


conf = Config()


weather = conf.get_data_loc('weather')
waterline = conf.get_data_loc('waterline')
pred_len, future_days, env_factor_num = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-days', 'env-factor-num'])
print(f'从{conf.path}载入gen_dataset模块, pred: {pred_len}, future: {future_days}, env: {env_factor_num}')


def gen_section_data(filename, section_name, normalize=True):
    """
    生成某一 section 的数据集
    :param filename: 数据来源的文件
    :param section_name: 段号
    :param normalize: 是否进行归一化
    :return: x 是该 section 上所有列的时间序列加上环境因素， y 是 col_id 列在预测那天的数值
             x in shape(m, col_num, pred_len + env_factor_len), y in shape(m, col_num)
    """

    frame = pd.read_csv(filename, parse_dates=True, index_col='date')
    cols = get_section_members(section_name, frame.columns)
    x_section = []
    y_section = []

    for col in cols:
        x, y = gen_data(filename, col, normalize=False)
        x_section.append(x)
        y_section.append(y)

    x_section = np.array(x_section)
    y_section = np.array(y_section)
    x_section = x_section.swapaxes(0, 1)  # 使 x 的形状为 (m, col_num, p + k)
    y_section = y_section.T

    if normalize:
        x_section = data_process.section_normalization(x_section)
    return x_section, y_section, cols


def load_one_section(filename, section_name):
    """
    加载一个 section 的数据集
    :param filename: 存放数据的文件
    :param section_name: section 名称，格式为 SXX
    :return: tuple of 4 datasets
    """
    print(f'从 {filename} 中加载数据')
    x_section, y_section, section_members = gen_section_data(filename, section_name)
    return ld.dataset_split(x_section, y_section) + (section_members,)


def get_section_members(section_name, cols):
    """
    给定一个 section，取出在这个 section 所有成员
    :param section_name: section 的名字，格式为 Sxx
    :param cols: 所有的列
    :return: 该 section 上所有成员的列表
    """
    return [col for col in cols if section_name in col]


def get_section_sensor_num(filename, section_name):
    """
    得到某个 section 上传感器的数量
    @param filename: 存储数据的文件名
    @param section_name: 块号
    @return: 传感器的数量
    """
    df = pd.read_csv(filename, index_col='date')
    return len([col for col in df.columns if section_name in col])
