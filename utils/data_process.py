import os
import pandas as pd
from utils import normalization


def split_date(frame):
    # 按日期分割，从缺失的日期处截断
    res, now = list(), list()
    for day in pd.date_range(frame.index[0], frame.index[-1], freq='D'):
        if day not in frame.index:
            if len(now) > 0:
                res.append(now)
            now = list()
        else:
            now.append(day)
    else:  # 最后剩的没有断的那一条连续日期
        if len(now) > 0:
            res.append(now)

    return res


def dump_csv(dirname, filename, data, average_func=None):
    """
    把统计结果的 DataFrame 写到 CSV 文件中
    :param dirname: 路径名
    :param filename: 写入的文件名
    :param data: 写入的数据，格式为 List[dict]
    :param average_func: 求均值的方法
    """

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    df = pd.DataFrame(data)
    filename = os.path.join(dirname, filename)
    df.to_csv(filename, index=False)
    print('完成预测，已写入', filename)

    if average_func:
        # 统计总体的平均 RMSE, MAE 和 PCC
        print('RMSE', average_func(df.loc[:, 'RMSE']))
        print('MAE', average_func(df.loc[:, 'MAE']))
        print('MAPE', average_func(df.loc[:, 'MAPE']))
        print('PCC', average_func(df.loc[:, 'PCC']))

    print()


def dump_pred_result(dirname, filename, y, pred, date):
    """
    把预测得到的数据和真实数据写入到 CSV 文件中
    :param dirname: 存放文件的目录
    :param filename: 存放数据的文件名
    :param y: 真实值
    :param pred: 预测值
    :param date: 日期，这里是所有的日期，要截断
    """

    if y is None:
        df = pd.DataFrame({'value': pred}, index=date)
    else:
        df = pd.DataFrame({'truth': y, 'predict': pred}, index=date[-len(y):])

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    df.to_csv(os.path.join(dirname, filename), index_label='date')


def avg(series):
    """
    计算一个序列的平均值
    :param series: 序列，格式为 pandas.Series
    :return: 把空值去掉之后的平均值
    """
    series = series.dropna()
    return sum(series) / len(series)


def col_normalization(data):
    """
    对样本集的每一列进行归一化
    :param data: 类型为 numpy 数组
    :return: 归一化后的样本集
    """
    return col_normalization_with_normalizer(data)[0]


def col_normalization_with_normalizer(data):
    """
    对样本集的每一列进行归一化
    :param data: 类型为 numpy 数组
    :return: 归一化后的样本集和各列的 normalizers
    """
    normalizers = []
    for i in range(data.shape[1]):
        x = data[:, i]
        normal_x = normalization.MinMaxNormal(x)
        data[:, i] = normal_x.transform(x)
        normalizers.append(normal_x)

    return data, normalizers


def section_normalization(data):
    """
    对样本集的每一个切面进行归一化
    :param data: 类型为 numpy 数组，形状为 (m, col_num, p + k)
    :return: 归一化后的样本集
    """
    return section_normalization_with_normalizer(data)[0]


def section_normalization_with_normalizer(data):
    """
    对样本集的每一个切面进行归一化
    :param data: 类型为 numpy 数组, 形状为 (m, col_num, p + k)
    :return: 归一化后的样本集和各个切面的 normalizers
    """
    normalizers = []
    for i in range(data.shape[2]):
        x = data[:, :, i]
        normal_x = normalization.MinMaxNormal(x)
        data[:, :, i] = normal_x.transform(x)
        normalizers.append(normal_x)

    return data, normalizers


def col_transform(data, normalizers):
    """
    根据 normalizers 对每一列数据进行转换
    :param data: 类型为 numpy 数组
    :param normalizers: 转换器
    :return: 转换好的数据
    """
    for i in range(data.shape[1]):
        data[:, i] = normalizers[i].transform(data[:, i])

    return data
