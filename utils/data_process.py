import pandas as pd


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
    else:
        if len(now) > 0:
            res.append(now)

    return res


def dump_csv(dirname, filename, data, avg=None):
    """
    把统计结果的 DataFrame 写到 CSV 文件中
    :param dirname: 路径名
    :param filename: 写入的文件名
    :param data: 写入的数据，格式为 List[dict]
    :param avg: 求均值的方法
    """
    df = pd.DataFrame(data)
    filename = dirname + '/' + filename
    df.to_csv(filename, index=False)
    print('完成预测，已写入', filename)

    if avg:
        # 统计总体的平均 RMSE, MAE 和 PCC
        print('RMSE', avg(df.loc[:, 'RMSE']))
        print('MAE', avg(df.loc[:, 'MAE']))
        print('PCC', avg(df.loc[:, 'PCC']))

    print()


def avg(series):
    """
    计算一个序列的平均值
    :param series: 序列，格式为 pandas.Series
    :return: 把空值去掉之后的平均值
    """
    series = series.dropna()
    return sum(series) / len(series)
