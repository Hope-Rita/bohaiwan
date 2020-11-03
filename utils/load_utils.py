import numpy as np
import random
from tqdm import tqdm


def load_all(filename, cols, load_func, random_pick=False):
    """
    公用函数，导入所有列的数据，并进行划分
    :param filename: 存放数据的 CSV 文件
    :param cols: 需要载入的列
    :param load_func: 使用这个函数来载入数据
    :param random_pick: 是否随机选取
    :return: 划分好的训练集和测试集
    """
    if not callable(load_func):
        raise ValueError('未提供载入数据的方法')

    x_train, y_train = list(), list()
    x_test, y_test = list(), list()

    print(f'开始从{filename}载入数据')
    t = tqdm(cols)
    for col in t:
        t.set_description(f'处理列 {col}')
        x, y = load_func(filename, col)

        splited_data = dataset_split(x, y, random_pick=random_pick)
        x_train.extend(splited_data[0])
        y_train.extend(splited_data[1])
        x_test.extend(splited_data[2])
        y_test.extend(splited_data[3])

    print('数据载入完毕\n')
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def load_every_col(filename, cols, load_func, random_pick=False):
    """
    载入训练集和测试集，训练集是整体的，测试集是分列的
    :param filename: 存放数据的 csv 文件路径
    :param cols: 要载入的列
    :param load_func: 载入数据使用的函数
    :param random_pick: 是否随机选取
    :return: x_train, y_train 是 numpy 数组, test_data 是 dict((x_test, y_test))
    """
    if not callable(load_func):
        raise ValueError('未提供载入数据的方法')

    x_train, y_train = [], []
    test_data = {}

    print(f'开始从{filename}载入数据')
    t = tqdm(cols)
    for col in t:
        t.set_description(f'处理列 {col}')
        x, y = load_func(filename, col)

        split_data = dataset_split(x, y, random_pick=random_pick, return_numpy=False)
        # 放入总体的训练集
        x_train.extend(split_data[0])
        y_train.extend(split_data[1])
        # 放入各列的测试集
        test_data[col] = (np.array(split_data[2]), np.array(split_data[3]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 训练集打乱一下
    tmp = np.concatenate((x_train, y_train.reshape((len(y_train), 1))), axis=1)
    np.random.shuffle(tmp)
    x_train, y_train = np.split(tmp, [tmp.shape[1] - 1], axis=1)

    print('已生成数据集\n')
    return x_train, y_train, test_data


def load_cols(filename, cols, load_func, random_pick=False):
    """
    分别载入所有的列，并划分训练集和测试集
    :param filename: 存放数据的 csv 文件
    :param cols: 要加载的列
    :param load_func: 加载数据使用的函数
    :param random_pick: 是否随机选取
    :return: dict(key=col_name, val=(tuple of 4 numpy.array))
    """
    if not callable(load_func):
        raise ValueError('未提供载入数据的方法')

    data = {}

    print(f'开始从{filename}载入数据')
    t = tqdm(cols)
    for col in t:
        t.set_description(f'处理列 {col}')
        x, y = load_func(filename, col)
        data[col] = dataset_split(x, y, random_pick=random_pick)

    print('已生成数据集\n')
    return data


def load_one_col(filename, col, load_func, random_pick=False, add_date=False, split=True, normalize=True):
    """
    根据列号，加载指定的列
    :param filename: 存放数据的 csv 文件
    :param col: 要加载的列
    :param load_func: 加载数据使用的函数
    :param random_pick: 是否随机选取
    :param add_date: 是否返回日期序列
    :param split: 是否划分训练集和测试集
    :param normalize: 是否对数据集进行归一化
    :return: 四个数据集
    """
    if not callable(load_func):
        raise ValueError('未提供载入数据的方法')

    print(f'开始从{filename}载入数据')
    # 加载数据集
    loaded_data = load_func(filename, col, add_date, normalize)

    if split:
        # 划分训练集和测试集
        data = dataset_split(loaded_data[0], loaded_data[1], random_pick=random_pick)
    else:
        # 不划分
        data = (loaded_data[0], loaded_data[1])

    print('已生成数据集\n')
    if add_date:
        return data + (loaded_data[2], )  # 把数据集和日期放在一块返回
    else:
        return data


def dataset_split(x, y, rate=0.8, random_pick=False, return_numpy=True):
    """
    将数据集分成训练集和测试集
    :param x: 样本集
    :param y: 结果集
    :param rate: 训练集占比
    :param random_pick: 是否随机选取
    :param return_numpy: 是否返回 numpy 数组
    :return: 元组，由四个数据集组成
    """
    if len(x) != len(y):
        raise ValueError('x, y 长度不一致')

    if random_pick:
        x_train, y_train, x_test, y_test = random_split(x, y, rate)
    else:
        train_size = int(rate * len(x))
        x_train = x[:train_size]
        y_train = y[:train_size]
        x_test = x[train_size:]
        y_test = y[train_size:]

    if return_numpy:
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    else:
        return x_train, y_train, x_test, y_test


def random_split(x, y, rate=0.8):
    """
    随机划分训练集和测试集
    :param x: 样本集
    :param y: 结果集
    :param rate: 测试集占比
    :return: 格式为 List
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(len(x)):
        if random.random() < rate:
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])

    return x_train, y_train, x_test, y_test
