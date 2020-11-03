import numpy as np
from sklearn.model_selection import KFold

from utils import metric
from utils import data_process
from utils import normalization
from utils import draw_pic


def one_col_cross_validation(data, date, func, k=10, is_draw_pic=True, csv_loc=None, pic_info=None):
    """
    对某一列传感器的数据进行多折交叉验证
    :param data: 用于训练和预测的数据集，可以分成 x 和 y
    :param date: 整个数据集的预测日期，和 y 一一对应
    :param func: 使用的预测模型
    :param k: 折数
    :param is_draw_pic: 是否绘制图像
    :param csv_loc: 存放 csv 文件的路径，dict 类型，包括文件夹名字和文件名字
    :param pic_info: 存放图像的相关信息，dict 类型
    :return: 这列的预测指标
    """
    x, y = data
    pred = np.zeros(y.shape)
    kf = KFold(n_splits=k, shuffle=False)

    for i, (train_index, test_index) in enumerate(kf.split(x), 1):
        print(f'正在训练第{i}折')
        x_train, y_train, x_test = x[train_index], y[train_index], x[test_index]
        pred_temp = func(x_train, y_train, x_test)  # 某一折的预测结果
        pred[test_index] = pred_temp

    col_metric = metric.all_metric(y, pred)
    data_process.dump_pred_result(csv_loc['dir'], csv_loc['filename'], y, pred, date)

    if is_draw_pic:
        draw_pic.compare(y,
                         pred,
                         save_path={'dir': pic_info['dir'], 'filename': pic_info['filename']},
                         title_info=pic_info['title']
                         )

    return col_metric


def predict_one_cols(func, data):
    """
    使用给定的模型对所有列的数据进行分别训练和测试
    :param func: 用于预测的模型
    :param data: 用于预测的数据, dict(key=col_name, val=(tuple of dataset))
    :return: 预测结果的评估指标，list[dict()], 每个列表元素代表一个列的指标（存在字典里面）
    """

    # 对数据和模型进行合法性检查
    if not callable(func):
        raise ValueError('未提供用于预测的方法')
    if type(data) is not dict:
        raise TypeError('数据格式有误')

    result_list = []
    for column in data:

        x_train, y_train, x_test, y_test = data[column]

        if any([name in func.__name__ for name in ['rnn', 'gru', 'lstm']]):
            # Recurrent 模型在训练时打印一下传感器的名字
            print(f'当前列: {column}')

        # x_train 和 x_test 在数据生成阶段就已经按列归一化了，这里对 y 进行归一化
        normal_y = normalization.MinMaxNormal([y_train, y_test])
        y_train = normal_y.transform(y_train)

        # 调用模型进行预测，得到预测结果
        pred = func(x_train=x_train, y_train=y_train, x_test=x_test)

        # 将预测结果与测试集进行比较，得到评估指标
        pred = pred.reshape(-1)
        d = {'Column': column}
        pred = normal_y.inverse_transform(pred)
        metric_dict = metric.all_metric(y=y_test, pred=pred)
        d.update(metric_dict)

        # 预测结果添加到列表中进行汇总
        result_list.append(d)

    return result_list
