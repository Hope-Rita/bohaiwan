from utils import metric
from utils import data_process
from utils import normalization


def predict_one_cols(func, data):
    """
    使用给定的模型对所有列的数据进行分别训练和测试
    :param func: 用于预测的模型
    :param data: 用于预测的数据, dict(key=col_name)
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

        # 普通模型的数据归一化，对 x_train 和 x_test 分别按列进行归一化
        # x_train = data_process.col_normalization(x_train)
        # x_test = data_process.col_normalization(x_test)
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
