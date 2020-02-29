from utils import metric


def predict_one_cols(func, data):
    """
    使用给定的模型对所有列的数据进行分别训练和测试
    :param func: 用于预测的模型
    :param data: 用于预测的数据, dict(key=col_name)
    :return: 预测结果的评估指标，list[dict()], 每个列表元素代表一个列的指标（存在字典里面）
    """

    # 对数据和模型进行合法性检查
    if not callable(func):
        raise ValueError('为提供用于预测的方法')
    if type(data) is not dict:
        raise TypeError('数据格式有误')

    result_list = []
    for column in data:
        # 调用模型进行预测，得到预测结果
        pred = func(x_train=data[column][0], y_train=data[column][1], x_test=data[column][2])

        # 将预测结果与测试集进行比较，得到评估指标
        d = {'Column': column}
        metric_dict = metric.all_metric(y=data[column][3], pred=pred.reshape(-1))
        d.update(metric_dict)

        result_list.append(d)

    return result_list
