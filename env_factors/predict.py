"""
采取三种方案来预测
1）所有的数据一起训练和测试
2）所有的数据一起训练，分列进行测试
3）分列进行训练和测试
"""


import utils.pred_utils as pu
from baseline import lr
from baseline import mlp
from baseline import svr
from baseline import xgb
from env_factors import gen_dataset
from utils import data_process
from utils import metric
from utils.config import *


# 存放预测结果的文件路径
result_dir, result2_dir = global_config.get_config('predict-result', inner_keys=['result', 'result2'])


def scheme3(filename):
    """
    方案3：采用多个模型分别对每一列进行训练和测试
    :param filename: 存放数据的文件
    """
    data = gen_dataset.load_cols(filename)

    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        predict_one_cols(func, data, filename)


def supervised_model(filename, func, col=None):
    """
    使用有监督学习的模型进行预测
    :param filename: 存放数据的文件
    :param func: 使用的模型
    :param col: 若为 None 则预测所有的列，否则预测特定的列
    """
    if col:
        x_train, y_train, x_test, y_test, normal_y = gen_dataset.load_one_col(filename, col=None)
    else:
        x_train, y_train, x_test, y_test, normal_y = gen_dataset.load_all(filename)

    pred = func(x_train, y_train, x_test)

    # 反归一化
    pred = normal_y.inverse_transform(pred)
    y_test = normal_y.inverse_transform(y_test)

    print(func.__name__, filename)
    print(metric.all_metric(y_test, pred))


def all_supervised_models(filename, normal=False):
    """
    使用所有的有监督学习模型进行预测
    :param filename: 存放数据的文件
    :param normal: 是否进行归一化
    """
    normal_y = None
    if normal:
        x_train, y_train, x_test, y_test, normal_y = gen_dataset.load_all(filename)
    else:
        x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)

    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        pred = func(x_train, y_train, x_test)

        if normal:  # 反归一化
            pred = normal_y.inverse_transform(pred)
            y_test = normal_y.inverse_transform(y_test)

        print(func.__name__, filename)
        print(metric.all_metric(y_test, pred))


def predict_every_col(filename):
    x_train, y_train, test_data = gen_dataset.load_every_col(filename)

    for func in [lr, svr, xgb, mlp]:
        # 每个模型都训练一遍
        print('模型：', func.__name__)

        result_list = []
        model = func.model_fit(x_train, y_train)

        for key in test_data:
            pred = func.predict(model, x_test=test_data[key][0])

            d = {'Column': key}
            metric_dict = metric.all_metric(y=test_data[key][1], pred=pred.reshape(-1))
            d.update(metric_dict)
            result_list.append(d)

        # 写到 CSV 文件里
        csv_name = func.__name__.split('.')[-1] + '_' + filename.split('\\')[-1]
        data_process.dump_csv(result_dir, csv_name, result_list)
        print('完成预测，已写入', csv_name)


def predict_one_cols(func, data, filename):
    """
    用给定的模型对每一列的数据分别进行预测
    :param func: 使用的模型
    :param data: 预测使用的数据，格式为字典
    :param filename: 存放数据的文件
    """
    print(f'模型: {func.__name__}')

    # 进行训练，得到每一列数据的预测指标
    cols_metrics = pu.predict_one_cols(func, data)

    # 把指标写到 CSV 文件里
    csv_name = func.__name__.split('_')[0] + '_' + filename.split('/')[-1]
    data_process.dump_csv(result2_dir, csv_name, cols_metrics)
    print('完成预测，已写入', csv_name)


if __name__ == '__main__':
    pred_target = global_config.get_config('predict-target')
    pred_target_filename = get_config('../data/data.json', pred_target, 'server')
    scheme3(pred_target_filename)
