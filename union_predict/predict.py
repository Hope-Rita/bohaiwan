import numpy as np
import platform

from utils.config import Config
config_path = '../union_predict/config.json'
conf = Config(config_path)

import utils.pred_utils as pu
from baseline import lr
from baseline import mlp
from baseline import recurrent
from baseline import svr
from baseline import xgb
from baseline import rf
from baseline import knn
from union_predict import gen_dataset
from utils import data_process
from utils import draw_pic
from utils import metric


pred_res_dir = None


def classical_models(filename):
    """
    对四个经典的模型，分列进行训练和测试
    :param filename: 存放数据的 CSV 文件路径
    """

    # 加载数据，格式为 dict(key=col_name, value=tuple(data))
    data = gen_dataset.load_cols(filename)

    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        predict_one_cols(func, data, filename)


def predict_one_cols(func, data, filename):
    """
    用给定的模型对每一列的数据分别进行预测
    :param func: 使用的模型
    :param data: 预测使用的数据，格式为字典
    :param filename: 存放数据的文件
    """
    print('模型：', func.__name__, 'future_days:', gen_dataset.future_days)

    # 进行训练，得到每一列数据的预测指标
    cols_metrics = pu.predict_one_cols(func, data)

    # 写入 CSV 文件。系统不同，处理方式不一样
    csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_'
    csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]

    data_process.dump_csv(pred_res_dir, csv_name, cols_metrics, average_func=data_process.avg)


def predict_all_data(func, filename):
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)

    pred = func(x_train, y_train, x_test)
    print(func.__name__, filename, gen_dataset.future_days)
    print(metric.all_metric(y_test, pred))


def analysis_all_cols(filename, func):
    """
    对每一列传感器进行分析，得到较为详细的数据，并绘制图像
    :param filename: 存放数据的文件
    :param func: 使用的模型
    """
    cols = gen_dataset.get_all_col_name(filename)
    for col in cols:
        predict_one_col(filename, col, func)


def cross_validation(filename, func, k=10):
    """
    对每一列传感器进行多折交叉验证，输出各列的值，并绘制图像
    :param filename: 存放数据的文件
    :param func: 使用的预测模型
    :param k: 折数
    """
    cols = gen_dataset.get_all_col_name(filename)
    col_metrics = []
    for col in cols:
        print('当前列：', col)
        metric_dict = one_col_cross_validation(filename, col, func, k)
        metric_dict['Column'] = col
        col_metrics.append(metric_dict)
        print()

    # 写入 CSV 文件。系统不同，处理方式不一样
    csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_'
    csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]

    data_process.dump_csv(f'kflod_valid/{func.__name__}', csv_name, col_metrics, average_func=data_process.avg)


def predict_one_col(filename, col, func, is_draw_pic=True):
    """
    使用指定的模型对某一列的数据进行预测, 用于对存在异常的数据进行检查测试
    :param filename: 存放数据的 CSV 文件
    :param col: 预测的列号
    :param func: 预测用的模型
    :param is_draw_pic: 是否绘制图像
    """
    x_train, y_train, x_test, y_test, dates = gen_dataset.load_one_col(filename, col, add_date=True)
    pred = func(x_train, y_train, x_test)
    print(col + ':', metric.all_metric(y_test, pred))
    data_process.dump_pred_result(f'onecol_pred_result/{func.__name__}/metrics', f'{col}.csv', y_test, pred, dates)

    # 画个比较图
    if is_draw_pic:
        draw_pic.compare(y=np.concatenate((y_train, y_test)),
                         pred=np.concatenate((y_train, pred)),
                         save_path={'dir': f'onecol_pred_result/{func.__name__}/pics', 'filename': f'{col}.jpg'},
                         title_info=(func.__name__ + ' ' + col)
                         )


def one_col_cross_validation(filename, col, func, k=10, is_draw_pic=True):
    """
    对某一列传感器的数据进行多折交叉验证
    :param filename: 存放数据的文件名
    :param col: 当前列号
    :param func: 使用的预测模型
    :param k: 折数
    :param is_draw_pic: 是否绘制图像
    :return: 这列的预测指标
    """
    x, y, date = gen_dataset.load_one_col_not_split(filename, col, add_date=True)
    return pu.one_col_cross_validation((x, y), date, func, k, is_draw_pic,
                                       csv_loc={
                                           'dir': f'kflod_valid/{func.__name__}/vals',
                                           'filename': f'{col}.csv'
                                       },
                                       pic_info={
                                           'dir': f'kflod_valid/{func.__name__}/pics',
                                           'filename': f'{col}.jpg',
                                           'title': col
                                       })


if __name__ == '__main__':
    # 存放预测结果文件的路径
    pred_res_dir = conf.get_config('predict-result', 'server')

    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)
    pred_col = conf.get_config('predict-col')

    # cross_validation(pred_target_filename, lr.lr_predict)
    # one_col_cross_validation(pred_target_filename, pred_col, lr.lr_predict)
    # analysis_all_cols(pred_target_filename, recurrent.rnn_union_predict)
    predict_one_col(pred_target_filename, pred_col, recurrent.rnn_union_predict, is_draw_pic=False)
    # target_data = gen_dataset.load_cols(pred_target_filename, random_pick=False)
    # predict_one_cols(recurrent.rnn_union_predict, target_data, pred_target_filename)
    # classical_models(pred_target_filename)
